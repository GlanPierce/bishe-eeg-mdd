from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from feature_model_utils import aggregate_fold_results, summarize_run_payloads, write_json


DEFAULT_BASE_DIR = Path("outputs/metrics/runs/static_feature_weightedstar_5x10_thracc_widethr")
DEFAULT_AUG_DIRS = [
    Path("outputs/metrics/runs/region_temporal_feature_node_gnn_5x10_thracc"),
    Path("outputs/metrics/runs/region_temporal_feature_node_gnn_5x10_scale005"),
    Path("outputs/metrics/runs/region_temporal_feature_node_gnn_5x10_scale0001"),
    Path("outputs/metrics/runs/region_temporal_feature_node_gnn_5x10_lbfgs_scale005"),
    Path("outputs/metrics/runs/region_temporal_nodefeature_gnn_5x10_fullscale"),
    Path("outputs/metrics/runs/region_temporal_nodefeature_gnn_5x10_scale005"),
]


def parse_optional_fold_ids(text: str) -> set[int] | None:
    values = [x.strip() for x in str(text).split(",") if x.strip()]
    if not values:
        return None
    return {int(x) for x in values}


def load_seed_payloads(run_dir: Path) -> dict[int, dict[str, object]]:
    payloads: dict[int, dict[str, object]] = {}
    for path in sorted(run_dir.glob("*seed*_cv10.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        seed = int(payload["params"]["seed"])
        payloads[seed] = payload
    if not payloads:
        raise FileNotFoundError(f"No seed payloads found in {run_dir}")
    return payloads


def fold_score(fold_result: dict[str, object]) -> float:
    for key in ("val_threshold_accuracy", "best_val_score", "best_val_accuracy"):
        value = fold_result.get(key)
        if value is not None:
            return float(value)
    return -1.0


def fold_aux_score(fold_result: dict[str, object]) -> float:
    for key in ("best_val_score", "best_val_accuracy", "val_threshold_accuracy"):
        value = fold_result.get(key)
        if value is not None:
            return float(value)
    return -1.0


def normalize_fold_map(payload: dict[str, object], fold_filter: set[int] | None) -> dict[int, dict[str, object]]:
    out: dict[int, dict[str, object]] = {}
    for fold_result in payload["fold_results"]:
        fold_id = int(fold_result["fold"])
        if fold_filter is not None and fold_id not in fold_filter:
            continue
        out[fold_id] = fold_result
    return out


def choose_fold_result(
    base_result: dict[str, object],
    aug_results: list[tuple[str, dict[str, object]]],
    min_val_margin: float,
    prefer_augmented_on_tie: bool,
) -> tuple[str, dict[str, object], dict[str, float]]:
    candidate_scores: dict[str, float] = {"base": fold_score(base_result)}
    best_name = "base"
    best_result = base_result
    best_score = fold_score(base_result)
    best_aux = fold_aux_score(base_result)

    for aug_name, aug_result in aug_results:
        score = fold_score(aug_result)
        aux = fold_aux_score(aug_result)
        candidate_scores[aug_name] = score
        if score > (best_score + float(min_val_margin)):
            best_name = aug_name
            best_result = aug_result
            best_score = score
            best_aux = aux
            continue
        if abs(score - best_score) <= 1e-12:
            if aux > (best_aux + 1e-12):
                best_name = aug_name
                best_result = aug_result
                best_score = score
                best_aux = aux
                continue
            if prefer_augmented_on_tie and best_name == "base" and abs(aux - best_aux) <= 1e-12:
                best_name = aug_name
                best_result = aug_result
                best_score = score
                best_aux = aux
    return best_name, best_result, candidate_scores


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validation-gated pure-GNN selector over static and region-temporal result payloads.")
    p.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    p.add_argument("--aug-dirs", nargs="*", type=Path, default=DEFAULT_AUG_DIRS)
    p.add_argument("--min-val-margin", type=float, default=0.0)
    p.add_argument("--prefer-augmented-on-tie", action="store_true")
    p.add_argument("--fold-ids", type=str, default="")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/metrics/runs/region_temporal_validation_selector_5x10"))
    p.add_argument("--experiment-name", type=str, default="region_temporal_validation_selector_5x10")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    fold_filter = parse_optional_fold_ids(args.fold_ids)
    base_payloads = load_seed_payloads(args.base_dir)

    aug_payloads: list[tuple[str, Path, dict[int, dict[str, object]]]] = []
    for aug_dir in args.aug_dirs:
        if not aug_dir.exists():
            continue
        aug_name = aug_dir.name
        aug_payloads.append((aug_name, aug_dir, load_seed_payloads(aug_dir)))
    if not aug_payloads:
        raise FileNotFoundError("No augmented result directories were found.")

    seeds = sorted(base_payloads.keys())
    run_payloads: list[dict[str, object]] = []
    overall_branch_counter: Counter[str] = Counter()

    for seed in seeds:
        base_payload = base_payloads[seed]
        base_fold_map = normalize_fold_map(base_payload, fold_filter)
        fold_results: list[dict[str, object]] = []
        seed_branch_counter: Counter[str] = Counter()

        for fold_id in sorted(base_fold_map):
            base_result = base_fold_map[fold_id]
            aug_candidates: list[tuple[str, dict[str, object]]] = []
            for aug_name, aug_dir, payload_map in aug_payloads:
                if seed not in payload_map:
                    continue
                aug_fold_map = normalize_fold_map(payload_map[seed], fold_filter)
                if fold_id not in aug_fold_map:
                    continue
                aug_result = aug_fold_map[fold_id]
                if list(base_result["test_subject_ids"]) != list(aug_result["test_subject_ids"]):
                    raise ValueError(f"Seed {seed} fold {fold_id}: test subject ids mismatch between base and {aug_dir}")
                aug_candidates.append((aug_name, aug_result))

            selected_branch, selected_result, candidate_scores = choose_fold_result(
                base_result=base_result,
                aug_results=aug_candidates,
                min_val_margin=float(args.min_val_margin),
                prefer_augmented_on_tie=bool(args.prefer_augmented_on_tie),
            )
            seed_branch_counter[selected_branch] += 1
            overall_branch_counter[selected_branch] += 1

            fold_results.append(
                {
                    "fold": int(fold_id),
                    "selected_branch": selected_branch,
                    "selected_val_score": float(fold_score(selected_result)),
                    "selected_val_aux_score": float(fold_aux_score(selected_result)),
                    "candidate_val_scores": {k: float(v) for k, v in sorted(candidate_scores.items())},
                    "n_train": int(selected_result["n_train"]),
                    "n_val": int(selected_result["n_val"]),
                    "n_test": int(selected_result["n_test"]),
                    "best_epoch": int(selected_result.get("best_epoch", 1)),
                    "selection_metric": selected_result.get("selection_metric", "threshold_accuracy"),
                    "test_threshold": float(selected_result["test_threshold"]),
                    "test_subject_ids": list(selected_result["test_subject_ids"]),
                    "test_metrics": dict(selected_result["test_metrics"]),
                }
            )

        if not fold_results:
            raise RuntimeError(f"No folds selected for seed={seed}")

        payload = {
            "schema_version": "eeg_mdd_region_temporal_validation_selector_5x10_v1",
            "experiment_name": args.experiment_name,
            "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model": "ValidationGatedPureGNNSelector",
            "selector_params": {
                "base_dir": str(args.base_dir),
                "aug_dirs": [str(x[1]) for x in aug_payloads],
                "min_val_margin": float(args.min_val_margin),
                "prefer_augmented_on_tie": bool(args.prefer_augmented_on_tie),
                "fold_ids": sorted(fold_filter or []),
            },
            "params": {
                "seed": int(seed),
                "seeds": [int(x) for x in seeds],
            },
            "branch_usage": dict(sorted(seed_branch_counter.items())),
            "aggregate": aggregate_fold_results(fold_results),
            "fold_results": fold_results,
        }
        out_path = args.out_dir / f"region_temporal_validation_selector_seed{seed}_cv10.json"
        write_json(out_path, payload)
        run_payloads.append({**payload, "_out_path": str(out_path)})
        print(f"seed={seed} usage={dict(sorted(seed_branch_counter.items()))} saved={out_path}")

    summary = summarize_run_payloads(run_payloads)
    summary_payload = {
        "schema_version": "eeg_mdd_region_temporal_validation_selector_5x10_summary_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "ValidationGatedPureGNNSelector",
        "selector_params": {
            "base_dir": str(args.base_dir),
            "aug_dirs": [str(x[1]) for x in aug_payloads],
            "min_val_margin": float(args.min_val_margin),
            "prefer_augmented_on_tie": bool(args.prefer_augmented_on_tie),
            "fold_ids": sorted(fold_filter or []),
        },
        "overall_branch_usage": dict(sorted(overall_branch_counter.items())),
        **summary,
    }
    summary_path = args.out_dir / "summary_5x10.json"
    write_json(summary_path, summary_payload)
    print(f"\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
