from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from app_backend.infer_edf import MODEL_PROFILES, _build_reference_model, _profile, _resolve
from run_multistate_explicit_region_temporal_node_gnn_5x10 import (
    build_group_aggregates,
    build_region_aggregates,
    build_solver,
    build_state_aggregates,
    build_state_region_aggregates,
    build_subject_multistate_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train fixed app inference model artifacts.")
    parser.add_argument("--profiles", type=str, default="clean,targeted_repair,taskonly61_multistate_arch")
    parser.add_argument("--pcc-manifest", type=Path, default=Path("data/processed/graphs_pcc_topk_task/manifest.csv"))
    parser.add_argument("--theta-manifest", type=Path, default=Path("data/processed/graphs_plv_theta_topk_task/manifest.csv"))
    parser.add_argument("--alpha-manifest", type=Path, default=Path("data/processed/graphs_plv_alpha_topk_task/manifest.csv"))
    parser.add_argument("--beta-manifest", type=Path, default=Path("data/processed/graphs_plv_beta_topk_task/manifest.csv"))
    parser.add_argument("--split-csv", type=Path, default=Path("data/splits/task_split_subject_level.csv"))
    parser.add_argument("--window-seconds", type=float, default=8.0)
    parser.add_argument("--step-seconds", type=float, default=8.0)
    parser.add_argument("--max-windows", type=int, default=12)
    parser.add_argument("--max-seconds", type=int, default=120)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--multistate-items-cache-path",
        type=Path,
        default=Path("outputs/cache/multistate_task_ec_eo_features_ms120_w8p0_s8p0_mw12_topk4_v2.pt"),
    )
    return parser.parse_args()


def build_multistate_app_artifact(args: argparse.Namespace) -> dict[str, object]:
    states = ["TASK", "EC", "EO"]
    cache = torch.load(_resolve(args.multistate_items_cache_path), map_location="cpu", weights_only=False)
    items = cache["items"] if isinstance(cache, dict) and "items" in cache else cache
    subject_ids, y, base_x, feature_region_map, feature_group_ids, feature_state_ids, group_keys, state_keys = build_subject_multistate_table(
        items=items,
        states=states,
        eligible_states=states,
        include_pairwise_contrasts=False,
    )
    region_x = build_region_aggregates(base_x, feature_region_map)
    state_region_x = build_state_region_aggregates(base_x, feature_region_map, feature_state_ids, n_states=len(state_keys))
    group_x = build_group_aggregates(base_x, feature_group_ids, n_groups=len(group_keys))
    state_x = build_state_aggregates(base_x, feature_state_ids, n_states=len(state_keys))
    expanded_x = np.concatenate([base_x, region_x, state_region_x, group_x, state_x], axis=1).astype(np.float32)

    solver = build_solver(c_value=0.05, seed=int(args.seed), penalty="l2", class_weight_mode="balanced")
    solver.fit(expanded_x, y)
    return {
        "schema": "app_multistate_model_artifact_v1",
        "model_type": "multistate_app",
        "source_run": "outputs/metrics/runs/multistate_accsel_probe_42_43_v2cache",
        "source_result": "TASK+EC+EO multistate 2-seed probe, balanced accuracy mean 0.925",
        "states": states,
        "eligible_states": states,
        "include_pairwise_contrasts": False,
        "top_k_per_node": 4,
        "window_seconds": float(args.window_seconds),
        "step_seconds": float(args.step_seconds),
        "max_windows": int(args.max_windows),
        "max_seconds": int(args.max_seconds),
        "c": 0.05,
        "penalty": "l2",
        "class_weight": "balanced",
        "seed": int(args.seed),
        "subject_ids": [str(item) for item in subject_ids.tolist()],
        "train_subject_count": int(len(y)),
        "train_label_counts": {
            "normal": int(np.sum(y == 0)),
            "mdd": int(np.sum(y == 1)),
        },
        "n_features": int(expanded_x.shape[1]),
        "feature_region_map": feature_region_map,
        "feature_group_ids": [int(item) for item in feature_group_ids],
        "feature_state_ids": [int(item) for item in feature_state_ids],
        "group_keys": [str(item) for item in group_keys],
        "state_keys": [str(item) for item in state_keys],
        "scaler": solver.named_steps["scaler"],
        "clf": solver.named_steps["clf"],
    }


def main() -> int:
    args = parse_args()
    profiles = [item.strip() for item in args.profiles.split(",") if item.strip()]
    for profile_key in profiles:
        if profile_key not in MODEL_PROFILES:
            raise ValueError(f"Unknown profile: {profile_key}")
        profile_args = argparse.Namespace(**vars(args))
        profile_args.model_profile = profile_key
        profile_args.custom_model_path = None
        profile_args.cache_path = None
        profile_args.rebuild_cache = True
        profile_args.trust_cache = False
        profile_args.top_edges = 28
        profile = _profile(profile_args)
        print(f"training app artifact: {profile_key} -> {profile['cache_path']}", flush=True)
        if profile_key == "taskonly61_multistate_arch":
            payload = build_multistate_app_artifact(args)
        else:
            model = _build_reference_model(profile_args)
            cache_key = {
                "schema": "app_reference_model_artifact_v2",
                "profile": profile_key,
                "training_mode": "full_fit",
                "fit_method": "sklearn_logistic_regression_weighted_star",
                "epochs": None,
                "input_scope": profile.get("input_scope", "TASK"),
                "source_result": profile.get("source_result", ""),
                "split_csv": str(_resolve(args.split_csv)),
                "manifests": {
                    "pcc": str(_resolve(args.pcc_manifest)),
                    "theta": str(_resolve(args.theta_manifest)),
                    "alpha": str(_resolve(args.alpha_manifest)),
                    "beta": str(_resolve(args.beta_manifest)),
                },
            }
            payload = {"cache_key": cache_key, "model": model}
        out_path = _resolve(profile["cache_path"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as handle:
            pickle.dump(payload, handle)
        print(f"saved app artifact: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
