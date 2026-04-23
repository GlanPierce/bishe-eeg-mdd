from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from feature_model_utils import aggregate_fold_results, parse_seed_list, summarize_run_payloads, write_json


def load_base_module():
    module_path = Path("src/__pycache__/train_gnn_dualgraph_multiband_spatiotemporal_cv10.cpython-311.pyc")
    src_dir = str(Path("src").resolve())
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    loader = importlib.machinery.SourcelessFileLoader("spatiotemporal_pyc_multiinit_mod", str(module_path))
    spec = importlib.util.spec_from_loader("spatiotemporal_pyc_multiinit_mod", loader)
    if spec is None:
        raise RuntimeError(f"Unable to load module spec from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)

    orig_load = torch.load

    def load_allow_pickle(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_load(*args, **kwargs)

    torch.load = load_allow_pickle
    mod.torch.load = load_allow_pickle
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repeated 5x10 CV for pyc-backed spatiotemporal GNN multi-init ensembles.")
    p.add_argument("--split-csv", type=Path, default=Path("data/splits/task_split_subject_level.csv"))
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--temporal-hidden", type=int, default=32)
    p.add_argument("--max-seconds", type=int, default=120)
    p.add_argument("--window-seconds", type=float, default=16.0)
    p.add_argument("--step-seconds", type=float, default=8.0)
    p.add_argument("--max-windows", type=int, default=12)
    p.add_argument("--items-cache-path", type=Path, default=Path("outputs/cache/spatiotemporal_items/ms120_w16p0_s8p0_mw12_q0p8_me30_k4.pt"))
    p.add_argument("--pcc-quantile", type=float, default=0.8)
    p.add_argument("--min-edges", type=int, default=30)
    p.add_argument("--top-k-per-node", type=int, default=4)
    p.add_argument("--threshold-mode", type=str, default="fixed", choices=["fixed", "val_opt"])
    p.add_argument("--threshold-default", type=float, default=0.5)
    p.add_argument("--threshold-metric", type=str, default="balanced_accuracy", choices=["balanced_accuracy", "f1", "accuracy"])
    p.add_argument("--threshold-grid-step", type=float, default=0.01)
    p.add_argument("--threshold-grid-low", type=float, default=0.05)
    p.add_argument("--threshold-grid-high", type=float, default=0.95)
    p.add_argument("--calibration", type=str, default="none", choices=["none", "temperature", "platt"])
    p.add_argument("--loss-class-weight", type=str, default="none", choices=["none", "balanced"])
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--init-seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--select-top-k", type=int, default=3)
    p.add_argument("--selection-metric", type=str, default="balanced_accuracy", choices=["balanced_accuracy", "accuracy", "f1"])
    p.add_argument("--fold-ids", type=str, default="", help="Optional comma-separated outer fold ids to run, e.g. 5,7.")
    p.add_argument("--ensemble-space", type=str, default="score", choices=["score", "prob"])
    p.add_argument("--ensemble-reduction", type=str, default="mean", choices=["mean", "median"])
    p.add_argument(
        "--ensemble-weighting",
        type=str,
        default="uniform",
        choices=["uniform", "val_metric", "val_metric_softmax"],
    )
    p.add_argument("--weighting-scale", type=float, default=20.0)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/metrics/runs/spatiotemporal_pyc_multiinit_5x10"))
    p.add_argument("--experiment-name", type=str, default="spatiotemporal_pyc_multiinit_5x10")
    return p.parse_args()


def expected_cache_meta(args: argparse.Namespace) -> dict[str, object]:
    return {
        "split_csv": str(args.split_csv),
        "max_seconds": int(args.max_seconds),
        "window_seconds": float(args.window_seconds),
        "step_seconds": float(args.step_seconds),
        "max_windows": int(args.max_windows),
        "pcc_quantile": float(args.pcc_quantile),
        "min_edges": int(args.min_edges),
        "top_k_per_node": int(args.top_k_per_node),
    }


def load_or_build_items(mod, args: argparse.Namespace):
    if args.items_cache_path.exists():
        cache_obj = torch.load(args.items_cache_path, map_location="cpu", weights_only=False)
        if isinstance(cache_obj, dict) and cache_obj.get("meta") == expected_cache_meta(args) and "items" in cache_obj:
            print(f"Loaded cache: {args.items_cache_path}")
            return cache_obj["items"]

    build_args = SimpleNamespace(
        split_csv=args.split_csv,
        max_seconds=args.max_seconds,
        window_seconds=args.window_seconds,
        step_seconds=args.step_seconds,
        max_windows=args.max_windows,
        items_cache_path=args.items_cache_path,
        pcc_quantile=args.pcc_quantile,
        min_edges=args.min_edges,
        top_k_per_node=args.top_k_per_node,
    )
    items = mod.build_subject_sequences(build_args)
    args.items_cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": expected_cache_meta(args), "items": items}, args.items_cache_path)
    return items


def metric_value(name: str, y_true: np.ndarray, prob_pos: np.ndarray) -> float:
    pred = (prob_pos >= 0.5).astype(np.int64)
    name = str(name).lower()
    if name == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, pred))
    if name == "accuracy":
        return float(accuracy_score(y_true, pred))
    if name == "f1":
        return float(f1_score(y_true, pred))
    raise ValueError(f"Unsupported metric: {name}")


def serialize_args(args: argparse.Namespace, seed: int, seeds: list[int]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    out["seed"] = int(seed)
    out["seeds"] = [int(x) for x in seeds]
    return out


def parse_optional_fold_ids(text: str) -> set[int] | None:
    values = [x.strip() for x in str(text).split(",") if x.strip()]
    if not values:
        return None
    return {int(x) for x in values}


def candidate_weights(selected: list[dict[str, object]], mode: str, scale: float) -> np.ndarray:
    if len(selected) == 1 or str(mode) == "uniform":
        return np.full(len(selected), 1.0 / max(len(selected), 1), dtype=np.float64)

    metrics = np.array([float(x["best_metric"]) for x in selected], dtype=np.float64)
    if str(mode) == "val_metric":
        metrics = np.clip(metrics, 1e-8, None)
        return metrics / metrics.sum()

    if str(mode) == "val_metric_softmax":
        logits = (metrics - float(np.max(metrics))) * float(scale)
        w = np.exp(logits)
        return w / np.sum(w)

    raise ValueError(f"Unsupported ensemble weighting mode: {mode}")


def reduce_candidate_arrays(
    arrays: list[np.ndarray],
    reduction: str,
    weights: np.ndarray | None,
) -> np.ndarray:
    stack = np.stack(arrays, axis=0)
    reduction = str(reduction)
    if reduction == "mean":
        if weights is None:
            return np.mean(stack, axis=0)
        return np.average(stack, axis=0, weights=weights)
    if reduction == "median":
        return np.median(stack, axis=0)
    raise ValueError(f"Unsupported ensemble reduction: {reduction}")


def make_loss_weight(labels: np.ndarray, mode: str, device: torch.device) -> torch.Tensor | None:
    if str(mode) == "none":
        return None
    counts = np.bincount(labels.astype(np.int64), minlength=2).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch_weighted(
    mod,
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weight: torch.Tensor | None,
) -> float:
    label_smoothing = float(getattr(model, "_label_smoothing", 0.0))
    if loss_weight is None and abs(label_smoothing) <= 1e-12:
        return float(mod.train_one_epoch(model, loader, optimizer, device))

    model.train()
    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weight, label_smoothing=label_smoothing)
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        y = batch["y"].to(device)
        optimizer.zero_grad()
        logits, _ = model(batch["seq_windows"])
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        n = int(y.shape[0])
        total_loss += float(loss.item()) * n
        total_n += n
    return total_loss / max(total_n, 1)


def run_one_seed(mod, args: argparse.Namespace, items: list[dict[str, object]], seed: int, device: torch.device) -> dict[str, object]:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    init_seeds = parse_seed_list(args.init_seeds)
    fold_filter = parse_optional_fold_ids(args.fold_ids)

    y_all = np.array([int(x["label"]) for x in items], dtype=np.int64)
    idx_all = np.arange(len(items))
    in_dim = int(items[0]["windows"][0]["pcc"].x.shape[1])
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=int(seed))
    fold_results: list[dict[str, object]] = []

    for fold_no, (trainval_idx, test_idx) in enumerate(skf.split(idx_all, y_all), start=1):
        if fold_filter is not None and fold_no not in fold_filter:
            continue
        trainval_items = [items[i] for i in trainval_idx]
        test_items = [items[i] for i in test_idx]
        trainval_y = np.array([int(x["label"]) for x in trainval_items], dtype=np.int64)
        tr_idx, val_idx = train_test_split(
            np.arange(len(trainval_items)),
            test_size=0.2,
            random_state=int(seed) + fold_no,
            stratify=trainval_y,
        )
        train_items = [trainval_items[i] for i in tr_idx]
        val_items = [trainval_items[i] for i in val_idx]

        train_ds = mod.TemporalDualGraphDataset(train_items)
        val_ds = mod.TemporalDualGraphDataset(val_items)
        test_ds = mod.TemporalDualGraphDataset(test_items)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=mod.collate_fn)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=mod.collate_fn)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=mod.collate_fn)
        train_y = np.array([int(x["label"]) for x in train_items], dtype=np.int64)
        loss_weight = make_loss_weight(train_y, mode=args.loss_class_weight, device=device)

        candidates: list[dict[str, object]] = []
        for init_seed in init_seeds:
            train_seed = (int(init_seed) * 1000) + fold_no
            np.random.seed(train_seed)
            torch.manual_seed(train_seed)

            model = mod.DualGraphMultiBandSpatioTemporalModel(
                in_dim=in_dim,
                hidden_dim=args.hidden_dim,
                temporal_hidden=args.temporal_hidden,
                dropout=args.dropout,
            ).to(device)
            model._label_smoothing = float(args.label_smoothing)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            best_state = None
            best_epoch = 1
            best_metric = -1.0
            for epoch in range(1, args.epochs + 1):
                _ = train_one_epoch_weighted(mod, model, train_loader, optimizer, device, loss_weight)
                y_val, p_val, _, _ = mod.predict(model, val_loader, device)
                val_metric = metric_value(args.selection_metric, y_true=y_val, prob_pos=p_val)
                if val_metric > best_metric:
                    best_metric = val_metric
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if best_state is None:
                continue

            model.load_state_dict(best_state)
            y_val_best, p_val_best, _, s_val_best = mod.predict(model, val_loader, device)
            y_test, p_test, w_test, s_test = mod.predict(model, test_loader, device)
            candidates.append(
                {
                    "init_seed": int(init_seed),
                    "train_seed": int(train_seed),
                    "best_epoch": int(best_epoch),
                    "best_metric": float(best_metric),
                    "y_val": y_val_best,
                    "p_val": p_val_best,
                    "s_val": s_val_best,
                    "y_test": y_test,
                    "p_test": p_test,
                    "s_test": s_test,
                    "w_test": w_test,
                }
            )

        if not candidates:
            raise RuntimeError(f"No candidates trained for seed={seed} fold={fold_no}")

        candidates.sort(key=lambda x: (float(x["best_metric"]), -abs(int(x["best_epoch"]) - (args.epochs // 2))), reverse=True)
        selected = candidates[: max(1, min(int(args.select_top_k), len(candidates)))]
        ens_weights = candidate_weights(selected, mode=args.ensemble_weighting, scale=args.weighting_scale)
        y_val_ref = selected[0]["y_val"]
        y_test_ref = selected[0]["y_test"]
        ens_w_test = reduce_candidate_arrays([x["w_test"] for x in selected], reduction=args.ensemble_reduction, weights=ens_weights)

        if args.ensemble_space == "score":
            ens_score_val = reduce_candidate_arrays([x["s_val"] for x in selected], reduction=args.ensemble_reduction, weights=ens_weights)
            ens_score_test = reduce_candidate_arrays([x["s_test"] for x in selected], reduction=args.ensemble_reduction, weights=ens_weights)
            p_val, p_test, calib_info = mod.calibrate_with_val(
                method=args.calibration,
                y_val=y_val_ref,
                score_val=ens_score_val,
                score_test=ens_score_test,
                seed=int(seed) + fold_no,
            )
        else:
            p_val = reduce_candidate_arrays([x["p_val"] for x in selected], reduction=args.ensemble_reduction, weights=ens_weights)
            p_test = reduce_candidate_arrays([x["p_test"] for x in selected], reduction=args.ensemble_reduction, weights=ens_weights)
            calib_info = {"method": "none_prob_ensemble"}

        if args.threshold_mode == "val_opt":
            test_thr, val_thr_score = mod.select_threshold_from_val(
                y_true=y_val_ref,
                prob_pos=p_val,
                metric=args.threshold_metric,
                step=args.threshold_grid_step,
                low=args.threshold_grid_low,
                high=args.threshold_grid_high,
            )
        else:
            test_thr = float(args.threshold_default)
            val_thr_score = float("nan")

        metrics = mod.calc_metrics(y_test_ref, p_test, threshold=test_thr)
        band_mean = ens_w_test.mean(axis=0)
        fold_results.append(
            {
                "fold": fold_no,
                "n_train": int(len(train_ds)),
                "n_val": int(len(val_ds)),
                "n_test": int(len(test_ds)),
                "selected_init_seeds": [int(x["init_seed"]) for x in selected],
                "selected_best_epochs": [int(x["best_epoch"]) for x in selected],
                "selected_best_metrics": [float(x["best_metric"]) for x in selected],
                "selected_ensemble_weights": [float(x) for x in ens_weights.tolist()],
                "test_threshold": float(test_thr),
                "calibration": calib_info,
                "val_threshold_metric": args.threshold_metric if args.threshold_mode == "val_opt" else None,
                "val_threshold_score": None if args.threshold_mode != "val_opt" else float(val_thr_score),
                "weights_mean": {
                    "pcc": float(band_mean[0]),
                    "theta": float(band_mean[1]),
                    "alpha": float(band_mean[2]),
                    "beta": float(band_mean[3]),
                },
                "test_metrics": metrics,
            }
        )
        print(
            f"seed={seed} fold={fold_no:02d} "
            f"acc={metrics['accuracy']:.4f} "
            f"bal_acc={metrics['balanced_accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'NA'} "
            f"thr={test_thr:.2f} "
            f"seeds={[int(x['init_seed']) for x in selected]}"
        )

    if not fold_results:
        raise RuntimeError(f"No folds were executed for seed={seed}; check --fold-ids.")

    return {
        "schema_version": "eeg_mdd_spatiotemporal_pyc_multiinit_5x10_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "DualGraphMultiBandSpatioTemporalModelMultiInitEnsemble",
        "device": str(device),
        "params": serialize_args(args, seed=seed, seeds=parse_seed_list(args.seeds)),
        "aggregate": aggregate_fold_results(fold_results),
        "fold_results": fold_results,
    }


def main() -> int:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod = load_base_module()
    items = load_or_build_items(mod, args)

    run_payloads: list[dict[str, object]] = []
    for seed in seeds:
        payload = run_one_seed(mod=mod, args=args, items=items, seed=int(seed), device=device)
        out_path = args.out_dir / f"spatiotemporal_pyc_multiinit_seed{seed}_cv10.json"
        write_json(out_path, payload)
        run_payloads.append({**payload, "_out_path": str(out_path)})
        print(f"saved: {out_path}")

    summary = summarize_run_payloads(run_payloads)
    summary_payload = {
        "schema_version": "eeg_mdd_spatiotemporal_pyc_multiinit_5x10_summary_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "DualGraphMultiBandSpatioTemporalModelMultiInitEnsemble",
        "params": {
            "folds": int(args.folds),
            "seeds": [int(x) for x in seeds],
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "temporal_hidden": int(args.temporal_hidden),
            "max_seconds": int(args.max_seconds),
            "window_seconds": float(args.window_seconds),
            "step_seconds": float(args.step_seconds),
            "max_windows": int(args.max_windows),
            "pcc_quantile": float(args.pcc_quantile),
            "min_edges": int(args.min_edges),
            "top_k_per_node": int(args.top_k_per_node),
            "threshold_mode": args.threshold_mode,
            "threshold_default": float(args.threshold_default),
            "threshold_metric": args.threshold_metric,
            "threshold_grid_step": float(args.threshold_grid_step),
            "threshold_grid_low": float(args.threshold_grid_low),
            "threshold_grid_high": float(args.threshold_grid_high),
            "calibration": args.calibration,
            "loss_class_weight": args.loss_class_weight,
            "label_smoothing": float(args.label_smoothing),
            "init_seeds": [int(x) for x in parse_seed_list(args.init_seeds)],
            "select_top_k": int(args.select_top_k),
            "selection_metric": args.selection_metric,
            "fold_ids": sorted(parse_optional_fold_ids(args.fold_ids) or []),
            "ensemble_space": args.ensemble_space,
            "ensemble_reduction": args.ensemble_reduction,
            "ensemble_weighting": args.ensemble_weighting,
            "weighting_scale": float(args.weighting_scale),
        },
        **summary,
    }
    summary_path = args.out_dir / "summary_5x10.json"
    write_json(summary_path, summary_payload)
    print(f"\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
