from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split


def load_base_module():
    module_path = Path("src/__pycache__/train_gnn_dualgraph_multiband_spatiotemporal_cv10.cpython-311.pyc")
    src_dir = str(Path("src").resolve())
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    loader = importlib.machinery.SourcelessFileLoader("spatiotemporal_base_mod", str(module_path))
    spec = importlib.util.spec_from_loader("spatiotemporal_base_mod", loader)
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
    p = argparse.ArgumentParser(description="Snapshot-ensemble 10-fold CV for the updated dual-graph spatiotemporal GNN")
    p.add_argument("--split-csv", type=Path, default=Path("data/splits/task_split_subject_level.csv"))
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temporal-hidden", type=int, default=32)
    p.add_argument("--max-seconds", type=int, default=120)
    p.add_argument("--window-seconds", type=float, default=16.0)
    p.add_argument("--step-seconds", type=float, default=8.0)
    p.add_argument("--max-windows", type=int, default=12)
    p.add_argument("--items-cache-path", type=Path, default=Path("outputs/cache/spatiotemporal_items/ms120_w16p0_s8p0_mw12_q0p8_me30_k4.pt"))
    p.add_argument("--pcc-quantile", type=float, default=0.8)
    p.add_argument("--min-edges", type=int, default=30)
    p.add_argument("--top-k-per-node", type=int, default=4)
    p.add_argument("--ssl-weight", type=float, default=0.0)
    p.add_argument("--ssl-temperature", type=float, default=0.2)
    p.add_argument("--ssl-feature-mask-prob", type=float, default=0.1)
    p.add_argument("--ssl-feature-noise-std", type=float, default=0.01)
    p.add_argument("--ssl-edge-drop-prob", type=float, default=0.1)
    p.add_argument("--threshold-mode", type=str, default="val_opt", choices=["fixed", "val_opt"])
    p.add_argument("--threshold-default", type=float, default=0.5)
    p.add_argument("--threshold-metric", type=str, default="accuracy", choices=["balanced_accuracy", "f1", "accuracy"])
    p.add_argument("--threshold-grid-step", type=float, default=0.01)
    p.add_argument("--threshold-grid-low", type=float, default=0.05)
    p.add_argument("--threshold-grid-high", type=float, default=0.95)
    p.add_argument("--calibration", type=str, default="temperature", choices=["none", "temperature", "platt"])
    p.add_argument("--snapshot-start-epoch", type=int, default=60)
    p.add_argument("--snapshot-period", type=int, default=5)
    p.add_argument("--snapshot-count", type=int, default=5)
    p.add_argument(
        "--selection-metric",
        type=str,
        default="balanced_accuracy",
        choices=["balanced_accuracy", "accuracy", "f1"],
        help="Validation metric used to rank snapshot checkpoints.",
    )
    p.add_argument(
        "--out-path",
        type=Path,
        default=Path("outputs/metrics/runs/snapshot_ensemble_cv10/snapshot_ensemble_cv10.json"),
    )
    p.add_argument("--experiment-name", type=str, default="snapshot_ensemble_cv10")
    return p.parse_args()


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


def aggregate_mean_std(values: list[float | None]) -> dict[str, float]:
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)
    return {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr))}


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
    if args.items_cache_path is not None and args.items_cache_path.exists():
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
    if args.items_cache_path is not None:
        args.items_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"meta": expected_cache_meta(args), "items": items}, args.items_cache_path)
    return items


def serialize_args(args: argparse.Namespace) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            out[key] = str(value)
        else:
            out[key] = value
    return out


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod = load_base_module()

    items = load_or_build_items(mod, args)
    y_all = np.array([int(x["label"]) for x in items], dtype=np.int64)
    idx_all = np.arange(len(items))
    in_dim = int(items[0]["windows"][0]["pcc"].x.shape[1])

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results: list[dict[str, object]] = []

    for fold_no, (trainval_idx, test_idx) in enumerate(skf.split(idx_all, y_all), start=1):
        trainval_items = [items[i] for i in trainval_idx]
        test_items = [items[i] for i in test_idx]
        trainval_y = np.array([int(x["label"]) for x in trainval_items], dtype=np.int64)
        tr_idx, val_idx = train_test_split(
            np.arange(len(trainval_items)),
            test_size=0.2,
            random_state=args.seed + fold_no,
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

        model = mod.DualGraphMultiBandSpatioTemporalModel(
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            temporal_hidden=args.temporal_hidden,
            dropout=args.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        candidates: list[dict[str, object]] = []
        best_state = None
        best_metric = -1.0

        for epoch in range(1, args.epochs + 1):
            _ = mod.train_one_epoch(model, train_loader, optimizer, device)
            y_val, p_val, _, _ = mod.predict(model, val_loader, device)
            val_metric = metric_value(args.selection_metric, y_true=y_val, prob_pos=p_val)
            state_cpu = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if val_metric > best_metric:
                best_metric = val_metric
                best_state = state_cpu

            if epoch >= args.snapshot_start_epoch and ((epoch - args.snapshot_start_epoch) % max(args.snapshot_period, 1) == 0):
                candidates.append({"epoch": epoch, "metric": float(val_metric), "state": state_cpu})

        if best_state is not None:
            candidates.append({"epoch": -1, "metric": float(best_metric), "state": best_state})

        candidates.sort(key=lambda x: (float(x["metric"]), float(x["epoch"])), reverse=True)
        snapshots = candidates[: max(1, int(args.snapshot_count))]

        val_scores_list: list[np.ndarray] = []
        test_scores_list: list[np.ndarray] = []
        test_band_weights_list: list[np.ndarray] = []
        snapshot_epochs: list[int] = []
        snapshot_metrics: list[float] = []

        for snap in snapshots:
            model.load_state_dict(snap["state"])
            y_val_snap, _, _, s_val_snap = mod.predict(model, val_loader, device)
            y_test_snap, _, w_test_snap, s_test_snap = mod.predict(model, test_loader, device)
            val_scores_list.append(s_val_snap)
            test_scores_list.append(s_test_snap)
            test_band_weights_list.append(w_test_snap)
            snapshot_epochs.append(int(snap["epoch"]))
            snapshot_metrics.append(float(snap["metric"]))

        y_val_ref = y_val_snap
        y_test_ref = y_test_snap
        ens_score_val = np.mean(np.stack(val_scores_list, axis=0), axis=0)
        ens_score_test = np.mean(np.stack(test_scores_list, axis=0), axis=0)
        ens_w_test = np.mean(np.stack(test_band_weights_list, axis=0), axis=0)

        p_val, p_test, calib_info = mod.calibrate_with_val(
            method=args.calibration,
            y_val=y_val_ref,
            score_val=ens_score_val,
            score_test=ens_score_test,
            seed=args.seed + fold_no,
        )

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
                "best_snapshot_metric": float(max(snapshot_metrics)),
                "snapshot_epochs": snapshot_epochs,
                "snapshot_metrics": snapshot_metrics,
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
            f"fold={fold_no:02d} "
            f"acc={metrics['accuracy']:.4f} "
            f"bal_acc={metrics['balanced_accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'NA'} "
            f"thr={test_thr:.2f} "
            f"snap={snapshot_epochs}"
        )

    payload = {
        "schema_version": "eeg_mdd_benchmark_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "DualGraphMultiBandSpatioTemporalModelSnapshotEnsemble",
        "device": str(device),
        "params": serialize_args(args),
        "aggregate": {
            "accuracy": aggregate_mean_std([fr["test_metrics"]["accuracy"] for fr in fold_results]),
            "balanced_accuracy": aggregate_mean_std([fr["test_metrics"]["balanced_accuracy"] for fr in fold_results]),
            "f1": aggregate_mean_std([fr["test_metrics"]["f1"] for fr in fold_results]),
            "roc_auc": aggregate_mean_std([fr["test_metrics"]["roc_auc"] for fr in fold_results]),
        },
        "fold_results": fold_results,
    }

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
