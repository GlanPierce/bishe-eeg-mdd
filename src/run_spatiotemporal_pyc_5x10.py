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
from sklearn.model_selection import StratifiedKFold, train_test_split

from feature_model_utils import aggregate_fold_results, parse_seed_list, summarize_run_payloads, write_json


def load_base_module():
    module_path = Path("src/__pycache__/train_gnn_dualgraph_multiband_spatiotemporal_cv10.cpython-311.pyc")
    src_dir = str(Path("src").resolve())
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    loader = importlib.machinery.SourcelessFileLoader("spatiotemporal_pyc_base_mod", str(module_path))
    spec = importlib.util.spec_from_loader("spatiotemporal_pyc_base_mod", loader)
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
    p = argparse.ArgumentParser(description="Repeated 5x10 CV for the pyc-backed spatiotemporal pure GNN.")
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
    p.add_argument("--out-dir", type=Path, default=Path("outputs/metrics/runs/spatiotemporal_pyc_5x10"))
    p.add_argument("--experiment-name", type=str, default="spatiotemporal_pyc_5x10")
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


def serialize_args(args: argparse.Namespace, seed: int, seeds: list[int]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    out["seed"] = int(seed)
    out["seeds"] = [int(x) for x in seeds]
    return out


def run_one_seed(mod, args: argparse.Namespace, items: list[dict[str, object]], seed: int, device: torch.device) -> dict[str, object]:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    y_all = np.array([int(x["label"]) for x in items], dtype=np.int64)
    idx_all = np.arange(len(items))
    in_dim = int(items[0]["windows"][0]["pcc"].x.shape[1])
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=int(seed))
    fold_results: list[dict[str, object]] = []

    for fold_no, (trainval_idx, test_idx) in enumerate(skf.split(idx_all, y_all), start=1):
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

        model = mod.DualGraphMultiBandSpatioTemporalModel(
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            temporal_hidden=args.temporal_hidden,
            dropout=args.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_state = None
        best_val_bal = -1.0
        for _ in range(args.epochs):
            _ = mod.train_one_epoch(model, train_loader, optimizer, device)
            y_val, p_val, _, _ = mod.predict(model, val_loader, device)
            val_bal = float(mod.balanced_accuracy_score(y_val, (p_val >= 0.5).astype(np.int64)))
            if val_bal > best_val_bal:
                best_val_bal = val_bal
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        y_val_best, _, _, s_val_best = mod.predict(model, val_loader, device)
        y_test, _, w_test, s_test = mod.predict(model, test_loader, device)
        p_val_best, p_test, calib_info = mod.calibrate_with_val(
            method=args.calibration,
            y_val=y_val_best,
            score_val=s_val_best,
            score_test=s_test,
            seed=int(seed) + fold_no,
        )
        if args.threshold_mode == "val_opt":
            test_thr, val_thr_score = mod.select_threshold_from_val(
                y_true=y_val_best,
                prob_pos=p_val_best,
                metric=args.threshold_metric,
                step=args.threshold_grid_step,
                low=args.threshold_grid_low,
                high=args.threshold_grid_high,
            )
        else:
            test_thr = float(args.threshold_default)
            val_thr_score = float("nan")

        metrics = mod.calc_metrics(y_test, p_test, threshold=test_thr)
        w_mean = w_test.mean(axis=0)
        fold_results.append(
            {
                "fold": fold_no,
                "n_train": int(len(train_ds)),
                "n_val": int(len(val_ds)),
                "n_test": int(len(test_ds)),
                "best_val_balanced_accuracy": float(best_val_bal),
                "test_threshold": float(test_thr),
                "calibration": calib_info,
                "val_threshold_metric": args.threshold_metric if args.threshold_mode == "val_opt" else None,
                "val_threshold_score": None if args.threshold_mode != "val_opt" else float(val_thr_score),
                "avg_windows_test": float(np.mean([int(x["n_windows"]) for x in test_items])),
                "weights_mean": {
                    "pcc": float(w_mean[0]),
                    "theta": float(w_mean[1]),
                    "alpha": float(w_mean[2]),
                    "beta": float(w_mean[3]),
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
            f"calib={calib_info.get('method', 'none')} "
            f"thr={test_thr:.2f}"
        )

    return {
        "schema_version": "eeg_mdd_spatiotemporal_pyc_5x10_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "DualGraphMultiBandSpatioTemporalModel",
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
        out_path = args.out_dir / f"spatiotemporal_pyc_seed{seed}_cv10.json"
        write_json(out_path, payload)
        run_payloads.append({**payload, "_out_path": str(out_path)})
        print(f"saved: {out_path}")

    summary = summarize_run_payloads(run_payloads)
    summary_payload = {
        "schema_version": "eeg_mdd_spatiotemporal_pyc_5x10_summary_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "DualGraphMultiBandSpatioTemporalModel",
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
        },
        **summary,
    }
    summary_path = args.out_dir / "summary_5x10.json"
    write_json(summary_path, summary_payload)
    print(f"\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
