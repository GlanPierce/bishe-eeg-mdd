from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split


def load_flat_module():
    src_dir = str(Path("src").resolve())
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import train_gnn_dualgraph_multiband_flatreadout_cv10 as mod

    return mod


def parse_seed_list(seed_text: str) -> list[int]:
    seeds = [int(x.strip()) for x in str(seed_text).split(",") if x.strip()]
    if not seeds:
        raise ValueError("Seed list must not be empty.")
    return seeds


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Multi-init ensemble 10-fold CV for flatreadout static pure GNN")
    p.add_argument("--pcc-manifest", type=Path, default=Path("data/processed/graphs_pcc_topk_task/manifest.csv"))
    p.add_argument("--theta-manifest", type=Path, default=Path("data/processed/graphs_plv_theta_topk_task/manifest.csv"))
    p.add_argument("--alpha-manifest", type=Path, default=Path("data/processed/graphs_plv_alpha_topk_task/manifest.csv"))
    p.add_argument("--beta-manifest", type=Path, default=Path("data/processed/graphs_plv_beta_topk_task/manifest.csv"))
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42, help="Outer CV split seed.")
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--min-epochs", type=int, default=30)
    p.add_argument("--thr-low", type=float, default=0.3)
    p.add_argument("--thr-high", type=float, default=0.7)
    p.add_argument("--thr-step", type=float, default=0.01)
    p.add_argument("--append-adj-row", action="store_true")
    p.add_argument("--append-node-id", action="store_true")
    p.add_argument("--init-seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--select-top-k", type=int, default=3)
    p.add_argument("--selection-metric", type=str, default="accuracy", choices=["balanced_accuracy", "accuracy", "f1"])
    p.add_argument(
        "--out-path",
        type=Path,
        default=Path("outputs/metrics/runs/flatreadout_multiinit_ensemble_cv10/flatreadout_multiinit_ensemble_cv10.json"),
    )
    p.add_argument("--experiment-name", type=str, default="flatreadout_multiinit_ensemble_cv10")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    init_seeds = parse_seed_list(args.init_seeds)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod = load_flat_module()

    master = mod.build_master_table(args.pcc_manifest, args.theta_manifest, args.alpha_manifest, args.beta_manifest)
    y_all = master["label"].to_numpy(dtype=np.int64)
    idx_all = np.arange(len(master))

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results: list[dict[str, object]] = []
    for fold_no, (trainval_idx, test_idx) in enumerate(skf.split(idx_all, y_all), start=1):
        trainval_df = master.iloc[trainval_idx].reset_index(drop=True)
        test_df = master.iloc[test_idx].reset_index(drop=True)
        tr_idx, val_idx = train_test_split(
            np.arange(len(trainval_df)),
            test_size=0.2,
            random_state=args.seed + fold_no,
            stratify=trainval_df["label"].to_numpy(),
        )
        train_df = trainval_df.iloc[tr_idx].reset_index(drop=True)
        val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

        train_ds = mod.MultiBandDataset(train_df.to_dict(orient="records"), append_adj_row=bool(args.append_adj_row), append_node_id=bool(args.append_node_id))
        val_ds = mod.MultiBandDataset(val_df.to_dict(orient="records"), append_adj_row=bool(args.append_adj_row), append_node_id=bool(args.append_node_id))
        test_ds = mod.MultiBandDataset(test_df.to_dict(orient="records"), append_adj_row=bool(args.append_adj_row), append_node_id=bool(args.append_node_id))
        sample_graph = train_ds[0]["pcc"]
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=mod.collate_multiband)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=mod.collate_multiband)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=mod.collate_multiband)

        candidates: list[dict[str, object]] = []
        for init_seed in init_seeds:
            train_seed = (int(init_seed) * 1000) + fold_no
            np.random.seed(train_seed)
            torch.manual_seed(train_seed)
            model = mod.FlatReadoutStaticGNN(
                in_dim=int(sample_graph.x.shape[1]),
                n_nodes=int(sample_graph.x.shape[0]),
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            best_state = None
            best_metric = -1.0
            best_epoch = 1
            stale = 0
            for epoch in range(1, args.epochs + 1):
                _ = mod.train_one_epoch(model, train_loader, optimizer, device)
                y_val, p_val, _ = mod.predict(model, val_loader, device)
                val_metric = metric_value(args.selection_metric, y_true=y_val, prob_pos=p_val)
                if val_metric > best_metric:
                    best_metric = val_metric
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                if epoch >= args.min_epochs and stale >= args.patience:
                    break

            if best_state is None:
                continue
            model.load_state_dict(best_state)
            y_val_best, p_val_best, _ = mod.predict(model, val_loader, device)
            y_test, p_test, w_test = mod.predict(model, test_loader, device)
            candidates.append(
                {
                    "init_seed": int(init_seed),
                    "train_seed": int(train_seed),
                    "best_epoch": int(best_epoch),
                    "best_metric": float(best_metric),
                    "y_val": y_val_best,
                    "p_val": p_val_best,
                    "y_test": y_test,
                    "p_test": p_test,
                    "w_test": w_test,
                }
            )

        if not candidates:
            raise RuntimeError(f"No candidates trained for fold {fold_no}")

        candidates.sort(key=lambda x: (float(x["best_metric"]), -abs(int(x["best_epoch"]) - (args.epochs // 2))), reverse=True)
        selected = candidates[: max(1, min(int(args.select_top_k), len(candidates)))]
        y_val_ref = selected[0]["y_val"]
        y_test_ref = selected[0]["y_test"]
        prob_val_ens = np.mean(np.stack([x["p_val"] for x in selected], axis=0), axis=0)
        prob_test_ens = np.mean(np.stack([x["p_test"] for x in selected], axis=0), axis=0)
        w_test_ens = np.mean(np.stack([x["w_test"] for x in selected], axis=0), axis=0)
        thr_sel, val_thr_score = mod.select_threshold(y_val_ref, prob_val_ens, low=args.thr_low, high=args.thr_high, step=args.thr_step)
        metrics = mod.calc_metrics(y_test_ref, prob_test_ens, threshold=thr_sel)
        w_mean = w_test_ens.mean(axis=0)
        fold_results.append(
            {
                "fold": fold_no,
                "n_train": int(len(train_ds)),
                "n_val": int(len(val_ds)),
                "n_test": int(len(test_ds)),
                "selected_init_seeds": [int(x["init_seed"]) for x in selected],
                "selected_best_epochs": [int(x["best_epoch"]) for x in selected],
                "selected_best_metrics": [float(x["best_metric"]) for x in selected],
                "test_threshold": float(thr_sel),
                "val_threshold_score": float(val_thr_score),
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
            f"fold={fold_no:02d} "
            f"acc={metrics['accuracy']:.4f} "
            f"bal_acc={metrics['balanced_accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'NA'} "
            f"thr={thr_sel:.2f} "
            f"seeds={[int(x['init_seed']) for x in selected]}"
        )

    payload = {
        "schema_version": "eeg_mdd_flatreadout_multiinit_ensemble_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "FlatReadoutStaticGNNMultiInitEnsemble",
        "device": str(device),
        "params": {
            "pcc_manifest": str(args.pcc_manifest),
            "theta_manifest": str(args.theta_manifest),
            "alpha_manifest": str(args.alpha_manifest),
            "beta_manifest": str(args.beta_manifest),
            "folds": int(args.folds),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "seed": int(args.seed),
            "patience": int(args.patience),
            "min_epochs": int(args.min_epochs),
            "thr_low": float(args.thr_low),
            "thr_high": float(args.thr_high),
            "thr_step": float(args.thr_step),
            "append_adj_row": bool(args.append_adj_row),
            "append_node_id": bool(args.append_node_id),
            "init_seeds": init_seeds,
            "select_top_k": int(args.select_top_k),
            "selection_metric": str(args.selection_metric),
        },
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
