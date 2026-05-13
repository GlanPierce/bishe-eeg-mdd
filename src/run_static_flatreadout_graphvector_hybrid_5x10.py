from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from feature_model_utils import (
    aggregate_fold_results,
    build_static_graph_feature_table,
    calc_metrics,
    parse_seed_list,
    summarize_run_payloads,
    write_json,
)


def load_flatreadout_module():
    src_dir = str(Path("src").resolve())
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import train_gnn_dualgraph_multiband_flatreadout_cv10 as mod

    return mod


def select_hybrid_threshold(
    y_val: np.ndarray,
    prob_hybrid: np.ndarray,
    thr_grid: np.ndarray,
) -> tuple[float, float]:
    best_thr = 0.5
    best_score = -1.0
    for thr in thr_grid:
        pred = (prob_hybrid >= thr).astype(np.int64)
        score = float((pred == y_val).mean())
        if (score > best_score) or (abs(score - best_score) <= 1e-12 and abs(thr - 0.5) < abs(best_thr - 0.5)):
            best_score = score
            best_thr = float(thr)
    return best_thr, best_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repeated 5x10 CV for flatreadout-GNN + graphvector hybrid.")
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
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--aux-c", type=float, default=1.0)
    p.add_argument("--hybrid-alpha", type=float, default=0.5, help="Weight on the GNN branch.")
    p.add_argument("--thr-low", type=float, default=0.3)
    p.add_argument("--thr-high", type=float, default=0.7)
    p.add_argument("--thr-step", type=float, default=0.01)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--refit-full-train", action="store_true")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/metrics/runs/static_flatreadout_graphvector_hybrid_5x10"))
    p.add_argument("--experiment-name", type=str, default="static_flatreadout_graphvector_hybrid_5x10")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    flat_mod = load_flatreadout_module()

    master = flat_mod.build_master_table(
        pcc_manifest=args.pcc_manifest,
        theta_manifest=args.theta_manifest,
        alpha_manifest=args.alpha_manifest,
        beta_manifest=args.beta_manifest,
    )
    meta, X_aux = build_static_graph_feature_table(
        pcc_manifest=args.pcc_manifest,
        theta_manifest=args.theta_manifest,
        alpha_manifest=args.alpha_manifest,
        beta_manifest=args.beta_manifest,
    )
    if master["subject_id"].tolist() != meta["subject_id"].tolist():
        raise ValueError("Subject order mismatch between graph table and graphvector features.")

    y = master["label"].to_numpy(dtype=np.int64)
    subjects = master["subject_id"].to_numpy()
    thr_grid = np.arange(float(args.thr_low), float(args.thr_high) + (float(args.thr_step) * 0.5), float(args.thr_step), dtype=np.float64)
    alpha = float(args.hybrid_alpha)

    run_payloads: list[dict[str, object]] = []
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        fold_results: list[dict[str, object]] = []

        for fold_no, (trainval_idx, test_idx) in enumerate(skf.split(np.arange(len(master)), y), start=1):
            trainval_df = master.iloc[trainval_idx].reset_index(drop=True)
            test_df = master.iloc[test_idx].reset_index(drop=True)
            X_trainval_aux = X_aux[trainval_idx]
            X_test_aux = X_aux[test_idx]
            y_trainval = y[trainval_idx]
            y_test = y[test_idx]

            tr_idx, val_idx = train_test_split(
                np.arange(len(trainval_df)),
                test_size=float(args.val_size),
                random_state=seed + fold_no,
                stratify=y_trainval,
            )
            train_df = trainval_df.iloc[tr_idx].reset_index(drop=True)
            val_df = trainval_df.iloc[val_idx].reset_index(drop=True)
            X_train_aux = X_trainval_aux[tr_idx]
            X_val_aux = X_trainval_aux[val_idx]
            y_train = y_trainval[tr_idx]
            y_val = y_trainval[val_idx]

            aux_model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=5000,
                            class_weight="balanced",
                            C=float(args.aux_c),
                            random_state=int(seed),
                        ),
                    ),
                ]
            )
            aux_model.fit(X_train_aux, y_train)
            prob_val_aux = aux_model.predict_proba(X_val_aux)[:, 1]
            prob_test_aux = aux_model.predict_proba(X_test_aux)[:, 1]

            train_ds = flat_mod.MultiBandDataset(train_df.to_dict(orient="records"), append_adj_row=True)
            val_ds = flat_mod.MultiBandDataset(val_df.to_dict(orient="records"), append_adj_row=True)
            test_ds = flat_mod.MultiBandDataset(test_df.to_dict(orient="records"), append_adj_row=True)
            sample_graph = train_ds[0]["pcc"]

            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=flat_mod.collate_multiband)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=flat_mod.collate_multiband)
            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=flat_mod.collate_multiband)

            model = flat_mod.FlatReadoutStaticGNN(
                in_dim=int(sample_graph.x.shape[1]),
                n_nodes=int(sample_graph.x.shape[0]),
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            best_state = None
            best_val_acc = -1.0
            best_epoch = 1
            for epoch_no in range(1, args.epochs + 1):
                _ = flat_mod.train_one_epoch(model, train_loader, optimizer, device)
                y_val_gnn, p_val_gnn, _ = flat_mod.predict(model, val_loader, device)
                val_acc = float(((p_val_gnn >= 0.5).astype(np.int64) == y_val_gnn).mean())
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = int(epoch_no)
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if best_state is not None:
                model.load_state_dict(best_state)

            y_val_gnn, p_val_gnn, _ = flat_mod.predict(model, val_loader, device)
            y_test_gnn, p_test_gnn, w_test = flat_mod.predict(model, test_loader, device)
            prob_val_hybrid = (alpha * p_val_gnn) + ((1.0 - alpha) * prob_val_aux)
            thr_sel, val_thr_acc = select_hybrid_threshold(y_val=y_val, prob_hybrid=prob_val_hybrid, thr_grid=thr_grid)

            final_prob_test_gnn = p_test_gnn
            final_prob_test_aux = prob_test_aux
            final_w_test = w_test
            if args.refit_full_train:
                aux_model_full = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            LogisticRegression(
                                max_iter=5000,
                                class_weight="balanced",
                                C=float(args.aux_c),
                                random_state=int(seed),
                            ),
                        ),
                    ]
                )
                aux_model_full.fit(X_trainval_aux, y_trainval)
                final_prob_test_aux = aux_model_full.predict_proba(X_test_aux)[:, 1]

                full_ds = flat_mod.MultiBandDataset(trainval_df.to_dict(orient="records"), append_adj_row=True)
                full_loader = torch.utils.data.DataLoader(full_ds, batch_size=args.batch_size, shuffle=True, collate_fn=flat_mod.collate_multiband)
                full_model = flat_mod.FlatReadoutStaticGNN(
                    in_dim=int(sample_graph.x.shape[1]),
                    n_nodes=int(sample_graph.x.shape[0]),
                    hidden_dim=args.hidden_dim,
                    dropout=args.dropout,
                ).to(device)
                full_opt = torch.optim.Adam(full_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                for _ in range(max(1, int(best_epoch))):
                    _ = flat_mod.train_one_epoch(full_model, full_loader, full_opt, device)
                _, final_prob_test_gnn, final_w_test = flat_mod.predict(full_model, test_loader, device)

            prob_test_hybrid = (alpha * final_prob_test_gnn) + ((1.0 - alpha) * final_prob_test_aux)
            metrics = calc_metrics(y_test_gnn, prob_test_hybrid, threshold=thr_sel)
            w_mean = final_w_test.mean(axis=0)

            fold_results.append(
                {
                    "fold": fold_no,
                    "n_train": int(len(train_ds)),
                    "n_val": int(len(val_ds)),
                    "n_test": int(len(test_ds)),
                    "best_val_accuracy_gnn": float(best_val_acc),
                    "best_epoch_gnn": int(best_epoch),
                    "hybrid_alpha": float(alpha),
                    "hybrid_threshold": float(thr_sel),
                    "hybrid_val_accuracy": float(val_thr_acc),
                    "test_subject_ids": subjects[test_idx].tolist(),
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
                f"alpha={alpha:.2f} thr={thr_sel:.2f}"
            )

        payload = {
            "schema_version": "eeg_mdd_flatreadout_gnn_graphvector_hybrid_5x10_v1",
            "experiment_name": args.experiment_name,
            "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model": "StaticFlatReadoutGNNGraphVectorHybrid",
            "params": {
                "folds": args.folds,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "hidden_dim": int(args.hidden_dim),
                "dropout": float(args.dropout),
                "seed": int(seed),
                "seeds": seeds,
                "aux_c": float(args.aux_c),
                "hybrid_alpha": float(alpha),
                "thr_low": float(args.thr_low),
                "thr_high": float(args.thr_high),
                "thr_step": float(args.thr_step),
                "val_size": float(args.val_size),
                "refit_full_train": bool(args.refit_full_train),
            },
            "manifests": {
                "pcc": str(args.pcc_manifest),
                "theta": str(args.theta_manifest),
                "alpha": str(args.alpha_manifest),
                "beta": str(args.beta_manifest),
            },
            "feature_shape": {
                "n_subjects": int(X_aux.shape[0]),
                "n_features": int(X_aux.shape[1]),
            },
            "aggregate": aggregate_fold_results(fold_results),
            "fold_results": fold_results,
        }
        out_path = args.out_dir / f"static_flatreadout_graphvector_hybrid_seed{seed}_cv10.json"
        write_json(out_path, payload)
        run_payloads.append({**payload, "_out_path": str(out_path)})
        print(f"saved: {out_path}")

    summary = summarize_run_payloads(run_payloads)
    summary_payload = {
        "schema_version": "eeg_mdd_flatreadout_gnn_graphvector_hybrid_5x10_summary_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "StaticFlatReadoutGNNGraphVectorHybrid",
        "params": {
            "folds": args.folds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "seeds": seeds,
            "aux_c": float(args.aux_c),
            "hybrid_alpha": float(alpha),
            "thr_low": float(args.thr_low),
            "thr_high": float(args.thr_high),
            "thr_step": float(args.thr_step),
            "val_size": float(args.val_size),
            "refit_full_train": bool(args.refit_full_train),
        },
        "manifests": {
            "pcc": str(args.pcc_manifest),
            "theta": str(args.theta_manifest),
            "alpha": str(args.alpha_manifest),
            "beta": str(args.beta_manifest),
        },
        "feature_shape": {
            "n_subjects": int(X_aux.shape[0]),
            "n_features": int(X_aux.shape[1]),
        },
        **summary,
    }
    summary_path = args.out_dir / "summary_5x10.json"
    write_json(summary_path, summary_payload)
    print(f"\nsummary: {summary_path}")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
