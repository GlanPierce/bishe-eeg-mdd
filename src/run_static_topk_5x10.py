from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repeated 5x10 CV for static graph-vector logistic baseline.")
    p.add_argument("--pcc-manifest", type=Path, default=Path("data/processed/graphs_pcc_topk_task/manifest.csv"))
    p.add_argument("--theta-manifest", type=Path, default=Path("data/processed/graphs_plv_theta_topk_task/manifest.csv"))
    p.add_argument("--alpha-manifest", type=Path, default=Path("data/processed/graphs_plv_alpha_topk_task/manifest.csv"))
    p.add_argument("--beta-manifest", type=Path, default=Path("data/processed/graphs_plv_beta_topk_task/manifest.csv"))
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/metrics/runs/static_graphvector_lr_5x10"),
    )
    p.add_argument("--experiment-name", type=str, default="static_graphvector_lr_5x10")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    meta, X = build_static_graph_feature_table(
        pcc_manifest=args.pcc_manifest,
        theta_manifest=args.theta_manifest,
        alpha_manifest=args.alpha_manifest,
        beta_manifest=args.beta_manifest,
    )
    y = meta["label"].to_numpy(dtype=np.int64)
    subjects = meta["subject_id"].to_numpy()

    run_payloads: list[dict[str, object]] = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        fold_results: list[dict[str, object]] = []
        for fold_no, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=5000,
                            class_weight="balanced",
                            C=float(args.c),
                            random_state=int(seed),
                        ),
                    ),
                ]
            )
            model.fit(X[train_idx], y[train_idx])
            prob = model.predict_proba(X[test_idx])[:, 1]
            metrics = calc_metrics(y[test_idx], prob, threshold=0.5)
            fold_results.append(
                {
                    "fold": fold_no,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "test_subject_ids": subjects[test_idx].tolist(),
                    "test_metrics": metrics,
                }
            )
            print(
                f"seed={seed} fold={fold_no:02d} "
                f"acc={metrics['accuracy']:.4f} "
                f"bal_acc={metrics['balanced_accuracy']:.4f} "
                f"f1={metrics['f1']:.4f} "
                f"auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'NA'}"
            )

        payload = {
            "schema_version": "eeg_mdd_graphvector_static_5x10_v1",
            "experiment_name": args.experiment_name,
            "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model": "GraphVectorLogisticRegression",
            "params": {
                "folds": args.folds,
                "seed": seed,
                "seeds": seeds,
                "c": float(args.c),
            },
            "manifests": {
                "pcc": str(args.pcc_manifest),
                "theta": str(args.theta_manifest),
                "alpha": str(args.alpha_manifest),
                "beta": str(args.beta_manifest),
            },
            "feature_shape": {
                "n_subjects": int(X.shape[0]),
                "n_features": int(X.shape[1]),
            },
            "aggregate": aggregate_fold_results(fold_results),
            "fold_results": fold_results,
        }
        out_path = args.out_dir / f"static_graphvector_lr_seed{seed}_cv10.json"
        write_json(out_path, payload)
        run_payloads.append({**payload, "_out_path": str(out_path)})
        print(f"saved: {out_path}")

    summary = summarize_run_payloads(run_payloads)
    summary_payload = {
        "schema_version": "eeg_mdd_graphvector_static_5x10_summary_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "GraphVectorLogisticRegression",
        "params": {
            "folds": args.folds,
            "seeds": seeds,
            "c": float(args.c),
        },
        "manifests": {
            "pcc": str(args.pcc_manifest),
            "theta": str(args.theta_manifest),
            "alpha": str(args.alpha_manifest),
            "beta": str(args.beta_manifest),
        },
        "feature_shape": {
            "n_subjects": int(X.shape[0]),
            "n_features": int(X.shape[1]),
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
