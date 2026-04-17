from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def calc_metrics(y_true: np.ndarray, prob_pos: np.ndarray) -> dict[str, object]:
    pred = (prob_pos >= 0.5).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, prob_pos)) if len(np.unique(y_true)) == 2 else None,
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def aggregate_metrics(fold_results: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    metric_names = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
    agg: dict[str, dict[str, float]] = {}
    for mn in metric_names:
        vals = np.array([fr["test_metrics"][mn] for fr in fold_results], dtype=np.float64)
        agg[mn] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    return agg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="10-fold CV for non-GNN baselines")
    p.add_argument("--feature-csv", type=Path, default=Path("data/splits/task_feature_table.csv"))
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.feature_csv)
    feature_cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_std")]

    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)
    subjects = df["subject_id"].to_numpy()

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    models: dict[str, object] = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=args.seed)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            random_state=args.seed,
            class_weight="balanced_subsample",
        ),
    }

    out: dict[str, object] = {
        "params": {"folds": args.folds, "seed": args.seed},
        "n_samples": int(len(df)),
        "class_counts": df["label_name"].value_counts().to_dict(),
        "models": {},
    }

    for model_name, model in models.items():
        fold_results: list[dict[str, object]] = []
        for fold_no, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            prob = model.predict_proba(X_test)[:, 1]
            metrics = calc_metrics(y_test, prob)
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
                f"{model_name} fold={fold_no:02d} "
                f"acc={metrics['accuracy']:.4f} "
                f"bal_acc={metrics['balanced_accuracy']:.4f} "
                f"f1={metrics['f1']:.4f} "
                f"auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'NA'}"
            )

        out["models"][model_name] = {
            "aggregate": aggregate_metrics(fold_results),
            "fold_results": fold_results,
        }

    out_dir = Path("outputs/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "nongnn_cv10_metrics.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
