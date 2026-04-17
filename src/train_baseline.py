from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


@dataclass(frozen=True)
class Sample:
    file_path: Path
    label: int
    label_name: str
    subject_id: str


def parse_sample(path: Path) -> Sample | None:
    name = path.stem
    upper = name.upper()
    if "TASK" not in upper:
        return None

    subj_match = re.search(r"S\s*(\d+)", upper)
    if not subj_match:
        return None
    subj = int(subj_match.group(1))

    if "MDD" in upper:
        label_name = "MDD"
        label = 1
    elif re.search(r"(^|_)H\s*S", upper):
        label_name = "H"
        label = 0
    else:
        return None

    subject_id = f"{label_name}_S{subj:02d}"
    return Sample(path, label, label_name, subject_id)


def collect_task_samples(raw_dir: Path) -> list[Sample]:
    all_files = sorted(raw_dir.glob("*.edf"))
    parsed = [s for s in (parse_sample(p) for p in all_files) if s is not None]

    # If a subject has duplicate TASK files, keep the largest one.
    dedup: dict[str, Sample] = {}
    for sample in parsed:
        prev = dedup.get(sample.subject_id)
        if prev is None or sample.file_path.stat().st_size > prev.file_path.stat().st_size:
            dedup[sample.subject_id] = sample
    return sorted(dedup.values(), key=lambda s: s.subject_id)


def _band_feature(freqs: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return np.full(psd.shape[0], 1e-12, dtype=float)
    return psd[:, mask].mean(axis=1) + 1e-12


def extract_features(file_path: Path, max_seconds: int = 120) -> dict[str, float]:
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose="ERROR")
    raw.pick("eeg")
    raw.filter(l_freq=0.5, h_freq=45.0, verbose="ERROR")

    sfreq = float(raw.info["sfreq"])
    n_stop = min(raw.n_times, int(max_seconds * sfreq))
    data = raw.get_data(start=0, stop=n_stop)
    nperseg = min(1024, data.shape[1])
    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=1)

    total = _band_feature(freqs, psd, 0.5, 45.0)
    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 45.0),
    }

    feats: dict[str, float] = {}
    for band_name, (fmin, fmax) in bands.items():
        bp = _band_feature(freqs, psd, fmin, fmax)
        rel = bp / total
        feats[f"{band_name}_mean"] = float(np.mean(rel))
        feats[f"{band_name}_std"] = float(np.std(rel))
    return feats


def build_feature_table(samples: Iterable[Sample]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for idx, s in enumerate(samples, start=1):
        print(f"[{idx}] feature extraction: {s.file_path.name}")
        feats = extract_features(s.file_path)
        row: dict[str, object] = {
            "file_path": str(s.file_path),
            "subject_id": s.subject_id,
            "label": s.label,
            "label_name": s.label_name,
        }
        row.update(feats)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw" / "mdd-patients-eeg-dataset"
    out_dir = root / "outputs" / "metrics"
    split_dir = root / "data" / "splits"
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)

    samples = collect_task_samples(raw_dir)
    if not samples:
        raise RuntimeError(f"No TASK EDF samples found in {raw_dir}")

    df = build_feature_table(samples)
    features_csv = split_dir / "task_feature_table.csv"
    df.to_csv(features_csv, index=False, encoding="utf-8-sig")

    subj_df = df[["subject_id", "label"]].drop_duplicates().sort_values("subject_id")
    train_subj, test_subj = train_test_split(
        subj_df["subject_id"],
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=subj_df["label"],
    )
    train_subj_set = set(train_subj.tolist())
    df["split"] = np.where(df["subject_id"].isin(train_subj_set), "train", "test")
    df.to_csv(split_dir / "task_split_subject_level.csv", index=False, encoding="utf-8-sig")

    feature_cols = [
        c
        for c in df.columns
        if c.endswith("_mean") or c.endswith("_std")
    ]
    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    models: dict[str, object] = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
        ),
    }

    result: dict[str, object] = {
        "dataset_summary": {
            "total_samples": int(len(df)),
            "train_samples": int(len(train_df)),
            "test_samples": int(len(test_df)),
            "train_class_counts": train_df["label_name"].value_counts().to_dict(),
            "test_class_counts": test_df["label_name"].value_counts().to_dict(),
        },
        "models": {},
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
            auc = float(roc_auc_score(y_test, prob))
        else:
            auc = None

        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "roc_auc": auc,
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "classification_report": classification_report(
                y_test, pred, target_names=["H", "MDD"], output_dict=True
            ),
        }
        result["models"][model_name] = metrics

    metrics_path = out_dir / "baseline_task_binary_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\nSaved:")
    print(f"- {features_csv}")
    print(f"- {split_dir / 'task_split_subject_level.csv'}")
    print(f"- {metrics_path}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
