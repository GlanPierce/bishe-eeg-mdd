from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from build_graphs import GraphConfig, _pcc_matrix, _phase_input_data, _plv_matrix, extract_node_features, read_eeg


def parse_seed_list(seed_text: str) -> list[int]:
    seeds = [int(x.strip()) for x in str(seed_text).split(",") if x.strip()]
    if not seeds:
        raise ValueError("Seed list must not be empty.")
    return seeds


def calc_metrics(y_true: np.ndarray, prob_pos: np.ndarray, threshold: float = 0.5) -> dict[str, object]:
    pred = (prob_pos >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, prob_pos)) if len(np.unique(y_true)) == 2 else None,
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def aggregate_fold_results(fold_results: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    metric_names = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
    agg: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values = np.array([fr["test_metrics"][metric_name] for fr in fold_results], dtype=np.float64)
        agg[metric_name] = {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values))}
    return agg


def summarize_run_payloads(run_payloads: list[dict[str, object]]) -> dict[str, object]:
    runs: list[dict[str, object]] = []
    for payload in run_payloads:
        agg = payload["aggregate"]
        run_path = payload.get("_out_path", "")
        runs.append(
            {
                "seed": int(payload["params"]["seed"]),
                "path": str(run_path),
                "accuracy_mean": float(agg["accuracy"]["mean"]),
                "accuracy_std": float(agg["accuracy"]["std"]),
                "balanced_accuracy_mean": float(agg["balanced_accuracy"]["mean"]),
                "balanced_accuracy_std": float(agg["balanced_accuracy"]["std"]),
                "f1_mean": float(agg["f1"]["mean"]),
                "f1_std": float(agg["f1"]["std"]),
                "roc_auc_mean": float(agg["roc_auc"]["mean"]),
                "roc_auc_std": float(agg["roc_auc"]["std"]),
            }
        )

    metric_pairs = [
        ("accuracy_mean", "accuracy_mean_of_means", "accuracy_std_of_means"),
        ("balanced_accuracy_mean", "balanced_accuracy_mean_of_means", "balanced_accuracy_std_of_means"),
        ("f1_mean", "f1_mean_of_means", "f1_std_of_means"),
        ("roc_auc_mean", "roc_auc_mean_of_means", "roc_auc_std_of_means"),
    ]
    summary_over_runs: dict[str, float] = {}
    for run_key, mean_key, std_key in metric_pairs:
        values = np.array([run[run_key] for run in runs], dtype=np.float64)
        summary_over_runs[mean_key] = float(np.nanmean(values))
        summary_over_runs[std_key] = float(np.nanstd(values))

    return {
        "n_runs": int(len(runs)),
        "runs": runs,
        "summary_over_runs": summary_over_runs,
    }


def _load_graph_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    x = np.asarray(d["x"], dtype=np.float32)
    edge_index = np.asarray(d["edge_index"], dtype=np.int64)
    edge_weight = np.asarray(d["edge_weight"], dtype=np.float32)
    n_nodes = int(x.shape[0])
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for (src, dst), weight in zip(edge_index.T, edge_weight):
        adj[int(src), int(dst)] = float(weight)
    return x, adj


def build_static_graph_feature_table(
    pcc_manifest: Path,
    theta_manifest: Path,
    alpha_manifest: Path,
    beta_manifest: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    pcc = pd.read_csv(pcc_manifest)[["subject_id", "label", "graph_path"]].rename(columns={"graph_path": "pcc_path"})
    theta = pd.read_csv(theta_manifest)[["subject_id", "graph_path"]].rename(columns={"graph_path": "theta_path"})
    alpha = pd.read_csv(alpha_manifest)[["subject_id", "graph_path"]].rename(columns={"graph_path": "alpha_path"})
    beta = pd.read_csv(beta_manifest)[["subject_id", "graph_path"]].rename(columns={"graph_path": "beta_path"})
    master = pcc.merge(theta, on="subject_id").merge(alpha, on="subject_id").merge(beta, on="subject_id")
    master = master.sort_values("subject_id").reset_index(drop=True)

    features: list[np.ndarray] = []
    for _, row in master.iterrows():
        node_x, adj_pcc = _load_graph_npz(Path(str(row["pcc_path"])))
        _, adj_theta = _load_graph_npz(Path(str(row["theta_path"])))
        _, adj_alpha = _load_graph_npz(Path(str(row["alpha_path"])))
        _, adj_beta = _load_graph_npz(Path(str(row["beta_path"])))
        tri = np.triu_indices(node_x.shape[0], k=1)
        feat = np.concatenate(
            [
                node_x.reshape(-1),
                adj_pcc[tri],
                adj_theta[tri],
                adj_alpha[tri],
                adj_beta[tri],
            ]
        ).astype(np.float32)
        features.append(feat)
    return master[["subject_id", "label"]].copy(), np.stack(features)


def _window_starts(n_samples: int, win: int, step: int) -> list[int]:
    if n_samples <= win:
        return [0]
    starts = list(range(0, n_samples - win + 1, step))
    if starts[-1] != n_samples - win:
        starts.append(n_samples - win)
    return starts


def _summarize_window_feature_stack(stacked: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            stacked.mean(axis=0),
            stacked.std(axis=0),
            stacked[-1] - stacked[0],
        ]
    ).astype(np.float32)


def _upper_tri_features(mat: np.ndarray) -> np.ndarray:
    tri = np.triu_indices(mat.shape[0], k=1)
    return mat[tri].astype(np.float32)


def _build_static_window_feature(segment: np.ndarray, sfreq: float) -> np.ndarray:
    node_x = extract_node_features(segment, sfreq).astype(np.float32)
    theta_data = _phase_input_data(segment, sfreq, "theta")
    alpha_data = _phase_input_data(segment, sfreq, "alpha")
    beta_data = _phase_input_data(segment, sfreq, "beta")
    return np.concatenate(
        [
            node_x.reshape(-1),
            _upper_tri_features(_pcc_matrix(segment)),
            _upper_tri_features(_plv_matrix(theta_data)),
            _upper_tri_features(_plv_matrix(alpha_data)),
            _upper_tri_features(_plv_matrix(beta_data)),
        ]
    ).astype(np.float32)


def build_temporal_summary_table(
    split_csv: Path,
    window_seconds: float,
    step_seconds: float,
    max_windows: int,
    max_seconds: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(split_csv)
    required_cols = {"file_path", "subject_id", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"split csv missing columns: {sorted(missing)}")

    df = df.sort_values("subject_id").drop_duplicates(subset=["subject_id"], keep="first").reset_index(drop=True)
    cfg = GraphConfig(max_seconds=max_seconds)

    features: list[np.ndarray] = []
    n_windows_all: list[int] = []
    for idx, row in df.iterrows():
        file_path = Path(str(row["file_path"]))
        subject_id = str(row["subject_id"])
        data, sfreq, _ = read_eeg(file_path, cfg)
        win = max(1, int(round(float(window_seconds) * sfreq)))
        step = max(1, int(round(float(step_seconds) * sfreq)))
        starts = _window_starts(data.shape[1], win, step)[: int(max_windows)]

        window_feats: list[np.ndarray] = []
        for start in starts:
            stop = min(start + win, data.shape[1])
            segment = data[:, start:stop]
            if segment.shape[1] < 8:
                continue
            window_feats.append(extract_node_features(segment, sfreq).reshape(-1))

        if not window_feats:
            window_feats.append(extract_node_features(data, sfreq).reshape(-1))

        stacked = np.stack(window_feats).astype(np.float32)
        summary = _summarize_window_feature_stack(stacked)
        print(
            f"[{idx + 1}/{len(df)}] temporal summary: {subject_id} "
            f"windows={stacked.shape[0]} summary_dim={summary.shape[0]}"
        )
        features.append(summary)
        n_windows_all.append(int(stacked.shape[0]))

    out_df = df[["subject_id", "label"]].copy()
    out_df["n_windows"] = n_windows_all
    return out_df, np.stack(features)


def build_targeted_clean_temporal_summary_table(
    split_csv: Path,
    window_seconds: float,
    step_seconds: float,
    max_windows: int,
    max_seconds: int,
    bad_window_abs_threshold: float,
    replace_bad_window_ratio: float,
    clean_window_abs_threshold: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(split_csv)
    required_cols = {"file_path", "subject_id", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"split csv missing columns: {sorted(missing)}")

    df = df.sort_values("subject_id").drop_duplicates(subset=["subject_id"], keep="first").reset_index(drop=True)
    cfg = GraphConfig(max_seconds=max_seconds)
    clean_thr = float(bad_window_abs_threshold if clean_window_abs_threshold is None else clean_window_abs_threshold)

    features: list[np.ndarray] = []
    n_windows_all: list[int] = []
    n_clean_windows_all: list[int] = []
    bad_ratio_all: list[float] = []
    replaced_all: list[int] = []
    for idx, row in df.iterrows():
        file_path = Path(str(row["file_path"]))
        subject_id = str(row["subject_id"])
        data, sfreq, _ = read_eeg(file_path, cfg)
        win = max(1, int(round(float(window_seconds) * sfreq)))
        step = max(1, int(round(float(step_seconds) * sfreq)))
        starts = _window_starts(data.shape[1], win, step)[: int(max_windows)]

        window_feats: list[np.ndarray] = []
        window_abs_max: list[float] = []
        for start in starts:
            stop = min(start + win, data.shape[1])
            segment = data[:, start:stop]
            if segment.shape[1] < 8:
                continue
            window_feats.append(extract_node_features(segment, sfreq).reshape(-1))
            window_abs_max.append(float(np.max(np.abs(segment))))

        if not window_feats:
            window_feats.append(extract_node_features(data, sfreq).reshape(-1))
            window_abs_max.append(float(np.max(np.abs(data))))

        stacked = np.stack(window_feats).astype(np.float32)
        abs_max = np.asarray(window_abs_max, dtype=np.float32)
        clean_mask = abs_max <= float(clean_thr)
        bad_ratio = float(np.mean(abs_max > float(bad_window_abs_threshold)))
        should_replace = bool(bad_ratio >= float(replace_bad_window_ratio))
        chosen = stacked[clean_mask] if should_replace and np.any(clean_mask) else stacked
        summary = _summarize_window_feature_stack(chosen)
        print(
            f"[{idx + 1}/{len(df)}] targeted temporal summary: {subject_id} "
            f"windows={stacked.shape[0]} clean_windows={int(np.sum(clean_mask))} "
            f"bad_ratio={bad_ratio:.3f} replaced={int(should_replace)}"
        )
        features.append(summary)
        n_windows_all.append(int(stacked.shape[0]))
        n_clean_windows_all.append(int(np.sum(clean_mask)))
        bad_ratio_all.append(bad_ratio)
        replaced_all.append(int(should_replace))

    out_df = df[["subject_id", "label"]].copy()
    out_df["n_windows"] = n_windows_all
    out_df["n_clean_windows"] = n_clean_windows_all
    out_df["bad_window_ratio"] = bad_ratio_all
    out_df["temporal_replaced"] = replaced_all
    return out_df, np.stack(features)


def build_clean_window_static_graph_feature_table(
    split_csv: Path,
    window_seconds: float,
    step_seconds: float,
    max_windows: int,
    max_seconds: int,
    clean_window_abs_threshold: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(split_csv)
    required_cols = {"file_path", "subject_id", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"split csv missing columns: {sorted(missing)}")

    df = df.sort_values("subject_id").drop_duplicates(subset=["subject_id"], keep="first").reset_index(drop=True)
    cfg = GraphConfig(max_seconds=max_seconds)

    features: list[np.ndarray] = []
    n_windows_all: list[int] = []
    n_clean_windows_all: list[int] = []
    bad_ratio_all: list[float] = []
    max_abs_all: list[float] = []
    for idx, row in df.iterrows():
        file_path = Path(str(row["file_path"]))
        subject_id = str(row["subject_id"])
        data, sfreq, _ = read_eeg(file_path, cfg)
        win = max(1, int(round(float(window_seconds) * sfreq)))
        step = max(1, int(round(float(step_seconds) * sfreq)))
        starts = _window_starts(data.shape[1], win, step)[: int(max_windows)]

        window_feats: list[np.ndarray] = []
        window_abs_max: list[float] = []
        for start in starts:
            stop = min(start + win, data.shape[1])
            segment = data[:, start:stop]
            if segment.shape[1] < 8:
                continue
            window_feats.append(_build_static_window_feature(segment, sfreq))
            window_abs_max.append(float(np.max(np.abs(segment))))

        if not window_feats:
            window_feats.append(_build_static_window_feature(data, sfreq))
            window_abs_max.append(float(np.max(np.abs(data))))

        stacked = np.stack(window_feats).astype(np.float32)
        abs_max = np.asarray(window_abs_max, dtype=np.float32)
        clean_mask = abs_max <= float(clean_window_abs_threshold)
        chosen = stacked[clean_mask] if np.any(clean_mask) else stacked
        summary = chosen.mean(axis=0).astype(np.float32)
        bad_ratio = float(np.mean(abs_max > float(clean_window_abs_threshold)))
        print(
            f"[{idx + 1}/{len(df)}] clean static graph: {subject_id} "
            f"windows={stacked.shape[0]} clean_windows={int(np.sum(clean_mask))} "
            f"bad_ratio={bad_ratio:.3f}"
        )
        features.append(summary)
        n_windows_all.append(int(stacked.shape[0]))
        n_clean_windows_all.append(int(np.sum(clean_mask)))
        bad_ratio_all.append(bad_ratio)
        max_abs_all.append(float(np.max(np.abs(data))))

    out_df = df[["subject_id", "label"]].copy()
    out_df["n_windows"] = n_windows_all
    out_df["n_clean_windows"] = n_clean_windows_all
    out_df["bad_window_ratio"] = bad_ratio_all
    out_df["max_abs"] = max_abs_all
    return out_df, np.stack(features)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
