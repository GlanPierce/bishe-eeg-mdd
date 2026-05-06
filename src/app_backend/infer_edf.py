from __future__ import annotations

import argparse
import contextlib
import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

SRC_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from build_graphs import GraphConfig, _pcc_matrix, _phase_input_data, _plv_matrix, extract_node_features, read_eeg
from feature_model_utils import (
    build_clean_window_static_graph_feature_table,
    _summarize_window_feature_stack,
    _upper_tri_features,
    _window_starts,
    build_static_graph_feature_table,
    build_targeted_clean_temporal_summary_table,
    build_temporal_summary_table,
)
from run_explicit_region_temporal_summary_node_gnn_5x10 import (
    DEFAULT_CH_NAMES,
    REGION_KEYS,
    TEMPORAL_GROUP_KEYS,
    base_feature_region_map,
    build_expanded_feature_table,
    channel_regions,
    combined_feature_region_map,
    node_feature_region_map,
    _norm_ch_name,
)


BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]
OPTIONAL_PAD_CHANNELS = {"23A-23R", "24A-24R"}
MODEL_PROFILES = {
    "clean": {
        "name": "ExplicitRegionTemporalWeightedStarGNN",
        "display": "Clean 61-subject TASK benchmark",
        "source_result": "0.9143 clean 5 seeds x 10 folds",
        "cache_path": Path("outputs/cache/app_reference_model_v1.pkl"),
        "c": 1.0,
        "targeted": {},
        "note": "Default thesis-facing clean benchmark profile.",
    },
    "targeted_repair": {
        "name": "ExplicitRegionTemporalWeightedStarGNN + targeted artifact repair",
        "display": "Benchmark-tuned artifact repair",
        "source_result": "0.9343 benchmark-tuned 5 seeds x 10 folds",
        "cache_path": Path("outputs/cache/app_reference_model_targeted_repair_v1.pkl"),
        "c": 0.25,
        "targeted": {
            "targeted_clean_bad_abs_threshold": 0.00025,
            "targeted_clean_bad_window_ratio": 0.16,
            "targeted_clean_summary_abs_threshold": 0.00025,
            "targeted_static_clean_bad_window_ratio": 0.33,
            "targeted_temporal_zero_bad_window_ratio": 0.33,
        },
        "note": "Higher-scoring benchmark-tuned profile; report separately from the clean result.",
    },
    "taskonly61_multistate_arch": {
        "name": "Multistate architecture on TASK-only 61-subject features",
        "display": "TASK-only multistate architecture reference",
        "source_result": "TASK-only 61-subject multistate-architecture reference",
        "cache_path": Path("outputs/cache/app_reference_model_taskonly61_multistate_arch_v1.pkl"),
        "c": 1.0,
        "targeted": {},
        "note": "Preview-compatible linear weighted-star refit for comparing the app feature pipeline.",
    },
}
PAIR_BY_REGION = {
    "frontal": [("Fp1", "Fp2"), ("F3", "F4"), ("F7", "F8")],
    "central": [("C3", "C4")],
    "temporal": [("T3", "T4"), ("T5", "T6")],
    "parietal": [("P3", "P4")],
    "occipital": [("O1", "O2")],
}


@dataclass(frozen=True)
class AppModel:
    scaler: StandardScaler
    clf: LogisticRegression
    slices: dict[str, tuple[int, int]]
    feature_region_map: list[list[int]]
    train_subject_count: int
    train_label_counts: dict[str, int]
    profile_key: str = "clean"
    profile_display: str = "Clean 61-subject TASK benchmark"
    source_result: str = "0.9143 clean 5 seeds x 10 folds"
    note: str = ""


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT_DIR / path


def _profile(args: argparse.Namespace) -> dict[str, Any]:
    if args.custom_model_path is not None:
        return {
            "name": "Custom imported AppModel",
            "display": "Custom imported model",
            "source_result": "User-provided compatible AppModel pickle",
            "cache_path": args.custom_model_path,
            "c": float(args.c),
            "targeted": {},
            "note": "Loaded from a user-selected pickle. It must contain an AppModel or {'model': AppModel}.",
        }
    if args.model_profile not in MODEL_PROFILES:
        raise ValueError(f"Unknown model profile: {args.model_profile}. Available: {sorted(MODEL_PROFILES)}")
    return MODEL_PROFILES[args.model_profile]


def _align_channels(data: np.ndarray, ch_names: list[str]) -> tuple[np.ndarray, list[str], list[str], set[str]]:
    """Reorder imported EDF channels to the channel order used by the 0.9143 model."""

    by_norm = {_norm_ch_name(name): idx for idx, name in enumerate(ch_names)}
    target_norm = [_norm_ch_name(name) for name in DEFAULT_CH_NAMES]
    missing = [name for name in target_norm if name not in by_norm]
    warnings: list[str] = []
    if not missing:
        order = [by_norm[name] for name in target_norm]
        return data[order], list(DEFAULT_CH_NAMES), warnings, set()

    if set(missing).issubset(OPTIONAL_PAD_CHANNELS):
        zero_signal = np.zeros(data.shape[1], dtype=data.dtype)
        aligned_rows: list[np.ndarray] = []
        for target_name in target_norm:
            if target_name in by_norm:
                aligned_rows.append(data[by_norm[target_name]])
            else:
                aligned_rows.append(zero_signal)
        warnings.append(
            "Imported EDF is missing optional template channels "
            f"{missing}; represented them as missing channels for model-shape compatibility. "
            "These missing channels are hidden from visual explanations."
        )
        return np.stack(aligned_rows), list(DEFAULT_CH_NAMES), warnings, set(missing)

    if data.shape[0] == len(DEFAULT_CH_NAMES):
        warnings.append(
            "EDF channel names do not fully match the training template; using original order because channel count is 22."
        )
        return data, list(ch_names), warnings, set()

    raise ValueError(
        "Imported EDF has incompatible channels. "
        f"Expected 22 model channels, got {data.shape[0]}; missing template channels: {missing[:8]}"
    )


def _hidden_channel_indices(ch_names: list[str], hidden_channels: set[str] | None) -> list[int]:
    hidden = hidden_channels or set()
    if not hidden:
        return []
    hidden_norm = {_norm_ch_name(name) for name in hidden}
    return [idx for idx, name in enumerate(ch_names) if _norm_ch_name(name) in hidden_norm]


def _zero_hidden_channel_features(
    node_x: np.ndarray,
    matrices: list[np.ndarray],
    hidden_indices: list[int],
) -> None:
    if not hidden_indices:
        return
    node_x[hidden_indices, :] = 0.0
    for mat in matrices:
        mat[hidden_indices, :] = 0.0
        mat[:, hidden_indices] = 0.0


def _static_features_from_data(
    data: np.ndarray,
    sfreq: float,
    hidden_indices: list[int] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    node_x = extract_node_features(data, sfreq).astype(np.float32)
    theta_data = _phase_input_data(data, sfreq, "theta")
    alpha_data = _phase_input_data(data, sfreq, "alpha")
    beta_data = _phase_input_data(data, sfreq, "beta")
    with np.errstate(invalid="ignore", divide="ignore"):
        pcc = _pcc_matrix(data).astype(np.float32)
    theta = _plv_matrix(theta_data).astype(np.float32)
    alpha = _plv_matrix(alpha_data).astype(np.float32)
    beta = _plv_matrix(beta_data).astype(np.float32)
    _zero_hidden_channel_features(node_x, [pcc, theta, alpha, beta], list(hidden_indices or []))
    feat = np.concatenate(
        [
            node_x.reshape(-1),
            _upper_tri_features(pcc),
            _upper_tri_features(theta),
            _upper_tri_features(alpha),
            _upper_tri_features(beta),
        ]
    ).astype(np.float32)
    return feat, {"node_x": node_x, "pcc": pcc, "theta": theta, "alpha": alpha, "beta": beta}


def _temporal_summary_from_data(
    data: np.ndarray,
    sfreq: float,
    window_seconds: float,
    step_seconds: float,
    max_windows: int,
    hidden_indices: list[int] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    win = max(1, int(round(float(window_seconds) * sfreq)))
    step = max(1, int(round(float(step_seconds) * sfreq)))
    starts = _window_starts(data.shape[1], win, step)[: int(max_windows)]

    window_feats: list[np.ndarray] = []
    for start in starts:
        stop = min(start + win, data.shape[1])
        segment = data[:, start:stop]
        if segment.shape[1] >= 8:
            node_x = extract_node_features(segment, sfreq).astype(np.float32)
            if hidden_indices:
                node_x[list(hidden_indices), :] = 0.0
            window_feats.append(node_x.reshape(-1))
    if not window_feats:
        node_x = extract_node_features(data, sfreq).astype(np.float32)
        if hidden_indices:
            node_x[list(hidden_indices), :] = 0.0
        window_feats.append(node_x.reshape(-1))

    stacked = np.stack(window_feats).astype(np.float32)
    summary = _summarize_window_feature_stack(stacked)
    return summary.astype(np.float32), {
        "n_windows": int(stacked.shape[0]),
        "window_seconds": float(window_seconds),
        "step_seconds": float(step_seconds),
        "max_windows": int(max_windows),
    }


def _feature_map() -> list[list[int]]:
    base_map = base_feature_region_map(DEFAULT_CH_NAMES)
    node_map = node_feature_region_map(DEFAULT_CH_NAMES)
    temporal_map = node_map + node_map + node_map
    return combined_feature_region_map(base_map, temporal_map)


def _build_reference_model(args: argparse.Namespace) -> AppModel:
    profile = _profile(args)
    targeted = profile.get("targeted", {})
    with contextlib.redirect_stdout(sys.stderr):
        static_meta, static_x = build_static_graph_feature_table(
            _resolve(args.pcc_manifest),
            _resolve(args.theta_manifest),
            _resolve(args.alpha_manifest),
            _resolve(args.beta_manifest),
        )
        if targeted:
            temporal_meta, temporal_x = build_targeted_clean_temporal_summary_table(
                _resolve(args.split_csv),
                window_seconds=args.window_seconds,
                step_seconds=args.step_seconds,
                max_windows=args.max_windows,
                max_seconds=args.max_seconds,
                bad_window_abs_threshold=float(targeted["targeted_clean_bad_abs_threshold"]),
                replace_bad_window_ratio=float(targeted["targeted_clean_bad_window_ratio"]),
                clean_window_abs_threshold=targeted.get("targeted_clean_summary_abs_threshold"),
            )
        else:
            temporal_meta, temporal_x = build_temporal_summary_table(
                _resolve(args.split_csv),
                window_seconds=args.window_seconds,
                step_seconds=args.step_seconds,
                max_windows=args.max_windows,
                max_seconds=args.max_seconds,
            )

    static_meta = static_meta.assign(_static_idx=np.arange(len(static_meta)))
    temporal_meta = temporal_meta.assign(_temporal_idx=np.arange(len(temporal_meta)))
    merged = (
        static_meta.merge(temporal_meta, on=["subject_id", "label"], how="inner")
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    if len(merged) == 0:
        raise RuntimeError("No overlapping subjects between static graph manifests and temporal split CSV.")

    static_x = static_x[merged["_static_idx"].to_numpy(dtype=np.int64)]
    temporal_x = temporal_x[merged["_temporal_idx"].to_numpy(dtype=np.int64)]
    if targeted.get("targeted_static_clean_bad_window_ratio") is not None:
        with contextlib.redirect_stdout(sys.stderr):
            clean_static_meta, clean_static_x = build_clean_window_static_graph_feature_table(
                _resolve(args.split_csv),
                window_seconds=args.window_seconds,
                step_seconds=args.step_seconds,
                max_windows=args.max_windows,
                max_seconds=args.max_seconds,
                clean_window_abs_threshold=float(
                    targeted.get("targeted_clean_summary_abs_threshold")
                    or targeted["targeted_clean_bad_abs_threshold"]
                ),
            )
        clean_static_meta = clean_static_meta.assign(_clean_static_idx=np.arange(len(clean_static_meta)))
        clean_index = clean_static_meta.set_index("subject_id")["_clean_static_idx"].to_dict()
        clean_static_x = np.stack(
            [clean_static_x[int(clean_index[sid])] for sid in merged["subject_id"].tolist()]
        ).astype(np.float32)
        bad_ratio = merged["bad_window_ratio"].to_numpy(dtype=np.float32)
        static_mask = bad_ratio >= float(targeted["targeted_static_clean_bad_window_ratio"])
        static_x = static_x.copy()
        static_x[static_mask] = clean_static_x[static_mask]

    if targeted.get("targeted_temporal_zero_bad_window_ratio") is not None:
        bad_ratio = merged["bad_window_ratio"].to_numpy(dtype=np.float32)
        temporal_mask = bad_ratio >= float(targeted["targeted_temporal_zero_bad_window_ratio"])
        temporal_x = temporal_x.copy()
        temporal_x[temporal_mask] = 0.0

    y = merged["label"].to_numpy(dtype=np.int64)

    feature_region_map = _feature_map()
    expanded_x, slices = build_expanded_feature_table(static_x, temporal_x, feature_region_map)
    scaler = StandardScaler()
    z = scaler.fit_transform(expanded_x).astype(np.float32)
    clf = LogisticRegression(max_iter=5000, class_weight="balanced", C=float(profile.get("c", args.c)), random_state=int(args.seed))
    clf.fit(z, y)

    counts = pd.Series(y).value_counts().to_dict()
    label_counts = {"normal": int(counts.get(0, 0)), "mdd": int(counts.get(1, 0))}
    return AppModel(
        scaler=scaler,
        clf=clf,
        slices=slices,
        feature_region_map=feature_region_map,
        train_subject_count=int(len(y)),
        train_label_counts=label_counts,
        profile_key=str(args.model_profile),
        profile_display=str(profile["display"]),
        source_result=str(profile["source_result"]),
        note=str(profile["note"]),
    )


def _load_or_build_model(args: argparse.Namespace) -> AppModel:
    profile = _profile(args)
    if args.custom_model_path is not None:
        custom_path = _resolve(args.custom_model_path)
        with custom_path.open("rb") as f:
            payload = pickle.load(f)
        model = payload.get("model") if isinstance(payload, dict) else payload
        required = ["scaler", "clf", "slices", "feature_region_map"]
        missing = [name for name in required if not hasattr(model, name)]
        if missing:
            raise ValueError(f"Custom model is not a compatible AppModel pickle; missing attributes: {missing}")
        return model

    cache_path = _resolve(args.cache_path if args.cache_path is not None else profile["cache_path"])
    cache_key = {
        "schema": "app_reference_model_v1",
        "model_profile": str(args.model_profile),
        "profile_source_result": str(profile["source_result"]),
        "pcc_manifest": str(_resolve(args.pcc_manifest)),
        "theta_manifest": str(_resolve(args.theta_manifest)),
        "alpha_manifest": str(_resolve(args.alpha_manifest)),
        "beta_manifest": str(_resolve(args.beta_manifest)),
        "split_csv": str(_resolve(args.split_csv)),
        "window_seconds": float(args.window_seconds),
        "step_seconds": float(args.step_seconds),
        "max_windows": int(args.max_windows),
        "max_seconds": int(args.max_seconds),
        "c": float(profile.get("c", args.c)),
        "targeted": profile.get("targeted", {}),
        "seed": int(args.seed),
    }
    if cache_path.exists() and not args.rebuild_cache:
        with cache_path.open("rb") as f:
            payload = pickle.load(f)
        if args.trust_cache:
            return payload["model"]
        if payload.get("cache_key") == cache_key:
            return payload["model"]

    model = _build_reference_model(args)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump({"cache_key": cache_key, "model": model}, f)
    return model


def _risk_level(prob_mdd: float) -> dict[str, Any]:
    if prob_mdd < 0.45:
        return {"code": "normal", "label": "正常", "severity": 0}
    if prob_mdd < 0.70:
        return {"code": "mild", "label": "轻度", "severity": 1}
    return {"code": "severe", "label": "重度", "severity": 2}


def _region_name_for_channel(name: str) -> str:
    regs = [REGION_KEYS[idx] for idx in channel_regions(name)]
    for key in ["frontal", "central", "temporal", "parietal", "occipital"]:
        if key in regs:
            return key
    return "other"


def _hemisphere_for_channel(name: str) -> str:
    regs = [REGION_KEYS[idx] for idx in channel_regions(name)]
    if "left" in regs:
        return "left"
    if "right" in regs:
        return "right"
    return "midline"


def _channel_short(name: str) -> str:
    return _norm_ch_name(name)


def _top_edges(
    matrix: np.ndarray,
    ch_names: list[str],
    channel_influence: np.ndarray,
    top_n: int,
    hidden_channels: set[str] | None = None,
) -> list[dict[str, Any]]:
    tri_i, tri_j = np.triu_indices(matrix.shape[0], k=1)
    hidden = hidden_channels or set()
    visible_mask = np.array(
        [
            _norm_ch_name(ch_names[int(i)]) not in hidden and _norm_ch_name(ch_names[int(j)]) not in hidden
            for i, j in zip(tri_i, tri_j)
        ],
        dtype=bool,
    )
    tri_i = tri_i[visible_mask]
    tri_j = tri_j[visible_mask]
    vals = matrix[tri_i, tri_j]
    score = np.abs(vals) * (1.0 + (channel_influence[tri_i] + channel_influence[tri_j]) / 2.0)
    order = np.argsort(score)[::-1][: int(top_n)]
    edges: list[dict[str, Any]] = []
    for rank, idx in enumerate(order, start=1):
        i = int(tri_i[idx])
        j = int(tri_j[idx])
        edges.append(
            {
                "rank": rank,
                "source": _channel_short(ch_names[i]),
                "target": _channel_short(ch_names[j]),
                "value": float(vals[idx]),
                "strength": float(abs(vals[idx])),
                "score": float(score[idx]),
            }
        )
    return edges


def _visual_payload(
    z: np.ndarray,
    coef: np.ndarray,
    slices: dict[str, tuple[int, int]],
    matrices: dict[str, np.ndarray],
    ch_names: list[str],
    node_x: np.ndarray,
    top_edges_n: int,
    hidden_channels: set[str] | None = None,
) -> dict[str, Any]:
    feature_start, feature_stop = slices["feature"]
    region_start, region_stop = slices["region"]
    temporal_start, temporal_stop = slices["temporal_group"]

    feature_contrib = z[feature_start:feature_stop] * coef[feature_start:feature_stop]
    node_band_contrib = feature_contrib[: len(ch_names) * len(BAND_NAMES)].reshape(len(ch_names), len(BAND_NAMES))
    channel_influence = np.sum(np.abs(node_band_contrib), axis=1)
    max_influence = float(np.max(channel_influence)) if np.max(channel_influence) > 0 else 1.0

    region_values = z[region_start:region_stop]
    region_coef = coef[region_start:region_stop]
    region_contrib = region_values * region_coef
    regions = [
        {
            "name": name,
            "value": float(region_values[idx]),
            "coef": float(region_coef[idx]),
            "contribution": float(region_contrib[idx]),
            "absContribution": float(abs(region_contrib[idx])),
        }
        for idx, name in enumerate(REGION_KEYS)
    ]
    regions.sort(key=lambda item: item["absContribution"], reverse=True)

    temporal_values = z[temporal_start:temporal_stop]
    temporal_coef = coef[temporal_start:temporal_stop]
    temporal_contrib = temporal_values * temporal_coef
    temporal_groups = [
        {
            "name": name,
            "value": float(temporal_values[idx]),
            "coef": float(temporal_coef[idx]),
            "contribution": float(temporal_contrib[idx]),
            "absContribution": float(abs(temporal_contrib[idx])),
        }
        for idx, name in enumerate(TEMPORAL_GROUP_KEYS)
    ]

    hidden = hidden_channels or set()
    nodes = []
    for idx, name in enumerate(ch_names):
        if _norm_ch_name(name) in hidden:
            continue
        hemi = _hemisphere_for_channel(name)
        region = _region_name_for_channel(name)
        x = -260 if hemi == "left" else 260 if hemi == "right" else 0
        y = {"frontal": -220, "central": -80, "temporal": 40, "parietal": 120, "occipital": 230}.get(region, 0)
        if region == "other":
            y = -20 + (idx % 4) * 28
        if hemi == "left":
            x -= (idx % 3) * 18
        elif hemi == "right":
            x += (idx % 3) * 18
        elif region == "other":
            x += ((idx % 2) * 2 - 1) * 32
        nodes.append(
            {
                "id": _channel_short(name),
                "name": _channel_short(name),
                "hemisphere": hemi,
                "region": region,
                "x": x,
                "y": y,
                "symbolSize": 18 + 34 * float(channel_influence[idx] / max_influence),
                "influence": float(channel_influence[idx]),
                "bandPower": {band: float(node_x[idx, band_idx]) for band_idx, band in enumerate(BAND_NAMES)},
                "bandContribution": {band: float(node_band_contrib[idx, band_idx]) for band_idx, band in enumerate(BAND_NAMES)},
            }
        )

    pcc_edges = _top_edges(matrices["pcc"], ch_names, channel_influence, top_edges_n, hidden)
    alpha_edges = _top_edges(matrices["alpha"], ch_names, channel_influence, max(6, top_edges_n // 2), hidden)

    asymmetry = []
    norm_to_idx = {_norm_ch_name(name): idx for idx, name in enumerate(ch_names)}
    for region, pairs in PAIR_BY_REGION.items():
        left_score = 0.0
        right_score = 0.0
        used_pairs = []
        for left, right in pairs:
            if left in norm_to_idx and right in norm_to_idx:
                li = norm_to_idx[left]
                ri = norm_to_idx[right]
                left_score += float(channel_influence[li])
                right_score += float(channel_influence[ri])
                used_pairs.append(f"{left}/{right}")
        denom = abs(left_score) + abs(right_score) + 1e-9
        asymmetry.append(
            {
                "region": region,
                "left": left_score,
                "right": right_score,
                "asymmetryIndex": float((left_score - right_score) / denom),
                "pairs": used_pairs,
            }
        )

    return {
        "regions": regions,
        "temporalGroups": temporal_groups,
        "channels": nodes,
        "graph": {"nodes": nodes, "edges": pcc_edges, "alphaEdges": alpha_edges},
        "asymmetry": asymmetry,
    }


def infer(args: argparse.Namespace) -> dict[str, Any]:
    edf_path = _resolve(args.edf)
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")

    model = _load_or_build_model(args)
    cfg = GraphConfig(max_seconds=int(args.max_seconds))
    data, sfreq, ch_names = read_eeg(edf_path, cfg)
    raw_n_channels = int(data.shape[0])
    data, aligned_names, warnings, hidden_channels = _align_channels(data, ch_names)
    hidden_indices = _hidden_channel_indices(aligned_names, hidden_channels)
    static_x, matrices = _static_features_from_data(data, sfreq, hidden_indices=hidden_indices)
    temporal_x, temporal_meta = _temporal_summary_from_data(
        data,
        sfreq,
        window_seconds=float(args.window_seconds),
        step_seconds=float(args.step_seconds),
        max_windows=int(args.max_windows),
        hidden_indices=hidden_indices,
    )

    expanded_x, slices = build_expanded_feature_table(
        static_x.reshape(1, -1),
        temporal_x.reshape(1, -1),
        model.feature_region_map,
    )
    if slices != model.slices:
        raise RuntimeError(f"Feature slice mismatch. expected={model.slices}, got={slices}")

    z = model.scaler.transform(expanded_x).astype(np.float32)
    prob_mdd = float(model.clf.predict_proba(z)[0, 1])
    coef = model.clf.coef_[0].astype(np.float32)
    intercept = float(model.clf.intercept_[0])
    logit = float(z[0] @ coef + intercept)

    visual = _visual_payload(
        z=z[0],
        coef=coef,
        slices=model.slices,
        matrices=matrices,
        ch_names=aligned_names,
        node_x=matrices["node_x"],
        top_edges_n=int(args.top_edges),
        hidden_channels=hidden_channels,
    )
    return {
        "schema": "eeg_mdd_electron_inference_v1",
        "model": {
            "profile": getattr(model, "profile_key", args.model_profile),
            "display": getattr(model, "profile_display", _profile(args)["display"]),
            "name": _profile(args)["name"],
            "sourceResult": getattr(model, "source_result", _profile(args)["source_result"]),
            "productionFit": "same features and weighted-star readout, fitted on all available labeled reference subjects",
            "trainSubjectCount": getattr(model, "train_subject_count", None),
            "trainLabelCounts": getattr(model, "train_label_counts", {}),
            "note": getattr(model, "note", _profile(args)["note"]),
        },
        "file": {
            "path": str(edf_path),
            "name": edf_path.name,
            "sfreq": float(sfreq),
            "nChannels": raw_n_channels,
            "modelChannels": int(data.shape[0]),
            "filledTemplateChannels": sorted(hidden_channels),
            "nSamples": int(data.shape[1]),
            "durationSeconds": float(data.shape[1] / sfreq),
            "warnings": warnings,
        },
        "prediction": {
            "probMdd": prob_mdd,
            "probNormal": float(1.0 - prob_mdd),
            "risk": _risk_level(prob_mdd),
            "logit": logit,
            "thresholds": {"normalMax": 0.45, "mildMax": 0.70},
            "note": "Risk bands are engineering display bands derived from MDD probability, not clinical diagnosis.",
        },
        "features": {
            "staticDim": int(static_x.shape[0]),
            "temporalDim": int(temporal_x.shape[0]),
            "expandedDim": int(expanded_x.shape[1]),
            "temporal": temporal_meta,
        },
        "visualization": visual,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Infer MDD risk from one EDF file and emit JSON for the Electron app.")
    p.add_argument("--edf", type=Path, required=True)
    p.add_argument("--pcc-manifest", type=Path, default=Path("data/processed/graphs_pcc_topk_task/manifest.csv"))
    p.add_argument("--theta-manifest", type=Path, default=Path("data/processed/graphs_plv_theta_topk_task/manifest.csv"))
    p.add_argument("--alpha-manifest", type=Path, default=Path("data/processed/graphs_plv_alpha_topk_task/manifest.csv"))
    p.add_argument("--beta-manifest", type=Path, default=Path("data/processed/graphs_plv_beta_topk_task/manifest.csv"))
    p.add_argument("--split-csv", type=Path, default=Path("data/splits/task_split_subject_level.csv"))
    p.add_argument("--model-profile", type=str, default="clean", choices=sorted(MODEL_PROFILES.keys()))
    p.add_argument("--custom-model-path", type=Path, default=None)
    p.add_argument("--cache-path", type=Path, default=None)
    p.add_argument("--window-seconds", type=float, default=8.0)
    p.add_argument("--step-seconds", type=float, default=8.0)
    p.add_argument("--max-windows", type=int, default=12)
    p.add_argument("--max-seconds", type=int, default=120)
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top-edges", type=int, default=28)
    p.add_argument("--rebuild-cache", action="store_true")
    p.add_argument("--trust-cache", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        payload = infer(args)
        print(json.dumps(payload, ensure_ascii=False, default=_json_default))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc), "type": type(exc).__name__}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
