from __future__ import annotations

import argparse
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from build_graphs import GraphConfig, build_edges, extract_node_features, read_eeg
from feature_model_utils import (
    aggregate_fold_results,
    calc_metrics,
    parse_seed_list,
    summarize_run_payloads,
    write_json,
)

STATE_KEYS = ["TASK", "EC", "EO"]
PRIMARY_REGION_KEYS = ["frontal", "central", "temporal", "parietal", "occipital"]
REGION_KEYS = [
    "frontal",
    "central",
    "temporal",
    "parietal",
    "occipital",
    "left",
    "right",
    "midline",
    "frontal_left",
    "frontal_right",
    "cross_region",
    "global",
]
DEFAULT_CH_NAMES = [
    "EEG Fp1-LE",
    "EEG F3-LE",
    "EEG C3-LE",
    "EEG P3-LE",
    "EEG O1-LE",
    "EEG F7-LE",
    "EEG T3-LE",
    "EEG T5-LE",
    "EEG Fz-LE",
    "EEG Fp2-LE",
    "EEG F4-LE",
    "EEG C4-LE",
    "EEG P4-LE",
    "EEG O2-LE",
    "EEG F8-LE",
    "EEG T4-LE",
    "EEG T6-LE",
    "EEG Cz-LE",
    "EEG Pz-LE",
    "EEG A2-A1",
    "EEG 23A-23R",
    "EEG 24A-24R",
]


def _norm_ch_name(name: str) -> str:
    s = str(name).upper().strip()
    if s.startswith("EEG "):
        s = s[4:]
    for suffix in ["-LE", "-REF"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s.strip()


DEFAULT_CH_INDEX = {_norm: idx for idx, _norm in enumerate([_norm_ch_name(x) for x in DEFAULT_CH_NAMES])}


def _channel_descriptor(name: str) -> dict[str, str | None]:
    ch = _norm_ch_name(name)
    primary = None
    if ch.startswith(("FP", "F")):
        primary = "frontal"
    elif ch.startswith("C"):
        primary = "central"
    elif ch.startswith("T"):
        primary = "temporal"
    elif ch.startswith("P"):
        primary = "parietal"
    elif ch.startswith("O"):
        primary = "occipital"

    hemi = "midline"
    if ch.endswith(("1", "3", "5", "7")):
        hemi = "left"
    elif ch.endswith(("2", "4", "6", "8")):
        hemi = "right"

    frontal_side = None
    if primary == "frontal" and hemi == "left":
        frontal_side = "frontal_left"
    elif primary == "frontal" and hemi == "right":
        frontal_side = "frontal_right"

    return {"primary": primary, "hemi": hemi, "frontal_side": frontal_side}


def channel_regions(name: str) -> list[int]:
    desc = _channel_descriptor(name)
    keys: list[str] = []
    if desc["primary"] is not None:
        keys.append(str(desc["primary"]))
    keys.append(str(desc["hemi"]))
    if desc["frontal_side"] is not None:
        keys.append(str(desc["frontal_side"]))
    keys.append("global")
    return [REGION_KEYS.index(k) for k in dict.fromkeys(keys)]


def base_feature_region_map(ch_names: list[str], n_bands: int = 5) -> list[list[int]]:
    n_channels = len(ch_names)
    feature_regions: list[list[int]] = []
    channel_map = [channel_regions(ch) for ch in ch_names]

    for channel_idx in range(n_channels):
        for _ in range(n_bands):
            feature_regions.append(channel_map[channel_idx])

    tri_i, tri_j = np.triu_indices(n_channels, k=1)
    for _band_block in range(4):
        for i, j in zip(tri_i, tri_j):
            regs = list(channel_map[int(i)]) + list(channel_map[int(j)])
            primary_i = _channel_descriptor(ch_names[int(i)])["primary"]
            primary_j = _channel_descriptor(ch_names[int(j)])["primary"]
            if primary_i is not None and primary_j is not None and primary_i != primary_j:
                regs.append(REGION_KEYS.index("cross_region"))
            feature_regions.append(sorted(set(regs)))
    return feature_regions


def node_feature_region_map(ch_names: list[str], n_bands: int = 5) -> list[list[int]]:
    out: list[list[int]] = []
    for ch in ch_names:
        regs = channel_regions(ch)
        for _ in range(n_bands):
            out.append(regs)
    return out


def ratio_feature_region_map() -> list[list[int]]:
    out: list[list[int]] = []
    regs_lr = [
        REGION_KEYS.index("left"),
        REGION_KEYS.index("right"),
        REGION_KEYS.index("cross_region"),
        REGION_KEYS.index("global"),
    ]
    regs_flr = [
        REGION_KEYS.index("frontal"),
        REGION_KEYS.index("left"),
        REGION_KEYS.index("right"),
        REGION_KEYS.index("frontal_left"),
        REGION_KEYS.index("frontal_right"),
        REGION_KEYS.index("cross_region"),
        REGION_KEYS.index("global"),
    ]
    for _ in range(5):
        out.append(regs_lr)
    for _ in range(5):
        out.append(regs_flr)
    return out


def hemisphere_indices(ch_names: list[str]) -> dict[str, list[int]]:
    out = {"left": [], "right": [], "frontal_left": [], "frontal_right": []}
    for idx, name in enumerate(ch_names):
        desc = _channel_descriptor(name)
        if desc["hemi"] in {"left", "right"}:
            out[str(desc["hemi"])].append(int(idx))
        if desc["frontal_side"] is not None:
            out[str(desc["frontal_side"])].append(int(idx))
    return out


def expand_node_features(node_x: np.ndarray, ch_names: list[str]) -> np.ndarray:
    out = np.zeros((len(DEFAULT_CH_NAMES), int(node_x.shape[1])), dtype=np.float32)
    for src_idx, name in enumerate(ch_names):
        dst_idx = DEFAULT_CH_INDEX.get(_norm_ch_name(name))
        if dst_idx is not None:
            out[int(dst_idx)] = node_x[int(src_idx)]
    return out


def expand_adj(adj: np.ndarray, ch_names: list[str]) -> np.ndarray:
    out = np.zeros((len(DEFAULT_CH_NAMES), len(DEFAULT_CH_NAMES)), dtype=np.float32)
    present: list[int] = []
    src_rows: list[int] = []
    for src_idx, name in enumerate(ch_names):
        dst_idx = DEFAULT_CH_INDEX.get(_norm_ch_name(name))
        if dst_idx is not None:
            present.append(int(dst_idx))
            src_rows.append(int(src_idx))
    for src_row, dst_row in zip(src_rows, present):
        for src_col, dst_col in zip(src_rows, present):
            out[int(dst_row), int(dst_col)] = float(adj[int(src_row), int(src_col)])
    return out


def ratio_window_features(node_x: np.ndarray, ch_names: list[str]) -> np.ndarray:
    hemi = hemisphere_indices(ch_names)
    eps = 1e-6
    feats: list[float] = []
    for band_idx in range(node_x.shape[1]):
        left = float(np.mean(node_x[hemi["left"], band_idx])) if hemi["left"] else 0.0
        right = float(np.mean(node_x[hemi["right"], band_idx])) if hemi["right"] else 0.0
        feats.append(float(np.log(right + eps) - np.log(left + eps)))
    for band_idx in range(node_x.shape[1]):
        fl = float(np.mean(node_x[hemi["frontal_left"], band_idx])) if hemi["frontal_left"] else 0.0
        fr = float(np.mean(node_x[hemi["frontal_right"], band_idx])) if hemi["frontal_right"] else 0.0
        feats.append(float(np.log(fr + eps) - np.log(fl + eps)))
    return np.asarray(feats, dtype=np.float32)


def _window_starts(n_samples: int, win: int, step: int) -> list[int]:
    if n_samples <= win:
        return [0]
    starts = list(range(0, n_samples - win + 1, step))
    if starts[-1] != n_samples - win:
        starts.append(n_samples - win)
    return starts


def summarize_window_stack(stacked: np.ndarray) -> np.ndarray:
    return np.concatenate([stacked.mean(axis=0), stacked.std(axis=0), stacked[-1] - stacked[0]]).astype(np.float32)


def edge_to_adj(n_nodes: int, edge_index: np.ndarray, edge_weight: np.ndarray) -> np.ndarray:
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for (src, dst), weight in zip(edge_index.T, edge_weight):
        adj[int(src), int(dst)] = float(weight)
    return adj


def parse_file_record(path: Path) -> dict[str, object] | None:
    upper = path.stem.upper()
    state = next((c for c in STATE_KEYS if c in upper), None)
    if state is None:
        return None
    subj_match = re.search(r"S\s*(\d+)", upper)
    if not subj_match:
        return None
    subj = int(subj_match.group(1))
    if "MDD" in upper:
        label_name = "MDD"
        label = 1
    elif re.search(r"(^|_|\s)H\s*S", upper):
        label_name = "H"
        label = 0
    else:
        return None
    subject_id = f"{label_name}_S{subj:02d}"
    return {
        "subject_id": subject_id,
        "label": label,
        "label_name": label_name,
        "state": state,
        "file_path": str(path),
        "file_size": int(path.stat().st_size),
    }


def collect_state_records(raw_dir: Path) -> list[dict[str, object]]:
    dedup: dict[tuple[str, str], dict[str, object]] = {}
    for path in sorted(raw_dir.glob("*.edf")):
        rec = parse_file_record(path)
        if rec is None:
            continue
        key = (str(rec["subject_id"]), str(rec["state"]))
        prev = dedup.get(key)
        if prev is None or int(rec["file_size"]) > int(prev["file_size"]):
            dedup[key] = rec
    return sorted(dedup.values(), key=lambda x: (str(x["subject_id"]), str(x["state"])))


def _extract_condition_features(
    file_path: Path,
    max_seconds: int,
    window_seconds: float,
    step_seconds: float,
    max_windows: int,
    top_k_per_node: int,
) -> dict[str, object]:
    cfg = GraphConfig(max_seconds=max_seconds)
    data, sfreq, ch_names = read_eeg(file_path, cfg)
    node_x_raw = extract_node_features(data, sfreq).astype(np.float32)
    node_x = expand_node_features(node_x_raw, ch_names)
    tri = np.triu_indices(len(DEFAULT_CH_NAMES), k=1)

    def sparse_adj(edge_mode: str, plv_band: str = "broad") -> np.ndarray:
        edge_index, edge_weight = build_edges(
            data=data,
            sfreq=sfreq,
            edge_mode=edge_mode,
            quantile=0.8,
            min_edges=30,
            edge_selection="per_node_topk",
            top_k_per_node=int(top_k_per_node),
            signed_topk_split=(edge_mode == "pcc"),
            fusion_alpha=0.5,
            plv_band=plv_band,
        )
        adj_raw = edge_to_adj(n_nodes=int(data.shape[0]), edge_index=edge_index, edge_weight=edge_weight)
        return expand_adj(adj_raw, ch_names)

    static_vec = np.concatenate(
        [
            node_x.reshape(-1),
            sparse_adj("pcc")[tri],
            sparse_adj("plv", "theta")[tri],
            sparse_adj("plv", "alpha")[tri],
            sparse_adj("plv", "beta")[tri],
        ]
    ).astype(np.float32)

    static_ratio = ratio_window_features(node_x, DEFAULT_CH_NAMES)
    win = max(1, int(round(float(window_seconds) * sfreq)))
    step = max(1, int(round(float(step_seconds) * sfreq)))
    starts = _window_starts(data.shape[1], win, step)[: int(max_windows)]
    node_windows: list[np.ndarray] = []
    ratio_windows: list[np.ndarray] = []
    for start in starts:
        stop = min(start + win, data.shape[1])
        segment = data[:, start:stop]
        if segment.shape[1] < 8:
            continue
        window_node_x = expand_node_features(extract_node_features(segment, sfreq).astype(np.float32), ch_names)
        node_windows.append(window_node_x.reshape(-1))
        ratio_windows.append(ratio_window_features(window_node_x, DEFAULT_CH_NAMES))
    if not node_windows:
        node_windows.append(node_x.reshape(-1))
        ratio_windows.append(static_ratio)

    temporal_node = summarize_window_stack(np.stack(node_windows).astype(np.float32))
    temporal_ratio = summarize_window_stack(np.stack(ratio_windows).astype(np.float32))
    return {
        "file_path": str(file_path),
        "ch_names": list(ch_names),
        "static_vec": static_vec,
        "temporal_node": temporal_node,
        "static_ratio": static_ratio.astype(np.float32),
        "temporal_ratio": temporal_ratio.astype(np.float32),
        "n_windows": int(len(node_windows)),
    }


def load_or_build_feature_cache(args: argparse.Namespace) -> list[dict[str, object]]:
    states = [x.strip().upper() for x in str(args.cache_states).split(",") if x.strip()]
    meta_expect = {
        "raw_dir": str(args.raw_dir),
        "states": states,
        "max_seconds": int(args.max_seconds),
        "window_seconds": float(args.window_seconds),
        "step_seconds": float(args.step_seconds),
        "max_windows": int(args.max_windows),
        "top_k_per_node": int(args.top_k_per_node),
    }
    if args.items_cache_path.exists():
        cache_obj = torch.load(args.items_cache_path, map_location="cpu", weights_only=False)
        if isinstance(cache_obj, dict) and cache_obj.get("meta") == meta_expect and "items" in cache_obj:
            print(f"Loaded multistate cache: {args.items_cache_path}")
            return cache_obj["items"]

    raw_records = collect_state_records(args.raw_dir)
    items: list[dict[str, object]] = []
    for idx, rec in enumerate(raw_records, start=1):
        state = str(rec["state"])
        if state not in states:
            continue
        subject_id = str(rec["subject_id"])
        fp = Path(str(rec["file_path"]))
        print(f"[{idx}/{len(raw_records)}] multistate feature build: {subject_id} {state} {fp.name}")
        feats = _extract_condition_features(
            file_path=fp,
            max_seconds=int(args.max_seconds),
            window_seconds=float(args.window_seconds),
            step_seconds=float(args.step_seconds),
            max_windows=int(args.max_windows),
            top_k_per_node=int(args.top_k_per_node),
        )
        items.append(
            {
                "subject_id": subject_id,
                "label": int(rec["label"]),
                "label_name": str(rec["label_name"]),
                "state": state,
                **feats,
            }
        )

    args.items_cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta_expect, "items": items}, args.items_cache_path)
    print(f"Saved multistate cache: {args.items_cache_path}")
    return items


def build_subject_multistate_table(
    items: list[dict[str, object]],
    states: list[str],
    eligible_states: list[str],
    include_pairwise_contrasts: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]], list[int], list[int], list[str], list[str]]:
    by_subject: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    labels: dict[str, int] = {}
    for item in items:
        subject_id = str(item["subject_id"])
        state = str(item["state"])
        by_subject[subject_id][state] = item
        labels[subject_id] = int(item["label"])

    subject_ids = sorted(
        [
            subject_id
            for subject_id, state_map in by_subject.items()
            if all(state in state_map for state in eligible_states) and all(state in state_map for state in states)
        ]
    )
    if not subject_ids:
        raise RuntimeError("No subjects matched the requested multistate eligibility filter.")

    contrast_pairs = [(states[i], states[j]) for i in range(len(states)) for j in range(i + 1, len(states))]
    group_keys: list[str] = []
    for state in states:
        prefix = state.lower()
        group_keys.extend(
            [
                f"{prefix}_static_raw",
                f"{prefix}_temporal_mean",
                f"{prefix}_temporal_std",
                f"{prefix}_temporal_delta",
                f"{prefix}_static_ratio",
                f"{prefix}_temporal_ratio_mean",
                f"{prefix}_temporal_ratio_std",
                f"{prefix}_temporal_ratio_delta",
            ]
        )
    state_keys = list(states)
    if include_pairwise_contrasts:
        for state_a, state_b in contrast_pairs:
            prefix = f"{state_a.lower()}_minus_{state_b.lower()}"
            group_keys.extend(
                [
                    f"{prefix}_temporal_mean",
                    f"{prefix}_temporal_std",
                    f"{prefix}_temporal_delta",
                    f"{prefix}_static_ratio",
                    f"{prefix}_temporal_ratio_mean",
                    f"{prefix}_temporal_ratio_std",
                    f"{prefix}_temporal_ratio_delta",
                ]
            )
            state_keys.append(f"{state_a}-{state_b}")

    group_index = {name: idx for idx, name in enumerate(group_keys)}
    state_index = {state: idx for idx, state in enumerate(state_keys)}

    base_map = base_feature_region_map(DEFAULT_CH_NAMES)
    temporal_node_map = node_feature_region_map(DEFAULT_CH_NAMES)
    ratio_map = ratio_feature_region_map()

    rows: list[np.ndarray] = []
    y: list[int] = []
    feature_region_map: list[list[int]] = []
    feature_group_ids: list[int] = []
    feature_state_ids: list[int] = []

    for subject_id in subject_ids:
        row_blocks: list[np.ndarray] = []
        state_blocks: dict[str, dict[str, np.ndarray]] = {}
        for state in states:
            item = by_subject[subject_id][state]
            state_static = np.asarray(item["static_vec"], dtype=np.float32)
            state_temporal = np.asarray(item["temporal_node"], dtype=np.float32)
            state_ratio = np.asarray(item["static_ratio"], dtype=np.float32)
            state_temporal_ratio = np.asarray(item["temporal_ratio"], dtype=np.float32)
            state_blocks[state] = {
                "static": state_static,
                "temporal": state_temporal,
                "ratio": state_ratio,
                "temporal_ratio": state_temporal_ratio,
            }

            row_blocks.append(state_static)
            row_blocks.append(state_temporal[:110])
            row_blocks.append(state_temporal[110:220])
            row_blocks.append(state_temporal[220:330])
            row_blocks.append(state_ratio)
            row_blocks.append(state_temporal_ratio[:10])
            row_blocks.append(state_temporal_ratio[10:20])
            row_blocks.append(state_temporal_ratio[20:30])

        if include_pairwise_contrasts:
            for state_a, state_b in contrast_pairs:
                block_a = state_blocks[state_a]
                block_b = state_blocks[state_b]
                row_blocks.append(block_a["temporal"][:110] - block_b["temporal"][:110])
                row_blocks.append(block_a["temporal"][110:220] - block_b["temporal"][110:220])
                row_blocks.append(block_a["temporal"][220:330] - block_b["temporal"][220:330])
                row_blocks.append(block_a["ratio"] - block_b["ratio"])
                row_blocks.append(block_a["temporal_ratio"][:10] - block_b["temporal_ratio"][:10])
                row_blocks.append(block_a["temporal_ratio"][10:20] - block_b["temporal_ratio"][10:20])
                row_blocks.append(block_a["temporal_ratio"][20:30] - block_b["temporal_ratio"][20:30])

        rows.append(np.concatenate(row_blocks).astype(np.float32))
        y.append(int(labels[subject_id]))

    for state in states:
        sid = state_index[state]
        prefix = state.lower()

        feature_region_map.extend(base_map)
        feature_group_ids.extend([group_index[f"{prefix}_static_raw"]] * len(base_map))
        feature_state_ids.extend([sid] * len(base_map))

        feature_region_map.extend(temporal_node_map)
        feature_group_ids.extend([group_index[f"{prefix}_temporal_mean"]] * 110)
        feature_state_ids.extend([sid] * 110)

        feature_region_map.extend(temporal_node_map)
        feature_group_ids.extend([group_index[f"{prefix}_temporal_std"]] * 110)
        feature_state_ids.extend([sid] * 110)

        feature_region_map.extend(temporal_node_map)
        feature_group_ids.extend([group_index[f"{prefix}_temporal_delta"]] * 110)
        feature_state_ids.extend([sid] * 110)

        feature_region_map.extend(ratio_map)
        feature_group_ids.extend([group_index[f"{prefix}_static_ratio"]] * 10)
        feature_state_ids.extend([sid] * 10)

        feature_region_map.extend(ratio_map)
        feature_group_ids.extend([group_index[f"{prefix}_temporal_ratio_mean"]] * 10)
        feature_state_ids.extend([sid] * 10)

        feature_region_map.extend(ratio_map)
        feature_group_ids.extend([group_index[f"{prefix}_temporal_ratio_std"]] * 10)
        feature_state_ids.extend([sid] * 10)

        feature_region_map.extend(ratio_map)
        feature_group_ids.extend([group_index[f"{prefix}_temporal_ratio_delta"]] * 10)
        feature_state_ids.extend([sid] * 10)

    if include_pairwise_contrasts:
        for state_a, state_b in contrast_pairs:
            sid = state_index[f"{state_a}-{state_b}"]
            prefix = f"{state_a.lower()}_minus_{state_b.lower()}"

            feature_region_map.extend(temporal_node_map)
            feature_group_ids.extend([group_index[f"{prefix}_temporal_mean"]] * 110)
            feature_state_ids.extend([sid] * 110)

            feature_region_map.extend(temporal_node_map)
            feature_group_ids.extend([group_index[f"{prefix}_temporal_std"]] * 110)
            feature_state_ids.extend([sid] * 110)

            feature_region_map.extend(temporal_node_map)
            feature_group_ids.extend([group_index[f"{prefix}_temporal_delta"]] * 110)
            feature_state_ids.extend([sid] * 110)

            feature_region_map.extend(ratio_map)
            feature_group_ids.extend([group_index[f"{prefix}_static_ratio"]] * 10)
            feature_state_ids.extend([sid] * 10)

            feature_region_map.extend(ratio_map)
            feature_group_ids.extend([group_index[f"{prefix}_temporal_ratio_mean"]] * 10)
            feature_state_ids.extend([sid] * 10)

            feature_region_map.extend(ratio_map)
            feature_group_ids.extend([group_index[f"{prefix}_temporal_ratio_std"]] * 10)
            feature_state_ids.extend([sid] * 10)

            feature_region_map.extend(ratio_map)
            feature_group_ids.extend([group_index[f"{prefix}_temporal_ratio_delta"]] * 10)
            feature_state_ids.extend([sid] * 10)

    x = np.stack(rows).astype(np.float32)
    if len(feature_region_map) != int(x.shape[1]) or len(feature_group_ids) != int(x.shape[1]) or len(feature_state_ids) != int(x.shape[1]):
        raise ValueError("Multistate feature maps do not match feature width.")
    return (
        np.asarray(subject_ids),
        np.asarray(y, dtype=np.int64),
        x,
        feature_region_map,
        feature_group_ids,
        feature_state_ids,
        group_keys,
        state_keys,
    )


def build_region_aggregates(x: np.ndarray, feature_region_map: list[list[int]]) -> np.ndarray:
    out = np.zeros((x.shape[0], len(REGION_KEYS)), dtype=np.float32)
    counts = np.zeros((len(REGION_KEYS),), dtype=np.float32)
    for feat_idx, regions in enumerate(feature_region_map):
        if not regions:
            continue
        scale = 1.0 / float(len(regions))
        for region_idx in regions:
            out[:, int(region_idx)] += x[:, feat_idx] * scale
            counts[int(region_idx)] += scale
    counts = np.where(counts < 1e-6, 1.0, counts)
    return (out / counts.reshape(1, -1)).astype(np.float32)


def build_group_aggregates(x: np.ndarray, feature_group_ids: list[int], n_groups: int) -> np.ndarray:
    out = np.zeros((x.shape[0], n_groups), dtype=np.float32)
    counts = np.zeros((n_groups,), dtype=np.float32)
    for feat_idx, group_idx in enumerate(feature_group_ids):
        out[:, int(group_idx)] += x[:, feat_idx]
        counts[int(group_idx)] += 1.0
    counts = np.where(counts < 1e-6, 1.0, counts)
    return (out / counts.reshape(1, -1)).astype(np.float32)


def build_state_region_aggregates(
    x: np.ndarray,
    feature_region_map: list[list[int]],
    feature_state_ids: list[int],
    n_states: int,
) -> np.ndarray:
    n_regions = len(REGION_KEYS)
    out = np.zeros((x.shape[0], n_states * n_regions), dtype=np.float32)
    counts = np.zeros((n_states * n_regions,), dtype=np.float32)
    for feat_idx, regions in enumerate(feature_region_map):
        state_idx = int(feature_state_ids[feat_idx])
        if not regions:
            regions = [REGION_KEYS.index("global")]
        scale = 1.0 / float(len(regions))
        for region_idx in regions:
            dst_idx = (state_idx * n_regions) + int(region_idx)
            out[:, dst_idx] += x[:, feat_idx] * scale
            counts[dst_idx] += scale
    counts = np.where(counts < 1e-6, 1.0, counts)
    return (out / counts.reshape(1, -1)).astype(np.float32)


def build_state_aggregates(x: np.ndarray, feature_state_ids: list[int], n_states: int) -> np.ndarray:
    out = np.zeros((x.shape[0], n_states), dtype=np.float32)
    counts = np.zeros((n_states,), dtype=np.float32)
    for feat_idx, state_idx in enumerate(feature_state_ids):
        out[:, int(state_idx)] += x[:, feat_idx]
        counts[int(state_idx)] += 1.0
    counts = np.where(counts < 1e-6, 1.0, counts)
    return (out / counts.reshape(1, -1)).astype(np.float32)


def build_graph_edges(
    feature_region_map: list[list[int]],
    feature_group_ids: list[int],
    feature_state_ids: list[int],
    n_feature_nodes: int,
    n_state_region_nodes: int,
    n_groups: int,
    n_states: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_regions = len(REGION_KEYS)
    region_offset = int(n_feature_nodes)
    state_region_offset = int(region_offset + n_regions)
    group_offset = int(state_region_offset + n_state_region_nodes)
    state_offset = int(group_offset + n_groups)
    global_node = int(state_offset + n_states)

    src: list[int] = []
    dst: list[int] = []
    weights: list[float] = []

    for feature_idx, regions in enumerate(feature_region_map):
        unique_regions = sorted(set(int(r) for r in regions))
        if not unique_regions:
            unique_regions = [REGION_KEYS.index("global")]
        scale = 1.0 / float(len(unique_regions))
        for region_idx in unique_regions:
            region_node = region_offset + region_idx
            src.extend([feature_idx, region_node])
            dst.extend([region_node, feature_idx])
            weights.extend([scale, scale])

            state_region_node = state_region_offset + (int(feature_state_ids[feature_idx]) * n_regions) + region_idx
            src.extend([feature_idx, state_region_node])
            dst.extend([state_region_node, feature_idx])
            weights.extend([scale, scale])

        group_node = group_offset + int(feature_group_ids[feature_idx])
        src.extend([feature_idx, group_node])
        dst.extend([group_node, feature_idx])
        weights.extend([1.0, 1.0])

        state_node = state_offset + int(feature_state_ids[feature_idx])
        src.extend([feature_idx, state_node])
        dst.extend([state_node, feature_idx])
        weights.extend([1.0, 1.0])

    for region_idx in range(n_regions):
        region_node = region_offset + region_idx
        src.extend([region_node, global_node])
        dst.extend([global_node, region_node])
        weights.extend([1.0, 1.0])

    for state_region_idx in range(n_state_region_nodes):
        state_region_node = state_region_offset + state_region_idx
        src.extend([state_region_node, global_node])
        dst.extend([global_node, state_region_node])
        weights.extend([1.0, 1.0])

    for group_idx in range(n_groups):
        group_node = group_offset + group_idx
        src.extend([group_node, global_node])
        dst.extend([global_node, group_node])
        weights.extend([1.0, 1.0])

    for state_idx in range(n_states):
        state_node = state_offset + state_idx
        src.extend([state_node, global_node])
        dst.extend([global_node, state_node])
        weights.extend([1.0, 1.0])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


class MultistateNodeDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> None:
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.subject_ids = np.asarray(subject_ids)
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.n_nodes = int(self.x.shape[1]) + 1
        self.node_id = torch.arange(self.n_nodes, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Data:
        values = np.concatenate([self.x[idx], np.array([0.0], dtype=np.float32)], axis=0).reshape(-1, 1)
        return Data(
            x=torch.tensor(values, dtype=torch.float32),
            node_id=self.node_id.clone(),
            edge_index=self.edge_index.clone(),
            edge_attr=self.edge_weight.clone(),
            y=torch.tensor([int(self.y[idx])], dtype=torch.long),
            subject_id=str(self.subject_ids[idx]),
        )


class MultistateExplicitRegionTemporalWeightedStarGNN(nn.Module):
    def __init__(
        self,
        n_feature_nodes: int,
        n_region_nodes: int,
        n_state_region_nodes: int,
        n_group_nodes: int,
        n_state_nodes: int,
    ) -> None:
        super().__init__()
        self.n_feature_nodes = int(n_feature_nodes)
        self.n_region_nodes = int(n_region_nodes)
        self.n_state_region_nodes = int(n_state_region_nodes)
        self.n_group_nodes = int(n_group_nodes)
        self.n_state_nodes = int(n_state_nodes)
        self.n_nodes = (
            self.n_feature_nodes
            + self.n_region_nodes
            + self.n_state_region_nodes
            + self.n_group_nodes
            + self.n_state_nodes
            + 1
        )
        self.feature_logits = nn.Embedding(self.n_feature_nodes, 2)
        self.region_logits = nn.Embedding(self.n_region_nodes, 2)
        self.state_region_logits = nn.Embedding(self.n_state_region_nodes, 2)
        self.group_logits = nn.Embedding(self.n_group_nodes, 2)
        self.state_logits = nn.Embedding(self.n_state_nodes, 2)
        self.bias = nn.Parameter(torch.zeros(2))
        nn.init.zeros_(self.feature_logits.weight)
        nn.init.zeros_(self.region_logits.weight)
        nn.init.zeros_(self.state_region_logits.weight)
        nn.init.zeros_(self.group_logits.weight)
        nn.init.zeros_(self.state_logits.weight)

    def load_logistic_parameters(self, coef: np.ndarray, intercept: np.ndarray) -> None:
        coef = np.asarray(coef, dtype=np.float32).reshape(-1)
        intercept = np.asarray(intercept, dtype=np.float32).reshape(-1)
        expected_dim = self.n_feature_nodes + self.n_region_nodes + self.n_state_region_nodes + self.n_group_nodes + self.n_state_nodes
        if int(coef.shape[0]) != int(expected_dim):
            raise ValueError("Coefficient dimension mismatch.")
        feat_end = self.n_feature_nodes
        reg_end = feat_end + self.n_region_nodes
        state_reg_end = reg_end + self.n_state_region_nodes
        group_end = state_reg_end + self.n_group_nodes
        state_end = group_end + self.n_state_nodes

        feat_w = torch.zeros((self.n_feature_nodes, 2), dtype=torch.float32)
        reg_w = torch.zeros((self.n_region_nodes, 2), dtype=torch.float32)
        state_reg_w = torch.zeros((self.n_state_region_nodes, 2), dtype=torch.float32)
        group_w = torch.zeros((self.n_group_nodes, 2), dtype=torch.float32)
        state_w = torch.zeros((self.n_state_nodes, 2), dtype=torch.float32)
        feat_w[:, 1] = torch.tensor(coef[:feat_end], dtype=torch.float32)
        reg_w[:, 1] = torch.tensor(coef[feat_end:reg_end], dtype=torch.float32)
        state_reg_w[:, 1] = torch.tensor(coef[reg_end:state_reg_end], dtype=torch.float32)
        group_w[:, 1] = torch.tensor(coef[state_reg_end:group_end], dtype=torch.float32)
        state_w[:, 1] = torch.tensor(coef[group_end:state_end], dtype=torch.float32)

        self.feature_logits.weight.data.copy_(feat_w)
        self.region_logits.weight.data.copy_(reg_w)
        self.state_region_logits.weight.data.copy_(state_reg_w)
        self.group_logits.weight.data.copy_(group_w)
        self.state_logits.weight.data.copy_(state_w)
        self.bias.data.zero_()
        self.bias.data[1] = float(intercept[0])

    def forward(self, data: Batch) -> torch.Tensor:
        batch_size = int(data.batch.max().item()) + 1 if data.batch.numel() else 0
        values = data.x.view(batch_size, self.n_nodes)
        feature_values = values[:, : self.n_feature_nodes]
        region_start = self.n_feature_nodes
        region_stop = region_start + self.n_region_nodes
        state_region_start = region_stop
        state_region_stop = state_region_start + self.n_state_region_nodes
        group_start = state_region_stop
        group_stop = group_start + self.n_group_nodes
        state_start = group_stop
        state_stop = state_start + self.n_state_nodes
        region_values = values[:, region_start:region_stop]
        state_region_values = values[:, state_region_start:state_region_stop]
        group_values = values[:, group_start:group_stop]
        state_values = values[:, state_start:state_stop]
        return (
            (feature_values @ self.feature_logits.weight)
            + (region_values @ self.region_logits.weight)
            + (state_region_values @ self.state_region_logits.weight)
            + (group_values @ self.group_logits.weight)
            + (state_values @ self.state_logits.weight)
            + self.bias
        )


def collate_graphs(batch: list[Data]) -> Batch:
    return Batch.from_data_list(batch)


@torch.no_grad()
def predict(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        y_all.append(batch.y.view(-1).detach().cpu().numpy())
        p_all.append(prob)
    return np.concatenate(y_all), np.concatenate(p_all)


def validation_score(
    metric_name: str,
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    tune_threshold: bool,
    thr_low: float,
    thr_high: float,
    thr_step: float,
) -> tuple[float, float]:
    if metric_name == "roc_auc":
        return float(roc_auc_score(y_true, prob_pos)), 0.5

    thresholds = [0.5]
    if tune_threshold:
        thresholds = list(np.arange(float(thr_low), float(thr_high) + (0.5 * float(thr_step)), float(thr_step)))

    best_score = -1.0
    best_thr = 0.5
    for thr in thresholds:
        pred = (prob_pos >= float(thr)).astype(np.int64)
        if metric_name == "accuracy":
            score = float(accuracy_score(y_true, pred))
        elif metric_name == "balanced_accuracy":
            score = float(balanced_accuracy_score(y_true, pred))
        else:
            raise ValueError(f"Unsupported selection metric: {metric_name}")
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_score, best_thr


def build_solver(c_value: float, seed: int, penalty: str, class_weight_mode: str) -> Pipeline:
    penalty = str(penalty).lower()
    class_weight = None if str(class_weight_mode).lower() == "none" else "balanced"
    if penalty == "l1":
        clf = LogisticRegression(
            max_iter=5000,
            class_weight=class_weight,
            C=float(c_value),
            random_state=int(seed),
            penalty="l1",
            solver="liblinear",
        )
    elif penalty == "l2":
        clf = LogisticRegression(
            max_iter=5000,
            class_weight=class_weight,
            C=float(c_value),
            random_state=int(seed),
            solver="liblinear",
        )
    else:
        raise ValueError(f"Unsupported penalty: {penalty}")
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repeated 5x10 CV for multistate explicit-region temporal pure weighted-star GNN.")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw/mdd-patients-eeg-dataset"))
    p.add_argument("--states", type=str, default="TASK,EC,EO")
    p.add_argument("--eligible-states", type=str, default="")
    p.add_argument("--cache-states", type=str, default="TASK,EC,EO")
    p.add_argument(
        "--items-cache-path",
        type=Path,
        default=Path("outputs/cache/multistate_task_ec_eo_features_ms120_w8p0_s8p0_mw12_topk4.pt"),
    )
    p.add_argument("--window-seconds", type=float, default=8.0)
    p.add_argument("--step-seconds", type=float, default=8.0)
    p.add_argument("--max-windows", type=int, default=12)
    p.add_argument("--max-seconds", type=int, default=120)
    p.add_argument("--top-k-per-node", type=int, default=4)
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--c-grid", type=str, default="0.05,0.1,0.2,0.5,1.0,2.0,5.0")
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--selection-metric", type=str, default="balanced_accuracy", choices=["accuracy", "balanced_accuracy", "roc_auc"])
    p.add_argument("--include-pairwise-contrasts", action="store_true")
    p.add_argument("--penalty", type=str, default="l2", choices=["l1", "l2"])
    p.add_argument("--class-weight", type=str, default="balanced", choices=["balanced", "none"])
    p.add_argument("--tune-threshold", action="store_true")
    p.add_argument("--thr-low", type=float, default=0.1)
    p.add_argument("--thr-high", type=float, default=0.9)
    p.add_argument("--thr-step", type=float, default=0.01)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/metrics/runs/multistate_explicit_region_temporal_node_gnn_5x10"),
    )
    p.add_argument("--experiment-name", type=str, default="multistate_explicit_region_temporal_node_gnn_5x10")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    c_grid = [float(x.strip()) for x in str(args.c_grid).split(",") if x.strip()]
    states = [x.strip().upper() for x in str(args.states).split(",") if x.strip()]
    eligible_states = [x.strip().upper() for x in str(args.eligible_states).split(",") if x.strip()] or list(states)

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items = load_or_build_feature_cache(args)
    subject_ids, y, base_x, feature_region_map, feature_group_ids, feature_state_ids, group_keys, state_keys = build_subject_multistate_table(
        items=items,
        states=states,
        eligible_states=eligible_states,
        include_pairwise_contrasts=bool(args.include_pairwise_contrasts),
    )

    region_x = build_region_aggregates(base_x, feature_region_map)
    state_region_x = build_state_region_aggregates(base_x, feature_region_map, feature_state_ids, n_states=len(state_keys))
    group_x = build_group_aggregates(base_x, feature_group_ids, n_groups=len(group_keys))
    state_x = build_state_aggregates(base_x, feature_state_ids, n_states=len(state_keys))
    expanded_x = np.concatenate([base_x, region_x, state_region_x, group_x, state_x], axis=1).astype(np.float32)

    n_feature_nodes = int(base_x.shape[1])
    n_region_nodes = int(region_x.shape[1])
    n_state_region_nodes = int(state_region_x.shape[1])
    n_group_nodes = int(group_x.shape[1])
    n_state_nodes = int(state_x.shape[1])
    edge_index, edge_weight = build_graph_edges(
        feature_region_map=feature_region_map,
        feature_group_ids=feature_group_ids,
        feature_state_ids=feature_state_ids,
        n_feature_nodes=n_feature_nodes,
        n_state_region_nodes=n_state_region_nodes,
        n_groups=n_group_nodes,
        n_states=n_state_nodes,
    )

    run_payloads: list[dict[str, object]] = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=int(seed))
        fold_results: list[dict[str, object]] = []

        for fold_no, (trainval_idx, test_idx) in enumerate(skf.split(expanded_x, y), start=1):
            y_trainval = y[trainval_idx]
            tr_rel_idx, val_rel_idx = train_test_split(
                np.arange(len(trainval_idx)),
                test_size=float(args.val_size),
                random_state=int(seed) + fold_no,
                stratify=y_trainval,
            )
            train_idx = trainval_idx[tr_rel_idx]
            val_idx = trainval_idx[val_rel_idx]

            best_c = None
            best_score = -1.0
            best_auc = -1.0
            best_thr = 0.5
            for c_value in c_grid:
                solver = build_solver(
                    c_value=float(c_value),
                    seed=int(seed),
                    penalty=str(args.penalty),
                    class_weight_mode=str(args.class_weight),
                )
                solver.fit(expanded_x[train_idx], y[train_idx])
                prob_val = solver.predict_proba(expanded_x[val_idx])[:, 1]
                score, thr = validation_score(
                    args.selection_metric,
                    y[val_idx],
                    prob_val,
                    tune_threshold=bool(args.tune_threshold),
                    thr_low=float(args.thr_low),
                    thr_high=float(args.thr_high),
                    thr_step=float(args.thr_step),
                )
                auc = float(roc_auc_score(y[val_idx], prob_val))
                if (score > best_score) or (abs(score - best_score) <= 1e-12 and auc > best_auc):
                    best_c = float(c_value)
                    best_score = float(score)
                    best_auc = float(auc)
                    best_thr = float(thr)

            if best_c is None:
                raise RuntimeError("No validation configuration selected.")

            refit_solver = build_solver(
                c_value=float(best_c),
                seed=int(seed),
                penalty=str(args.penalty),
                class_weight_mode=str(args.class_weight),
            )
            refit_solver.fit(expanded_x[trainval_idx], y[trainval_idx])
            scaler: StandardScaler = refit_solver.named_steps["scaler"]
            clf: LogisticRegression = refit_solver.named_steps["clf"]
            test_z = scaler.transform(expanded_x[test_idx]).astype(np.float32)

            model = MultistateExplicitRegionTemporalWeightedStarGNN(
                n_feature_nodes=n_feature_nodes,
                n_region_nodes=n_region_nodes,
                n_state_region_nodes=n_state_region_nodes,
                n_group_nodes=n_group_nodes,
                n_state_nodes=n_state_nodes,
            ).to(device)
            model.load_logistic_parameters(clf.coef_[0], clf.intercept_)

            test_ds = MultistateNodeDataset(
                x=test_z,
                y=y[test_idx],
                subject_ids=subject_ids[test_idx],
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_graphs)
            y_test, prob_test = predict(model, test_loader, device=device)
            metrics = calc_metrics(y_test, prob_test, threshold=float(best_thr))

            fold_results.append(
                {
                    "fold": fold_no,
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    "n_test": int(len(test_idx)),
                    "selected_c": float(best_c),
                    "selection_metric": str(args.selection_metric),
                    "best_val_score": float(best_score),
                    "best_val_auc": float(best_auc),
                    "selected_threshold": float(best_thr),
                    "test_subject_ids": subject_ids[test_idx].tolist(),
                    "test_metrics": metrics,
                }
            )
            print(
                f"seed={seed} fold={fold_no:02d} "
                f"acc={metrics['accuracy']:.4f} "
                f"bal_acc={metrics['balanced_accuracy']:.4f} "
                f"f1={metrics['f1']:.4f} "
                f"auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'NA'} "
                f"C={best_c:.2f} thr={best_thr:.2f}"
            )

        payload = {
            "schema_version": "eeg_mdd_multistate_explicit_region_temporal_node_gnn_5x10_v1",
            "experiment_name": args.experiment_name,
            "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model": "MultistateExplicitRegionTemporalWeightedStarGNN",
            "device": str(device),
            "params": {
                "folds": int(args.folds),
                "seed": int(seed),
                "seeds": [int(x) for x in seeds],
                "states": states,
                "eligible_states": eligible_states,
                "c_grid": [float(x) for x in c_grid],
                "val_size": float(args.val_size),
                "selection_metric": str(args.selection_metric),
                "include_pairwise_contrasts": bool(args.include_pairwise_contrasts),
                "penalty": str(args.penalty),
                "class_weight": str(args.class_weight),
                "tune_threshold": bool(args.tune_threshold),
                "thr_low": float(args.thr_low),
                "thr_high": float(args.thr_high),
                "thr_step": float(args.thr_step),
                "window_seconds": float(args.window_seconds),
                "step_seconds": float(args.step_seconds),
                "max_windows": int(args.max_windows),
                "max_seconds": int(args.max_seconds),
                "top_k_per_node": int(args.top_k_per_node),
                "region_keys": REGION_KEYS,
                "group_keys": group_keys,
                "state_keys": state_keys,
            },
            "feature_shape": {
                "n_subjects": int(expanded_x.shape[0]),
                "n_feature_nodes": int(n_feature_nodes),
                "n_region_nodes": int(n_region_nodes),
                "n_state_region_nodes": int(n_state_region_nodes),
                "n_group_nodes": int(n_group_nodes),
                "n_state_nodes": int(n_state_nodes),
                "n_total_nodes_without_global": int(expanded_x.shape[1]),
            },
            "aggregate": aggregate_fold_results(fold_results),
            "fold_results": fold_results,
        }
        out_path = args.out_dir / f"multistate_explicit_region_temporal_node_gnn_seed{seed}_cv10.json"
        write_json(out_path, payload)
        run_payloads.append({**payload, "_out_path": str(out_path)})
        print(f"saved: {out_path}")

    summary = summarize_run_payloads(run_payloads)
    summary_payload = {
        "schema_version": "eeg_mdd_multistate_explicit_region_temporal_node_gnn_5x10_summary_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "MultistateExplicitRegionTemporalWeightedStarGNN",
        "params": {
            "folds": int(args.folds),
            "seeds": [int(x) for x in seeds],
            "states": states,
            "eligible_states": eligible_states,
            "c_grid": [float(x) for x in c_grid],
            "val_size": float(args.val_size),
            "selection_metric": str(args.selection_metric),
            "include_pairwise_contrasts": bool(args.include_pairwise_contrasts),
            "penalty": str(args.penalty),
            "class_weight": str(args.class_weight),
            "tune_threshold": bool(args.tune_threshold),
            "thr_low": float(args.thr_low),
            "thr_high": float(args.thr_high),
            "thr_step": float(args.thr_step),
            "window_seconds": float(args.window_seconds),
            "step_seconds": float(args.step_seconds),
            "max_windows": int(args.max_windows),
            "max_seconds": int(args.max_seconds),
            "top_k_per_node": int(args.top_k_per_node),
            "region_keys": REGION_KEYS,
            "group_keys": group_keys,
            "state_keys": state_keys,
        },
        "feature_shape": {
            "n_subjects": int(expanded_x.shape[0]),
            "n_feature_nodes": int(n_feature_nodes),
            "n_region_nodes": int(n_region_nodes),
            "n_state_region_nodes": int(n_state_region_nodes),
            "n_group_nodes": int(n_group_nodes),
            "n_state_nodes": int(n_state_nodes),
            "n_total_nodes_without_global": int(expanded_x.shape[1]),
        },
        **summary,
    }
    summary_path = args.out_dir / "summary_5x10.json"
    write_json(summary_path, summary_payload)
    print(f"\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
