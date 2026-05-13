from __future__ import annotations

import argparse
from datetime import datetime
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

import train_gnn_dualgraph_multiband_spatiotemporal_cv10 as temporal_mod
from feature_model_utils import (
    aggregate_fold_results,
    build_static_graph_feature_table,
    calc_metrics,
    parse_seed_list,
    summarize_run_payloads,
    write_json,
)

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
GROUP_KEYS = [
    "static_raw",
    "temporal_node_mean",
    "temporal_node_std",
    "temporal_node_delta",
    "temporal_conn_mean",
    "temporal_conn_std",
    "temporal_conn_delta",
    "static_ratio",
    "temporal_ratio_mean",
    "temporal_ratio_std",
    "temporal_ratio_delta",
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
PRIMARY_REGION_PAIRS = list(combinations_with_replacement(PRIMARY_REGION_KEYS, 2))


def _norm_ch_name(name: str) -> str:
    s = str(name).upper().strip()
    if s.startswith("EEG "):
        s = s[4:]
    for suffix in ["-LE", "-REF"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s.strip()


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


def primary_region_indices(ch_names: list[str]) -> dict[str, list[int]]:
    out = {k: [] for k in PRIMARY_REGION_KEYS}
    for idx, name in enumerate(ch_names):
        primary = _channel_descriptor(name)["primary"]
        if primary in out:
            out[str(primary)].append(int(idx))
    return out


def hemisphere_indices(ch_names: list[str]) -> dict[str, list[int]]:
    out = {"left": [], "right": [], "midline": [], "frontal_left": [], "frontal_right": []}
    for idx, name in enumerate(ch_names):
        desc = _channel_descriptor(name)
        out[str(desc["hemi"])].append(int(idx))
        if desc["frontal_side"] is not None:
            out[str(desc["frontal_side"])].append(int(idx))
    return out


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


def conn_window_feature_region_map() -> list[list[int]]:
    out: list[list[int]] = []
    for ra, rb in PRIMARY_REGION_PAIRS:
        regs = [REGION_KEYS.index(ra), REGION_KEYS.index("global")]
        if ra != rb:
            regs.extend([REGION_KEYS.index(rb), REGION_KEYS.index("cross_region")])
        for _ in range(5):
            out.append(sorted(set(regs)))
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


def graph_to_adj(graph: Data) -> np.ndarray:
    n_nodes = int(graph.x.shape[0])
    edge_index = graph.edge_index.detach().cpu().numpy().astype(np.int64)
    edge_weight = graph.edge_attr.detach().cpu().numpy().astype(np.float32)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for (src, dst), weight in zip(edge_index.T, edge_weight):
        adj[int(src), int(dst)] = float(weight)
    return adj


def upper_to_matrix(vec: np.ndarray, n_nodes: int) -> np.ndarray:
    tri = np.triu_indices(n_nodes, k=1)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    adj[tri] = vec.astype(np.float32)
    adj[(tri[1], tri[0])] = vec.astype(np.float32)
    return adj


def pair_mean(mat: np.ndarray, idx_a: list[int], idx_b: list[int]) -> float:
    if not idx_a or not idx_b:
        return 0.0
    if idx_a is idx_b or idx_a == idx_b:
        if len(idx_a) < 2:
            return 0.0
        block = mat[np.ix_(idx_a, idx_a)]
        tri = np.triu_indices(len(idx_a), k=1)
        vals = block[tri]
    else:
        vals = mat[np.ix_(idx_a, idx_b)].reshape(-1)
    if vals.size == 0:
        return 0.0
    return float(np.mean(vals))


def conn_window_features(adjs: dict[str, np.ndarray], ch_names: list[str]) -> np.ndarray:
    primary = primary_region_indices(ch_names)
    feats: list[float] = []
    for ra, rb in PRIMARY_REGION_PAIRS:
        idx_a = primary[ra]
        idx_b = primary[rb]
        feats.append(pair_mean(adjs["pcc"], idx_a, idx_b))
        feats.append(pair_mean(np.abs(adjs["pcc"]), idx_a, idx_b))
        feats.append(pair_mean(adjs["theta"], idx_a, idx_b))
        feats.append(pair_mean(adjs["alpha"], idx_a, idx_b))
        feats.append(pair_mean(adjs["beta"], idx_a, idx_b))
    return np.asarray(feats, dtype=np.float32)


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


def summarize_window_stack(stacked: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            stacked.mean(axis=0),
            stacked.std(axis=0),
            stacked[-1] - stacked[0],
        ]
    ).astype(np.float32)


def load_or_build_temporal_items(args: argparse.Namespace) -> list[dict[str, object]]:
    meta_expect = {
        "split_csv": str(args.split_csv),
        "max_seconds": int(args.max_seconds),
        "window_seconds": float(args.window_seconds),
        "step_seconds": float(args.step_seconds),
        "max_windows": int(args.max_windows),
        "pcc_quantile": float(args.pcc_quantile),
        "min_edges": int(args.min_edges),
        "top_k_per_node": int(args.top_k_per_node),
    }
    if args.items_cache_path.exists():
        cache_obj = torch.load(args.items_cache_path, map_location="cpu", weights_only=False)
        if isinstance(cache_obj, dict) and cache_obj.get("meta") == meta_expect and "items" in cache_obj:
            print(f"Loaded temporal cache: {args.items_cache_path}")
            return cache_obj["items"]

    build_args = argparse.Namespace(
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
    return temporal_mod.build_subject_sequences(build_args)


def build_static_ratio_table(static_x: np.ndarray, ch_names: list[str]) -> np.ndarray:
    n_channels = len(ch_names)
    node_dim = n_channels * 5
    rows: list[np.ndarray] = []
    for row in static_x:
        node_x = row[:node_dim].reshape(n_channels, 5)
        rows.append(ratio_window_features(node_x, ch_names))
    return np.stack(rows).astype(np.float32)


def build_temporal_dynamic_tables(
    items: list[dict[str, object]],
    ch_names: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    subject_ids: list[str] = []
    labels: list[int] = []
    node_rows: list[np.ndarray] = []
    conn_rows: list[np.ndarray] = []
    ratio_rows: list[np.ndarray] = []

    for item in sorted(items, key=lambda x: str(x["subject_id"])):
        subject_ids.append(str(item["subject_id"]))
        labels.append(int(item["label"]))

        node_windows: list[np.ndarray] = []
        conn_windows: list[np.ndarray] = []
        ratio_windows: list[np.ndarray] = []
        for window in item["windows"]:
            node_x = window["pcc"].x.detach().cpu().numpy().astype(np.float32)
            node_windows.append(node_x.reshape(-1))
            adjs = {
                "pcc": graph_to_adj(window["pcc"]),
                "theta": graph_to_adj(window["theta"]),
                "alpha": graph_to_adj(window["alpha"]),
                "beta": graph_to_adj(window["beta"]),
            }
            conn_windows.append(conn_window_features(adjs, ch_names))
            ratio_windows.append(ratio_window_features(node_x, ch_names))

        node_rows.append(summarize_window_stack(np.stack(node_windows).astype(np.float32)))
        conn_rows.append(summarize_window_stack(np.stack(conn_windows).astype(np.float32)))
        ratio_rows.append(summarize_window_stack(np.stack(ratio_windows).astype(np.float32)))

    return (
        np.asarray(subject_ids),
        np.asarray(labels, dtype=np.int64),
        np.stack(node_rows).astype(np.float32),
        np.stack(conn_rows).astype(np.float32),
        np.stack(ratio_rows).astype(np.float32),
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


def build_group_aggregates(x: np.ndarray, feature_group_ids: list[int]) -> np.ndarray:
    out = np.zeros((x.shape[0], len(GROUP_KEYS)), dtype=np.float32)
    counts = np.zeros((len(GROUP_KEYS),), dtype=np.float32)
    for feat_idx, group_idx in enumerate(feature_group_ids):
        out[:, int(group_idx)] += x[:, feat_idx]
        counts[int(group_idx)] += 1.0
    counts = np.where(counts < 1e-6, 1.0, counts)
    return (out / counts.reshape(1, -1)).astype(np.float32)


def build_graph_edges(feature_region_map: list[list[int]], feature_group_ids: list[int], n_feature_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    n_regions = len(REGION_KEYS)
    n_groups = len(GROUP_KEYS)
    region_offset = int(n_feature_nodes)
    group_offset = int(region_offset + n_regions)
    global_node = int(group_offset + n_groups)

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

    for feature_idx, group_idx in enumerate(feature_group_ids):
        group_node = group_offset + int(group_idx)
        src.extend([feature_idx, group_node])
        dst.extend([group_node, feature_idx])
        weights.extend([1.0, 1.0])

    for region_idx in range(n_regions):
        region_node = region_offset + region_idx
        src.extend([region_node, global_node])
        dst.extend([global_node, region_node])
        weights.extend([1.0, 1.0])

    for group_idx in range(n_groups):
        group_node = group_offset + group_idx
        src.extend([group_node, global_node])
        dst.extend([global_node, group_node])
        weights.extend([1.0, 1.0])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


class ExplicitRegionTemporalDynamicsDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, subject_ids: np.ndarray, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> None:
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


class ExplicitRegionTemporalDynamicsWeightedStarGNN(nn.Module):
    def __init__(self, n_feature_nodes: int, n_region_nodes: int, n_group_nodes: int) -> None:
        super().__init__()
        self.n_feature_nodes = int(n_feature_nodes)
        self.n_region_nodes = int(n_region_nodes)
        self.n_group_nodes = int(n_group_nodes)
        self.n_nodes = self.n_feature_nodes + self.n_region_nodes + self.n_group_nodes + 1
        self.feature_logits = nn.Embedding(self.n_feature_nodes, 2)
        self.region_logits = nn.Embedding(self.n_region_nodes, 2)
        self.group_logits = nn.Embedding(self.n_group_nodes, 2)
        self.bias = nn.Parameter(torch.zeros(2))
        nn.init.zeros_(self.feature_logits.weight)
        nn.init.zeros_(self.region_logits.weight)
        nn.init.zeros_(self.group_logits.weight)

    def load_logistic_parameters(self, coef: np.ndarray, intercept: np.ndarray) -> None:
        coef = np.asarray(coef, dtype=np.float32).reshape(-1)
        intercept = np.asarray(intercept, dtype=np.float32).reshape(-1)
        expected = self.n_feature_nodes + self.n_region_nodes + self.n_group_nodes
        if int(coef.shape[0]) != int(expected):
            raise ValueError(f"Coefficient dimension mismatch: {coef.shape[0]} != {expected}")

        feat_end = self.n_feature_nodes
        reg_end = feat_end + self.n_region_nodes
        grp_end = reg_end + self.n_group_nodes

        feat_w = torch.zeros((self.n_feature_nodes, 2), dtype=torch.float32)
        reg_w = torch.zeros((self.n_region_nodes, 2), dtype=torch.float32)
        grp_w = torch.zeros((self.n_group_nodes, 2), dtype=torch.float32)
        feat_w[:, 1] = torch.tensor(coef[:feat_end], dtype=torch.float32)
        reg_w[:, 1] = torch.tensor(coef[feat_end:reg_end], dtype=torch.float32)
        grp_w[:, 1] = torch.tensor(coef[reg_end:grp_end], dtype=torch.float32)

        self.feature_logits.weight.data.copy_(feat_w)
        self.region_logits.weight.data.copy_(reg_w)
        self.group_logits.weight.data.copy_(grp_w)
        self.bias.data.zero_()
        self.bias.data[1] = float(intercept[0])

    def forward(self, data: Batch) -> torch.Tensor:
        batch_size = int(data.batch.max().item()) + 1 if data.batch.numel() else 0
        values = data.x.view(batch_size, self.n_nodes)
        feature_values = values[:, : self.n_feature_nodes]
        region_start = self.n_feature_nodes
        region_stop = region_start + self.n_region_nodes
        group_start = region_stop
        group_stop = group_start + self.n_group_nodes
        region_values = values[:, region_start:region_stop]
        group_values = values[:, group_start:group_stop]
        return (
            (feature_values @ self.feature_logits.weight)
            + (region_values @ self.region_logits.weight)
            + (group_values @ self.group_logits.weight)
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


def select_threshold(y_val: np.ndarray, prob_val: np.ndarray, low: float, high: float, step: float) -> tuple[float, float]:
    thresholds = np.arange(float(low), float(high) + (float(step) * 0.5), float(step), dtype=np.float64)
    best_thr = 0.5
    best_acc = -1.0
    for thr in thresholds:
        acc = float(accuracy_score(y_val, (prob_val >= thr).astype(np.int64)))
        if (acc > best_acc) or (abs(acc - best_acc) <= 1e-12 and abs(thr - 0.5) < abs(best_thr - 0.5)):
            best_acc = acc
            best_thr = float(thr)
    return best_thr, best_acc


def validation_score(metric: str, y_val: np.ndarray, prob_val: np.ndarray, low: float, high: float, step: float) -> tuple[float, float]:
    if str(metric) == "threshold_accuracy":
        thr, acc = select_threshold(y_val, prob_val, low=low, high=high, step=step)
        return float(acc), float(thr)
    if str(metric) == "accuracy":
        return float(accuracy_score(y_val, (prob_val >= 0.5).astype(np.int64))), 0.5
    if str(metric) == "auc":
        if len(np.unique(y_val)) < 2:
            return 0.5, 0.5
        return float(roc_auc_score(y_val, prob_val)), 0.5
    raise ValueError(f"Unsupported validation metric: {metric}")


def parse_float_list(text: str) -> list[float]:
    values = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("Float list must not be empty.")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repeated 5x10 CV for explicit region-temporal dynamics pure weighted-star GNN.")
    p.add_argument("--split-csv", type=Path, default=Path("data/splits/task_split_subject_level.csv"))
    p.add_argument("--pcc-manifest", type=Path, default=Path("data/processed/graphs_pcc_topk_task/manifest.csv"))
    p.add_argument("--theta-manifest", type=Path, default=Path("data/processed/graphs_plv_theta_topk_task/manifest.csv"))
    p.add_argument("--alpha-manifest", type=Path, default=Path("data/processed/graphs_plv_alpha_topk_task/manifest.csv"))
    p.add_argument("--beta-manifest", type=Path, default=Path("data/processed/graphs_plv_beta_topk_task/manifest.csv"))
    p.add_argument("--items-cache-path", type=Path, default=Path("outputs/cache/spatiotemporal_items/ms120_w8p0_s8p0_mw12_q0p8_me30_k4.pt"))
    p.add_argument("--max-seconds", type=int, default=120)
    p.add_argument("--window-seconds", type=float, default=8.0)
    p.add_argument("--step-seconds", type=float, default=8.0)
    p.add_argument("--max-windows", type=int, default=12)
    p.add_argument("--pcc-quantile", type=float, default=0.8)
    p.add_argument("--min-edges", type=int, default=30)
    p.add_argument("--top-k-per-node", type=int, default=4)
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--c-grid", type=str, default="0.05,0.1,0.2,0.5,1.0,2.0,5.0,10.0")
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--selection-metric", type=str, default="threshold_accuracy", choices=["threshold_accuracy", "accuracy", "auc"])
    p.add_argument("--thr-low", type=float, default=0.05)
    p.add_argument("--thr-high", type=float, default=0.95)
    p.add_argument("--thr-step", type=float, default=0.01)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out-dir", type=Path, default=Path("outputs/metrics/runs/explicit_region_temporal_dynamics_node_gnn_5x10"))
    p.add_argument("--experiment-name", type=str, default="explicit_region_temporal_dynamics_node_gnn_5x10")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    c_grid = parse_float_list(args.c_grid)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    static_meta, static_x = build_static_graph_feature_table(
        pcc_manifest=args.pcc_manifest,
        theta_manifest=args.theta_manifest,
        alpha_manifest=args.alpha_manifest,
        beta_manifest=args.beta_manifest,
    )
    static_ratio_x = build_static_ratio_table(static_x, DEFAULT_CH_NAMES)

    temporal_items = load_or_build_temporal_items(args)
    temporal_subject_ids, temporal_labels, temporal_node_x, temporal_conn_x, temporal_ratio_x = build_temporal_dynamic_tables(
        temporal_items,
        DEFAULT_CH_NAMES,
    )

    static_subject_ids = static_meta["subject_id"].to_numpy()
    static_labels = static_meta["label"].to_numpy(dtype=np.int64)
    if list(static_subject_ids) != list(temporal_subject_ids):
        raise ValueError("Static and temporal subject orders do not match.")
    if not np.array_equal(static_labels, temporal_labels):
        raise ValueError("Static and temporal labels do not match.")

    feature_region_map: list[list[int]] = []
    feature_group_ids: list[int] = []
    blocks: list[np.ndarray] = []

    base_map = base_feature_region_map(DEFAULT_CH_NAMES)
    temporal_node_map = node_feature_region_map(DEFAULT_CH_NAMES)
    conn_map = conn_window_feature_region_map()
    ratio_map = ratio_feature_region_map()

    blocks.append(static_x.astype(np.float32))
    feature_region_map.extend(base_map)
    feature_group_ids.extend([GROUP_KEYS.index("static_raw")] * int(static_x.shape[1]))

    blocks.append(temporal_node_x[:, :110].astype(np.float32))
    feature_region_map.extend(temporal_node_map)
    feature_group_ids.extend([GROUP_KEYS.index("temporal_node_mean")] * 110)

    blocks.append(temporal_node_x[:, 110:220].astype(np.float32))
    feature_region_map.extend(temporal_node_map)
    feature_group_ids.extend([GROUP_KEYS.index("temporal_node_std")] * 110)

    blocks.append(temporal_node_x[:, 220:330].astype(np.float32))
    feature_region_map.extend(temporal_node_map)
    feature_group_ids.extend([GROUP_KEYS.index("temporal_node_delta")] * 110)

    blocks.append(temporal_conn_x[:, :75].astype(np.float32))
    feature_region_map.extend(conn_map)
    feature_group_ids.extend([GROUP_KEYS.index("temporal_conn_mean")] * 75)

    blocks.append(temporal_conn_x[:, 75:150].astype(np.float32))
    feature_region_map.extend(conn_map)
    feature_group_ids.extend([GROUP_KEYS.index("temporal_conn_std")] * 75)

    blocks.append(temporal_conn_x[:, 150:225].astype(np.float32))
    feature_region_map.extend(conn_map)
    feature_group_ids.extend([GROUP_KEYS.index("temporal_conn_delta")] * 75)

    blocks.append(static_ratio_x.astype(np.float32))
    feature_region_map.extend(ratio_map)
    feature_group_ids.extend([GROUP_KEYS.index("static_ratio")] * 10)

    blocks.append(temporal_ratio_x[:, :10].astype(np.float32))
    feature_region_map.extend(ratio_map)
    feature_group_ids.extend([GROUP_KEYS.index("temporal_ratio_mean")] * 10)

    blocks.append(temporal_ratio_x[:, 10:20].astype(np.float32))
    feature_region_map.extend(ratio_map)
    feature_group_ids.extend([GROUP_KEYS.index("temporal_ratio_std")] * 10)

    blocks.append(temporal_ratio_x[:, 20:30].astype(np.float32))
    feature_region_map.extend(ratio_map)
    feature_group_ids.extend([GROUP_KEYS.index("temporal_ratio_delta")] * 10)

    base_x = np.concatenate(blocks, axis=1).astype(np.float32)
    if len(feature_region_map) != int(base_x.shape[1]) or len(feature_group_ids) != int(base_x.shape[1]):
        raise ValueError("Feature map lengths do not match base feature width.")

    region_x = build_region_aggregates(base_x, feature_region_map)
    group_x = build_group_aggregates(base_x, feature_group_ids)
    expanded_x = np.concatenate([base_x, region_x, group_x], axis=1).astype(np.float32)

    n_feature_nodes = int(base_x.shape[1])
    n_region_nodes = int(region_x.shape[1])
    n_group_nodes = int(group_x.shape[1])
    edge_index, edge_weight = build_graph_edges(feature_region_map, feature_group_ids, n_feature_nodes=n_feature_nodes)
    y = static_labels
    subject_ids = static_subject_ids

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

            best_solver = None
            best_c = None
            best_thr = 0.5
            best_score = -1.0
            best_tiebreak = -float("inf")

            for c_value in c_grid:
                solver = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "clf",
                            LogisticRegression(
                                max_iter=5000,
                                class_weight="balanced",
                                C=float(c_value),
                                random_state=int(seed),
                            ),
                        ),
                    ]
                )
                solver.fit(expanded_x[train_idx], y[train_idx])
                prob_val = solver.predict_proba(expanded_x[val_idx])[:, 1]
                score, thr_sel = validation_score(
                    args.selection_metric,
                    y[val_idx],
                    prob_val,
                    low=args.thr_low,
                    high=args.thr_high,
                    step=args.thr_step,
                )
                p_clip = np.clip(prob_val, 1e-6, 1.0 - 1e-6)
                tiebreak = -float(log_loss(y[val_idx], p_clip, labels=[0, 1]))
                if (score > best_score) or (abs(score - best_score) <= 1e-12 and tiebreak > best_tiebreak):
                    best_score = float(score)
                    best_tiebreak = float(tiebreak)
                    best_thr = float(thr_sel)
                    best_c = float(c_value)
                    best_solver = solver

            if best_solver is None or best_c is None:
                raise RuntimeError("No validation solver was selected.")

            refit_solver = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=5000,
                            class_weight="balanced",
                            C=float(best_c),
                            random_state=int(seed),
                        ),
                    ),
                ]
            )
            refit_solver.fit(expanded_x[trainval_idx], y[trainval_idx])
            scaler: StandardScaler = refit_solver.named_steps["scaler"]
            clf: LogisticRegression = refit_solver.named_steps["clf"]
            test_z = scaler.transform(expanded_x[test_idx]).astype(np.float32)

            model = ExplicitRegionTemporalDynamicsWeightedStarGNN(
                n_feature_nodes=n_feature_nodes,
                n_region_nodes=n_region_nodes,
                n_group_nodes=n_group_nodes,
            ).to(device)
            model.load_logistic_parameters(clf.coef_[0], clf.intercept_)

            test_ds = ExplicitRegionTemporalDynamicsDataset(
                x=test_z,
                y=y[test_idx],
                subject_ids=subject_ids[test_idx],
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_graphs)
            y_test, prob_test = predict(model, test_loader, device=device)
            metrics = calc_metrics(y_test, prob_test, threshold=best_thr)

            fold_results.append(
                {
                    "fold": fold_no,
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    "n_test": int(len(test_idx)),
                    "selected_c": float(best_c),
                    "selection_metric": args.selection_metric,
                    "best_val_score": float(best_score),
                    "best_val_neg_log_loss": float(best_tiebreak),
                    "test_threshold": float(best_thr),
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
            "schema_version": "eeg_mdd_explicit_region_temporal_dynamics_node_gnn_5x10_v1",
            "experiment_name": args.experiment_name,
            "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model": "ExplicitRegionTemporalDynamicsWeightedStarGNN",
            "device": str(device),
            "params": {
                "folds": int(args.folds),
                "seed": int(seed),
                "seeds": [int(x) for x in seeds],
                "c_grid": [float(x) for x in c_grid],
                "val_size": float(args.val_size),
                "selection_metric": str(args.selection_metric),
                "thr_low": float(args.thr_low),
                "thr_high": float(args.thr_high),
                "thr_step": float(args.thr_step),
                "window_seconds": float(args.window_seconds),
                "step_seconds": float(args.step_seconds),
                "max_windows": int(args.max_windows),
                "max_seconds": int(args.max_seconds),
                "region_keys": REGION_KEYS,
                "group_keys": GROUP_KEYS,
            },
            "feature_shape": {
                "n_subjects": int(expanded_x.shape[0]),
                "n_feature_nodes": int(n_feature_nodes),
                "n_region_nodes": int(n_region_nodes),
                "n_group_nodes": int(n_group_nodes),
                "n_total_nodes_without_global": int(expanded_x.shape[1]),
            },
            "aggregate": aggregate_fold_results(fold_results),
            "fold_results": fold_results,
        }
        out_path = args.out_dir / f"explicit_region_temporal_dynamics_node_gnn_seed{seed}_cv10.json"
        write_json(out_path, payload)
        run_payloads.append({**payload, "_out_path": str(out_path)})
        print(f"saved: {out_path}")

    summary = summarize_run_payloads(run_payloads)
    summary_payload = {
        "schema_version": "eeg_mdd_explicit_region_temporal_dynamics_node_gnn_5x10_summary_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "ExplicitRegionTemporalDynamicsWeightedStarGNN",
        "params": {
            "folds": int(args.folds),
            "seeds": [int(x) for x in seeds],
            "c_grid": [float(x) for x in c_grid],
            "val_size": float(args.val_size),
            "selection_metric": str(args.selection_metric),
            "thr_low": float(args.thr_low),
            "thr_high": float(args.thr_high),
            "thr_step": float(args.thr_step),
            "window_seconds": float(args.window_seconds),
            "step_seconds": float(args.step_seconds),
            "max_windows": int(args.max_windows),
            "max_seconds": int(args.max_seconds),
            "region_keys": REGION_KEYS,
            "group_keys": GROUP_KEYS,
        },
        "feature_shape": {
            "n_subjects": int(expanded_x.shape[0]),
            "n_feature_nodes": int(n_feature_nodes),
            "n_region_nodes": int(n_region_nodes),
            "n_group_nodes": int(n_group_nodes),
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
