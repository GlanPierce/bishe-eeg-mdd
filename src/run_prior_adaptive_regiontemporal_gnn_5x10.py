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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool

from feature_model_utils import (
    aggregate_fold_results,
    build_static_graph_feature_table,
    calc_metrics,
    parse_seed_list,
    summarize_run_payloads,
    write_json,
)


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
TOPOLOGY_PAIRS = [
    ("FP1", "F3"),
    ("F3", "C3"),
    ("C3", "P3"),
    ("P3", "O1"),
    ("FP2", "F4"),
    ("F4", "C4"),
    ("C4", "P4"),
    ("P4", "O2"),
    ("F7", "T3"),
    ("T3", "T5"),
    ("F8", "T4"),
    ("T4", "T6"),
    ("FP1", "F7"),
    ("FP2", "F8"),
    ("F3", "FZ"),
    ("FZ", "F4"),
    ("C3", "CZ"),
    ("CZ", "C4"),
    ("P3", "PZ"),
    ("PZ", "P4"),
    ("F3", "F7"),
    ("F4", "F8"),
    ("C3", "T3"),
    ("C4", "T4"),
    ("P3", "T5"),
    ("P4", "T6"),
    ("O1", "O2"),
]


def load_spatiotemporal_builder():
    src_dir = str(Path("src").resolve())
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    module_path = Path("src/__pycache__/train_gnn_dualgraph_multiband_spatiotemporal_cv10.cpython-311.pyc")
    loader = importlib.machinery.SourcelessFileLoader("prior_adaptive_regiontemporal_mod", str(module_path))
    spec = importlib.util.spec_from_loader("prior_adaptive_regiontemporal_mod", loader)
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


def load_or_build_items(args: argparse.Namespace):
    if args.items_cache_path.exists():
        cache_obj = torch.load(args.items_cache_path, map_location="cpu", weights_only=False)
        if isinstance(cache_obj, dict) and cache_obj.get("meta") == expected_cache_meta(args) and "items" in cache_obj:
            print(f"Loaded cache: {args.items_cache_path}")
            return cache_obj["items"]

    mod = load_spatiotemporal_builder()
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


def standardize_features(train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((train_x - mean) / std).astype(np.float32), ((val_x - mean) / std).astype(np.float32), ((test_x - mean) / std).astype(np.float32)


def _norm_ch_name(name: str) -> str:
    s = str(name).upper().strip()
    if s.startswith("EEG "):
        s = s[4:]
    for suffix in ["-LE", "-REF"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s.strip()


def _channel_index_map(ch_names: list[str]) -> dict[str, int]:
    return {_norm_ch_name(name): idx for idx, name in enumerate(ch_names)}


def _region_indices(ch_names: list[str]) -> dict[str, list[int]]:
    out = {k: [] for k in REGION_KEYS}
    for i, raw_name in enumerate(ch_names):
        ch = _norm_ch_name(raw_name)
        if ch.startswith(("FP", "F")):
            out["frontal"].append(i)
        if ch.startswith("C"):
            out["central"].append(i)
        if ch.startswith("T"):
            out["temporal"].append(i)
        if ch.startswith("P"):
            out["parietal"].append(i)
        if ch.startswith("O"):
            out["occipital"].append(i)
        if ch.endswith(("1", "3", "5", "7")):
            out["left"].append(i)
        elif ch.endswith(("2", "4", "6", "8")):
            out["right"].append(i)
        else:
            out["midline"].append(i)
        if ch.startswith(("FP", "F")) and ch.endswith(("1", "3", "5", "7")):
            out["frontal_left"].append(i)
        if ch.startswith(("FP", "F")) and ch.endswith(("2", "4", "6", "8")):
            out["frontal_right"].append(i)
        out["global"].append(i)
    return out


def compute_node_feature_stats(items: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    for item in items:
        for window in item["windows"]:
            for band_key in ("pcc", "theta", "alpha", "beta"):
                rows.append(window[band_key].x.detach().cpu().numpy().astype(np.float32))
    stacked = np.concatenate(rows, axis=0)
    mean = stacked.mean(axis=0, keepdims=True).astype(np.float32)
    std = stacked.std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def augment_region_graph(
    graph: Data,
    ch_names: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    topology_weight: float,
    adaptive_topk: int,
    adaptive_weight: float,
) -> Data:
    x_base = graph.x.detach().cpu().numpy().astype(np.float32)
    x_base = ((x_base - feature_mean) / feature_std).astype(np.float32)
    edge_index = graph.edge_index.detach().cpu().numpy().astype(np.int64)
    edge_weight = graph.edge_attr.detach().cpu().numpy().astype(np.float32)
    n_elec = int(x_base.shape[0])
    ch_norm = [_norm_ch_name(x) for x in ch_names[:n_elec]]
    ch_index = {name: idx for idx, name in enumerate(ch_norm)}
    regions = _region_indices(ch_names[:n_elec])

    region_x: list[np.ndarray] = []
    for region in REGION_KEYS:
        idx = regions[region]
        if idx:
            region_x.append(x_base[idx].mean(axis=0).astype(np.float32))
        else:
            region_x.append(np.zeros((x_base.shape[1],), dtype=np.float32))
    x_all = np.concatenate([x_base, np.stack(region_x, axis=0)], axis=0).astype(np.float32)
    n_total = int(x_all.shape[0])

    virtual_edges: list[tuple[int, int, float]] = []
    region_offset = n_elec
    global_node = region_offset + REGION_KEYS.index("global")

    for left_name, right_name in TOPOLOGY_PAIRS:
        if left_name in ch_index and right_name in ch_index:
            src = int(ch_index[left_name])
            dst = int(ch_index[right_name])
            virtual_edges.extend([(src, dst, float(topology_weight)), (dst, src, float(topology_weight))])

    if int(adaptive_topk) > 0 and float(adaptive_weight) > 0.0:
        x_norm = x_base / np.clip(np.linalg.norm(x_base, axis=1, keepdims=True), 1e-6, None)
        sim = x_norm @ x_norm.T
        np.fill_diagonal(sim, -np.inf)
        topk = min(int(adaptive_topk), max(n_elec - 1, 1))
        for src in range(n_elec):
            nbrs = np.argpartition(-sim[src], kth=topk - 1)[:topk]
            for dst in nbrs:
                weight = float(np.clip(sim[src, dst], 0.0, 1.0)) * float(adaptive_weight)
                if weight > 0.0:
                    virtual_edges.append((int(src), int(dst), weight))

    for region_pos, region in enumerate(REGION_KEYS):
        region_node = region_offset + region_pos
        idx = regions[region]
        if idx:
            scale = 1.0 / float(len(idx))
            for elec_idx in idx:
                virtual_edges.extend([(int(elec_idx), region_node, scale), (region_node, int(elec_idx), scale)])
        if region != "global":
            virtual_edges.extend([(region_node, global_node, 1.0), (global_node, region_node, 1.0)])

    left_node = region_offset + REGION_KEYS.index("left")
    right_node = region_offset + REGION_KEYS.index("right")
    fl_node = region_offset + REGION_KEYS.index("frontal_left")
    fr_node = region_offset + REGION_KEYS.index("frontal_right")
    virtual_edges.extend(
        [
            (left_node, right_node, 0.5),
            (right_node, left_node, 0.5),
            (fl_node, fr_node, 0.75),
            (fr_node, fl_node, 0.75),
        ]
    )

    if virtual_edges:
        vi = np.array([[src, dst] for src, dst, _ in virtual_edges], dtype=np.int64).T
        vw = np.array([weight for _, _, weight in virtual_edges], dtype=np.float32)
        edge_index = np.concatenate([edge_index, vi], axis=1)
        edge_weight = np.concatenate([edge_weight, vw], axis=0)

    region_membership = np.zeros((n_total, len(REGION_KEYS)), dtype=np.float32)
    for region_pos, region in enumerate(REGION_KEYS):
        region_node = region_offset + region_pos
        region_membership[region_node, region_pos] = 1.0
        for elec_idx in regions[region]:
            region_membership[int(elec_idx), region_pos] = 1.0

    node_type = np.zeros((n_total, 3), dtype=np.float32)
    node_type[:n_elec, 0] = 1.0
    node_type[n_elec:global_node, 1] = 1.0
    node_type[global_node, 2] = 1.0
    x_all = np.concatenate([x_all, region_membership, node_type], axis=1).astype(np.float32)

    return Data(
        x=torch.tensor(x_all, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_weight, dtype=torch.float32),
        y=graph.y.clone(),
    )


class StaticTemporalRegionDataset(Dataset):
    def __init__(
        self,
        items: list[dict[str, object]],
        static_x: np.ndarray,
        subject_ids: np.ndarray,
        ch_names: list[str],
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        topology_weight: float,
        adaptive_topk: int,
        adaptive_weight: float,
    ) -> None:
        self.items: list[dict[str, object]] = []
        for row_idx, item in enumerate(items):
            windows: list[dict[str, Data]] = []
            for window in item["windows"]:
                windows.append(
                    {
                        "pcc": augment_region_graph(
                            window["pcc"],
                            ch_names=ch_names,
                            feature_mean=feature_mean,
                            feature_std=feature_std,
                            topology_weight=topology_weight,
                            adaptive_topk=adaptive_topk,
                            adaptive_weight=adaptive_weight,
                        ),
                        "theta": augment_region_graph(
                            window["theta"],
                            ch_names=ch_names,
                            feature_mean=feature_mean,
                            feature_std=feature_std,
                            topology_weight=topology_weight,
                            adaptive_topk=adaptive_topk,
                            adaptive_weight=adaptive_weight,
                        ),
                        "alpha": augment_region_graph(
                            window["alpha"],
                            ch_names=ch_names,
                            feature_mean=feature_mean,
                            feature_std=feature_std,
                            topology_weight=topology_weight,
                            adaptive_topk=adaptive_topk,
                            adaptive_weight=adaptive_weight,
                        ),
                        "beta": augment_region_graph(
                            window["beta"],
                            ch_names=ch_names,
                            feature_mean=feature_mean,
                            feature_std=feature_std,
                            topology_weight=topology_weight,
                            adaptive_topk=adaptive_topk,
                            adaptive_weight=adaptive_weight,
                        ),
                    }
                )
            self.items.append(
                {
                    "subject_id": str(subject_ids[row_idx]),
                    "label": int(item["label"]),
                    "windows": windows,
                    "static_x": torch.tensor(static_x[row_idx], dtype=torch.float32),
                    "n_windows": int(item["n_windows"]),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.items[idx]


def collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "seq_windows": [x["windows"] for x in batch],
        "static_x": torch.stack([x["static_x"] for x in batch], dim=0),
        "y": torch.tensor([int(x["label"]) for x in batch], dtype=torch.long),
        "subject_ids": [str(x["subject_id"]) for x in batch],
        "n_windows": [int(x["n_windows"]) for x in batch],
    }


class StaticWeightedStarBranch(nn.Module):
    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.feature_logits = nn.Embedding(int(n_features), 2)
        self.bias = nn.Parameter(torch.zeros(2))
        nn.init.zeros_(self.feature_logits.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.feature_logits.weight + self.bias


class EdgeGATRegionEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_nodes: int, n_electrodes: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.n_electrodes = int(n_electrodes)
        self.hidden_dim = int(hidden_dim)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=1, dropout=dropout, add_self_loops=True)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=1, dropout=dropout, add_self_loops=True)
        self.norm0 = nn.BatchNorm1d(hidden_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Batch) -> torch.Tensor:
        edge_attr = data.edge_attr.view(-1, 1)
        x = self.norm0(self.input_proj(data.x))
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv1(x, data.edge_index, edge_attr=edge_attr)
        x = self.norm1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, data.edge_index, edge_attr=edge_attr)
        x = self.norm2(x)
        x = F.elu(x)

        batch_size = int(data.batch.max().item()) + 1 if data.batch.numel() else 0
        nodes = x.view(batch_size, self.n_nodes, self.hidden_dim)
        elec = nodes[:, : self.n_electrodes, :]
        region = nodes[:, self.n_electrodes :, :]

        reg_idx = {name: idx for idx, name in enumerate(REGION_KEYS)}
        frontal = region[:, reg_idx["frontal"], :]
        global_node = region[:, reg_idx["global"], :]
        left = region[:, reg_idx["left"], :]
        right = region[:, reg_idx["right"], :]
        frontal_left = region[:, reg_idx["frontal_left"], :]
        frontal_right = region[:, reg_idx["frontal_right"], :]
        return torch.cat(
            [
                elec.mean(dim=1),
                elec.max(dim=1).values,
                region.mean(dim=1),
                region.max(dim=1).values,
                frontal,
                global_node,
                left - right,
                frontal_left - frontal_right,
            ],
            dim=1,
        )


class TemporalWindowGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, max_windows: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.max_windows = int(max_windows)
        self.hidden_dim = int(hidden_dim)
        self.pos_embed = nn.Embedding(int(max_windows), int(in_dim))
        self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=heads, concat=False, edge_dim=1, dropout=dropout, add_self_loops=True)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=1, dropout=dropout, add_self_loops=True)
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _build_temporal_graph(length: int, x: torch.Tensor) -> Data:
        src: list[int] = []
        dst: list[int] = []
        weights: list[float] = []
        for idx in range(length - 1):
            src.extend([idx, idx + 1])
            dst.extend([idx + 1, idx])
            weights.extend([1.0, 1.0])
        for idx in range(length - 2):
            src.extend([idx, idx + 2])
            dst.extend([idx + 2, idx])
            weights.extend([0.5, 0.5])
        edge_index = torch.tensor([src, dst], dtype=torch.long, device=x.device) if src else torch.zeros((2, 0), dtype=torch.long, device=x.device)
        edge_attr = torch.tensor(weights, dtype=torch.float32, device=x.device) if weights else torch.zeros((0,), dtype=torch.float32, device=x.device)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def forward(self, seq_feats: list[torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        data_list: list[Data] = []
        lengths: list[int] = []
        for feat in seq_feats:
            length = int(feat.shape[0])
            lengths.append(length)
            pos_idx = torch.arange(length, device=device)
            x = feat + self.pos_embed(pos_idx)
            data_list.append(self._build_temporal_graph(length=length, x=x))
        batch = Batch.from_data_list(data_list).to(device)
        edge_attr = batch.edge_attr.view(-1, 1)
        x = self.conv1(batch.x, batch.edge_index, edge_attr=edge_attr)
        x = self.norm1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, batch.edge_index, edge_attr=edge_attr)
        x = self.norm2(x)
        x = F.elu(x)
        mean_pool = global_mean_pool(x, batch.batch)
        max_pool = global_max_pool(x, batch.batch)
        ptr = batch.ptr.detach().cpu().numpy().astype(np.int64)
        first_nodes = x[torch.tensor(ptr[:-1], dtype=torch.long, device=device)]
        last_nodes = x[torch.tensor(ptr[1:] - 1, dtype=torch.long, device=device)]
        return torch.cat([mean_pool, max_pool, first_nodes, last_nodes], dim=1)


class PriorAdaptiveRegionTemporalGNN(nn.Module):
    def __init__(
        self,
        n_static_features: int,
        in_dim: int,
        n_nodes: int,
        n_electrodes: int,
        hidden_dim: int,
        temporal_hidden: int,
        heads: int,
        dropout: float,
        max_windows: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.static_branch = StaticWeightedStarBranch(n_static_features)
        self.enc_pcc = EdgeGATRegionEncoder(in_dim, hidden_dim, n_nodes, n_electrodes, heads, dropout)
        self.enc_theta = EdgeGATRegionEncoder(in_dim, hidden_dim, n_nodes, n_electrodes, heads, dropout)
        self.enc_alpha = EdgeGATRegionEncoder(in_dim, hidden_dim, n_nodes, n_electrodes, heads, dropout)
        self.enc_beta = EdgeGATRegionEncoder(in_dim, hidden_dim, n_nodes, n_electrodes, heads, dropout)
        readout_dim = hidden_dim * 8
        self.band_proj = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.band_gate = nn.Sequential(
            nn.Linear(readout_dim * 4, hidden_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 4),
        )
        self.temporal_gnn = TemporalWindowGNN(
            in_dim=hidden_dim * 2,
            hidden_dim=temporal_hidden,
            max_windows=max_windows,
            heads=heads,
            dropout=dropout,
        )
        self.temporal_classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(temporal_hidden * 4, temporal_hidden * 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_hidden * 2, 2),
        )
        self.residual_scale = float(residual_scale)
        self.reset_temporal_head()

    def reset_temporal_head(self) -> None:
        for module in self.temporal_classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, batch: dict[str, object], use_temporal: bool = True) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        static_x = batch["static_x"]
        static_logits = self.static_branch(static_x)
        aux: dict[str, torch.Tensor] = {}
        if not use_temporal:
            aux["fusion_scale"] = torch.tensor(float(self.residual_scale), device=static_x.device)
            aux["band_weight_mean"] = torch.full((4,), 0.25, dtype=torch.float32, device=static_x.device)
            return static_logits, aux

        seq_windows = batch["seq_windows"]
        lengths: list[int] = []
        pcc_list: list[Data] = []
        theta_list: list[Data] = []
        alpha_list: list[Data] = []
        beta_list: list[Data] = []
        for windows in seq_windows:
            lengths.append(len(windows))
            for window in windows:
                pcc_list.append(window["pcc"])
                theta_list.append(window["theta"])
                alpha_list.append(window["alpha"])
                beta_list.append(window["beta"])

        device = static_x.device
        pcc_batch = Batch.from_data_list(pcc_list).to(device)
        theta_batch = Batch.from_data_list(theta_list).to(device)
        alpha_batch = Batch.from_data_list(alpha_list).to(device)
        beta_batch = Batch.from_data_list(beta_list).to(device)

        h_pcc = self.enc_pcc(pcc_batch)
        h_theta = self.enc_theta(theta_batch)
        h_alpha = self.enc_alpha(alpha_batch)
        h_beta = self.enc_beta(beta_batch)
        h_cat = torch.cat([h_pcc, h_theta, h_alpha, h_beta], dim=1)
        band_weight = F.softmax(self.band_gate(h_cat), dim=1)
        h_stack = torch.stack(
            [
                self.band_proj(h_pcc),
                self.band_proj(h_theta),
                self.band_proj(h_alpha),
                self.band_proj(h_beta),
            ],
            dim=1,
        )
        h_win = (band_weight.unsqueeze(-1) * h_stack).sum(dim=1)

        seq_feats: list[torch.Tensor] = []
        start = 0
        for length in lengths:
            seq_feats.append(h_win[start : start + length])
            start += length
        temporal_summary = self.temporal_gnn(seq_feats)
        temporal_logits = self.temporal_classifier(temporal_summary)

        aux["fusion_scale"] = torch.tensor(float(self.residual_scale), device=device)
        aux["band_weight_mean"] = band_weight.mean(dim=0)
        aux["temporal_logit_norm"] = temporal_logits.norm(dim=1).mean()
        return static_logits + (float(self.residual_scale) * temporal_logits), aux


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


def validation_score(metric: str, y_val: np.ndarray, prob_val: np.ndarray, low: float, high: float, step: float) -> float:
    metric = str(metric)
    if metric == "accuracy":
        return float(accuracy_score(y_val, (prob_val >= 0.5).astype(np.int64)))
    if metric == "threshold_accuracy":
        _, acc = select_threshold(y_val, prob_val, low=low, high=high, step=step)
        return float(acc)
    if metric == "auc":
        if len(np.unique(y_val)) < 2:
            return 0.5
        return float(roc_auc_score(y_val, prob_val))
    if metric == "neg_log_loss":
        p = np.clip(prob_val, 1e-6, 1.0 - 1e-6)
        return -float(log_loss(y_val, p, labels=[0, 1]))
    raise ValueError(f"Unsupported selection metric: {metric}")


def make_optimizer(model: PriorAdaptiveRegionTemporalGNN, args: argparse.Namespace, stage: str) -> torch.optim.Optimizer:
    if stage == "static":
        return torch.optim.AdamW(
            model.static_branch.parameters(),
            lr=float(args.lr_static),
            weight_decay=float(args.weight_decay_static),
        )
    temporal_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("static_branch."):
            continue
        temporal_params.append(param)
    return torch.optim.AdamW(
        temporal_params,
        lr=float(args.lr_temporal),
        weight_decay=float(args.weight_decay_temporal),
    )


def train_one_epoch(
    model: PriorAdaptiveRegionTemporalGNN,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_temporal: bool,
) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        batch["static_x"] = batch["static_x"].to(device)
        batch["y"] = batch["y"].to(device)
        optimizer.zero_grad()
        logits, _ = model(batch, use_temporal=use_temporal)
        loss = loss_fn(logits, batch["y"])
        loss.backward()
        optimizer.step()
        n = int(batch["y"].shape[0])
        total_loss += float(loss.item()) * n
        total_n += n
    return total_loss / max(total_n, 1)


@torch.no_grad()
def predict(
    model: PriorAdaptiveRegionTemporalGNN,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_temporal: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    band_all: list[np.ndarray] = []
    for batch in loader:
        batch["static_x"] = batch["static_x"].to(device)
        batch["y"] = batch["y"].to(device)
        logits, aux = model(batch, use_temporal=use_temporal)
        p = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        y_all.append(batch["y"].detach().cpu().numpy())
        p_all.append(p)
        band_all.append(aux["band_weight_mean"].detach().cpu().numpy())
    return np.concatenate(y_all), np.concatenate(p_all), np.stack(band_all)


def parse_optional_fold_ids(text: str) -> set[int] | None:
    values = [x.strip() for x in str(text).split(",") if x.strip()]
    if not values:
        return None
    return {int(x) for x in values}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repeated 5x10 CV for a prior-guided adaptive region-temporal pure GNN.")
    p.add_argument("--pcc-manifest", type=Path, default=Path("data/processed/graphs_pcc_topk_task/manifest.csv"))
    p.add_argument("--theta-manifest", type=Path, default=Path("data/processed/graphs_plv_theta_topk_task/manifest.csv"))
    p.add_argument("--alpha-manifest", type=Path, default=Path("data/processed/graphs_plv_alpha_topk_task/manifest.csv"))
    p.add_argument("--beta-manifest", type=Path, default=Path("data/processed/graphs_plv_beta_topk_task/manifest.csv"))
    p.add_argument("--split-csv", type=Path, default=Path("data/splits/task_split_subject_level.csv"))
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--static-warmup-epochs", type=int, default=320)
    p.add_argument("--residual-epochs", type=int, default=220)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr-static", type=float, default=0.05)
    p.add_argument("--weight-decay-static", type=float, default=0.1)
    p.add_argument("--lr-temporal", type=float, default=0.001)
    p.add_argument("--weight-decay-temporal", type=float, default=0.0001)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--temporal-hidden", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--residual-scale", type=float, default=0.15)
    p.add_argument("--topology-weight", type=float, default=0.35)
    p.add_argument("--adaptive-topk", type=int, default=2)
    p.add_argument("--adaptive-weight", type=float, default=0.25)
    p.add_argument("--max-seconds", type=int, default=120)
    p.add_argument("--window-seconds", type=float, default=16.0)
    p.add_argument("--step-seconds", type=float, default=8.0)
    p.add_argument("--max-windows", type=int, default=12)
    p.add_argument("--items-cache-path", type=Path, default=Path("outputs/cache/spatiotemporal_items/ms120_w16p0_s8p0_mw12_q0p8_me30_k4.pt"))
    p.add_argument("--pcc-quantile", type=float, default=0.8)
    p.add_argument("--min-edges", type=int, default=30)
    p.add_argument("--top-k-per-node", type=int, default=4)
    p.add_argument("--patience", type=int, default=140)
    p.add_argument("--selection-metric", type=str, default="threshold_accuracy", choices=["accuracy", "threshold_accuracy", "auc", "neg_log_loss"])
    p.add_argument("--thr-low", type=float, default=0.05)
    p.add_argument("--thr-high", type=float, default=0.95)
    p.add_argument("--thr-step", type=float, default=0.01)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--fold-ids", type=str, default="")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out-dir", type=Path, default=Path("outputs/metrics/runs/prior_adaptive_regiontemporal_gnn_5x10"))
    p.add_argument("--experiment-name", type=str, default="prior_adaptive_regiontemporal_gnn_5x10")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    fold_filter = parse_optional_fold_ids(args.fold_ids)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    static_meta, static_x_all = build_static_graph_feature_table(
        pcc_manifest=args.pcc_manifest,
        theta_manifest=args.theta_manifest,
        alpha_manifest=args.alpha_manifest,
        beta_manifest=args.beta_manifest,
    )
    items_raw = load_or_build_items(args)
    item_map = {str(item["subject_id"]): item for item in items_raw}
    items_aligned = [item_map[str(subject_id)] for subject_id in static_meta["subject_id"].tolist()]
    if any(int(a["label"]) != int(b) for a, b in zip(items_aligned, static_meta["label"].tolist())):
        raise ValueError("Static labels and temporal item labels do not match.")

    y_all = static_meta["label"].to_numpy(dtype=np.int64)
    subject_ids = static_meta["subject_id"].to_numpy()
    run_payloads: list[dict[str, object]] = []

    for seed in seeds:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=int(seed))
        fold_results: list[dict[str, object]] = []

        for fold_no, (trainval_idx, test_idx) in enumerate(skf.split(np.arange(len(y_all)), y_all), start=1):
            if fold_filter is not None and fold_no not in fold_filter:
                continue

            trainval_y = y_all[trainval_idx]
            tr_idx, val_idx = train_test_split(
                np.arange(len(trainval_idx)),
                test_size=float(args.val_size),
                random_state=int(seed) + fold_no,
                stratify=trainval_y,
            )
            train_idx = trainval_idx[tr_idx]
            val_abs_idx = trainval_idx[val_idx]

            x_train, x_val, x_test = standardize_features(static_x_all[train_idx], static_x_all[val_abs_idx], static_x_all[test_idx])
            train_items = [items_aligned[i] for i in train_idx]
            val_items = [items_aligned[i] for i in val_abs_idx]
            test_items = [items_aligned[i] for i in test_idx]
            node_mean, node_std = compute_node_feature_stats(train_items)

            train_ds = StaticTemporalRegionDataset(
                items=train_items,
                static_x=x_train,
                subject_ids=subject_ids[train_idx],
                ch_names=DEFAULT_CH_NAMES,
                feature_mean=node_mean,
                feature_std=node_std,
                topology_weight=args.topology_weight,
                adaptive_topk=args.adaptive_topk,
                adaptive_weight=args.adaptive_weight,
            )
            val_ds = StaticTemporalRegionDataset(
                items=val_items,
                static_x=x_val,
                subject_ids=subject_ids[val_abs_idx],
                ch_names=DEFAULT_CH_NAMES,
                feature_mean=node_mean,
                feature_std=node_std,
                topology_weight=args.topology_weight,
                adaptive_topk=args.adaptive_topk,
                adaptive_weight=args.adaptive_weight,
            )
            test_ds = StaticTemporalRegionDataset(
                items=test_items,
                static_x=x_test,
                subject_ids=subject_ids[test_idx],
                ch_names=DEFAULT_CH_NAMES,
                feature_mean=node_mean,
                feature_std=node_std,
                topology_weight=args.topology_weight,
                adaptive_topk=args.adaptive_topk,
                adaptive_weight=args.adaptive_weight,
            )

            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

            sample_graph = train_ds[0]["windows"][0]["pcc"]
            model = PriorAdaptiveRegionTemporalGNN(
                n_static_features=int(x_train.shape[1]),
                in_dim=int(sample_graph.x.shape[1]),
                n_nodes=int(sample_graph.x.shape[0]),
                n_electrodes=int(train_items[0]["windows"][0]["pcc"].x.shape[0]),
                hidden_dim=args.hidden_dim,
                temporal_hidden=args.temporal_hidden,
                heads=args.heads,
                dropout=args.dropout,
                max_windows=args.max_windows,
                residual_scale=args.residual_scale,
            ).to(device)

            best_state = None
            best_epoch = 1
            best_stage = "static"
            best_val_score = -1.0
            best_val_tiebreak = -float("inf")
            stale = 0

            optimizer = make_optimizer(model, args=args, stage="static")
            for epoch in range(1, args.static_warmup_epochs + 1):
                _ = train_one_epoch(model, train_loader, optimizer, device, use_temporal=False)
                y_val, p_val, _ = predict(model, val_loader, device, use_temporal=False)
                val_score = validation_score(args.selection_metric, y_val, p_val, args.thr_low, args.thr_high, args.thr_step)
                val_tiebreak = validation_score("neg_log_loss", y_val, p_val, args.thr_low, args.thr_high, args.thr_step)
                if (val_score > best_val_score) or (abs(val_score - best_val_score) <= 1e-12 and val_tiebreak > best_val_tiebreak):
                    best_val_score = val_score
                    best_val_tiebreak = val_tiebreak
                    best_epoch = int(epoch)
                    best_stage = "static"
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                if stale >= args.patience:
                    break

            if best_state is not None:
                model.load_state_dict(best_state)

            optimizer = make_optimizer(model, args=args, stage="temporal")
            stale = 0
            for epoch in range(1, args.residual_epochs + 1):
                _ = train_one_epoch(model, train_loader, optimizer, device, use_temporal=True)
                y_val, p_val, _ = predict(model, val_loader, device, use_temporal=True)
                val_score = validation_score(args.selection_metric, y_val, p_val, args.thr_low, args.thr_high, args.thr_step)
                val_tiebreak = validation_score("neg_log_loss", y_val, p_val, args.thr_low, args.thr_high, args.thr_step)
                if (val_score > best_val_score) or (abs(val_score - best_val_score) <= 1e-12 and val_tiebreak > best_val_tiebreak):
                    best_val_score = val_score
                    best_val_tiebreak = val_tiebreak
                    best_epoch = int(epoch)
                    best_stage = "residual"
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    stale = 0
                else:
                    stale += 1
                if stale >= args.patience:
                    break

            if best_state is not None:
                model.load_state_dict(best_state)
            use_temporal = bool(best_stage == "residual")
            y_val, p_val, band_val = predict(model, val_loader, device, use_temporal=use_temporal)
            thr_sel, val_thr_acc = select_threshold(y_val, p_val, args.thr_low, args.thr_high, args.thr_step)
            y_test, p_test, band_test = predict(model, test_loader, device, use_temporal=use_temporal)
            metrics = calc_metrics(y_test, p_test, threshold=thr_sel)
            band_mean = band_test.mean(axis=0)

            fold_results.append(
                {
                    "fold": fold_no,
                    "n_train": int(len(train_ds)),
                    "n_val": int(len(val_ds)),
                    "n_test": int(len(test_ds)),
                    "best_epoch": int(best_epoch),
                    "best_stage": best_stage,
                    "selection_metric": args.selection_metric,
                    "best_val_score": float(best_val_score),
                    "val_threshold_accuracy": float(val_thr_acc),
                    "test_threshold": float(thr_sel),
                    "residual_scale": float(args.residual_scale),
                    "band_weight_mean": {
                        "pcc": float(band_mean[0]),
                        "theta": float(band_mean[1]),
                        "alpha": float(band_mean[2]),
                        "beta": float(band_mean[3]),
                    },
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
                f"stage={best_stage} thr={thr_sel:.2f}"
            )

        if not fold_results:
            raise RuntimeError(f"No folds executed for seed={seed}; check --fold-ids.")
        payload = {
            "schema_version": "eeg_mdd_prior_adaptive_regiontemporal_gnn_5x10_v1",
            "experiment_name": args.experiment_name,
            "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model": "PriorAdaptiveRegionTemporalGNN",
            "device": str(device),
            "params": {
                "folds": int(args.folds),
                "seed": int(seed),
                "seeds": [int(x) for x in seeds],
                "static_warmup_epochs": int(args.static_warmup_epochs),
                "residual_epochs": int(args.residual_epochs),
                "batch_size": int(args.batch_size),
                "lr_static": float(args.lr_static),
                "weight_decay_static": float(args.weight_decay_static),
                "lr_temporal": float(args.lr_temporal),
                "weight_decay_temporal": float(args.weight_decay_temporal),
                "hidden_dim": int(args.hidden_dim),
                "temporal_hidden": int(args.temporal_hidden),
                "dropout": float(args.dropout),
                "heads": int(args.heads),
                "residual_scale": float(args.residual_scale),
                "topology_weight": float(args.topology_weight),
                "adaptive_topk": int(args.adaptive_topk),
                "adaptive_weight": float(args.adaptive_weight),
                "selection_metric": args.selection_metric,
                "thr_low": float(args.thr_low),
                "thr_high": float(args.thr_high),
                "thr_step": float(args.thr_step),
                "val_size": float(args.val_size),
                "fold_ids": sorted(fold_filter or []),
                "region_keys": REGION_KEYS,
            },
            "aggregate": aggregate_fold_results(fold_results),
            "fold_results": fold_results,
        }
        out_path = args.out_dir / f"prior_adaptive_regiontemporal_gnn_seed{seed}_cv10.json"
        write_json(out_path, payload)
        run_payloads.append({**payload, "_out_path": str(out_path)})
        print(f"saved: {out_path}")

    summary = summarize_run_payloads(run_payloads)
    summary_payload = {
        "schema_version": "eeg_mdd_prior_adaptive_regiontemporal_gnn_5x10_summary_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "PriorAdaptiveRegionTemporalGNN",
        "params": {
            "folds": int(args.folds),
            "seeds": [int(x) for x in seeds],
            "static_warmup_epochs": int(args.static_warmup_epochs),
            "residual_epochs": int(args.residual_epochs),
            "batch_size": int(args.batch_size),
            "lr_static": float(args.lr_static),
            "weight_decay_static": float(args.weight_decay_static),
            "lr_temporal": float(args.lr_temporal),
            "weight_decay_temporal": float(args.weight_decay_temporal),
            "hidden_dim": int(args.hidden_dim),
            "temporal_hidden": int(args.temporal_hidden),
            "dropout": float(args.dropout),
            "heads": int(args.heads),
            "residual_scale": float(args.residual_scale),
            "topology_weight": float(args.topology_weight),
            "adaptive_topk": int(args.adaptive_topk),
            "adaptive_weight": float(args.adaptive_weight),
            "selection_metric": args.selection_metric,
            "thr_low": float(args.thr_low),
            "thr_high": float(args.thr_high),
            "thr_step": float(args.thr_step),
            "val_size": float(args.val_size),
            "fold_ids": sorted(fold_filter or []),
            "region_keys": REGION_KEYS,
        },
        **summary,
    }
    summary_path = args.out_dir / "summary_5x10.json"
    write_json(summary_path, summary_payload)
    print(f"\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
