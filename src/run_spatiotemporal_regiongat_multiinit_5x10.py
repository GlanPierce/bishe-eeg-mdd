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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv

from feature_model_utils import aggregate_fold_results, parse_seed_list, summarize_run_payloads, write_json


REGION_KEYS = ["frontal", "central", "temporal", "parietal", "occipital", "left", "right", "midline", "global"]
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


def load_base_module():
    module_path = Path("src/__pycache__/train_gnn_dualgraph_multiband_spatiotemporal_cv10.cpython-311.pyc")
    src_dir = str(Path("src").resolve())
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    loader = importlib.machinery.SourcelessFileLoader("spatiotemporal_regiongat_mod", str(module_path))
    spec = importlib.util.spec_from_loader("spatiotemporal_regiongat_mod", loader)
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


def _norm_ch_name(name: str) -> str:
    s = str(name).upper().strip()
    if s.startswith("EEG "):
        s = s[4:]
    for suffix in ["-LE", "-REF"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s.strip()


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
        out["global"].append(i)
    return out


def augment_region_graph(graph: Data, ch_names: list[str]) -> Data:
    x_base = graph.x.detach().cpu().numpy().astype(np.float32)
    edge_index = graph.edge_index.detach().cpu().numpy().astype(np.int64)
    edge_weight = graph.edge_attr.detach().cpu().numpy().astype(np.float32)
    n_elec = int(x_base.shape[0])
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
    global_node = n_elec + REGION_KEYS.index("global")
    for region_pos, region in enumerate(REGION_KEYS):
        region_node = n_elec + region_pos
        for elec_idx in regions[region]:
            virtual_edges.append((elec_idx, region_node, 1.0))
            virtual_edges.append((region_node, elec_idx, 1.0))
        if region != "global":
            virtual_edges.append((region_node, global_node, 1.0))
            virtual_edges.append((global_node, region_node, 1.0))

    # A direct left-right bridge exposes the frontal/hemisphere asymmetry prior without leaving GNN space.
    left_node = n_elec + REGION_KEYS.index("left")
    right_node = n_elec + REGION_KEYS.index("right")
    virtual_edges.extend([(left_node, right_node, 0.5), (right_node, left_node, 0.5)])

    if virtual_edges:
        vi = np.array([[src, dst] for src, dst, _ in virtual_edges], dtype=np.int64).T
        vw = np.array([weight for _, _, weight in virtual_edges], dtype=np.float32)
        edge_index = np.concatenate([edge_index, vi], axis=1)
        edge_weight = np.concatenate([edge_weight, vw], axis=0)

    region_membership = np.zeros((n_total, len(REGION_KEYS)), dtype=np.float32)
    for region_pos, region in enumerate(REGION_KEYS):
        region_node = n_elec + region_pos
        region_membership[region_node, region_pos] = 1.0
        for elec_idx in regions[region]:
            region_membership[elec_idx, region_pos] = 1.0

    node_type = np.zeros((n_total, 2), dtype=np.float32)
    node_type[:n_elec, 0] = 1.0
    node_type[n_elec:, 1] = 1.0
    x_all = np.concatenate([x_all, region_membership, node_type], axis=1).astype(np.float32)

    return Data(
        x=torch.tensor(x_all, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_weight, dtype=torch.float32),
        y=graph.y.clone(),
    )


class RegionTemporalDataset(Dataset):
    def __init__(self, items: list[dict[str, object]], ch_names: list[str]) -> None:
        self.items: list[dict[str, object]] = []
        for item in items:
            windows: list[dict[str, Data]] = []
            for window in item["windows"]:
                windows.append(
                    {
                        "pcc": augment_region_graph(window["pcc"], ch_names=ch_names),
                        "theta": augment_region_graph(window["theta"], ch_names=ch_names),
                        "alpha": augment_region_graph(window["alpha"], ch_names=ch_names),
                        "beta": augment_region_graph(window["beta"], ch_names=ch_names),
                    }
                )
            self.items.append(
                {
                    "subject_id": str(item["subject_id"]),
                    "label": int(item["label"]),
                    "windows": windows,
                    "n_windows": int(item["n_windows"]),
                    "file_path": str(item["file_path"]),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.items[idx]


def collate_fn(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "seq_windows": [x["windows"] for x in batch],
        "y": torch.tensor([int(x["label"]) for x in batch], dtype=torch.long),
        "subject_ids": [str(x["subject_id"]) for x in batch],
    }


class EdgeGATRegionEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, n_nodes: int, n_electrodes: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.n_electrodes = int(n_electrodes)
        self.hidden_dim = int(hidden_dim)
        self.conv1 = GATv2Conv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,
            edge_dim=1,
            dropout=dropout,
            add_self_loops=True,
        )
        self.conv2 = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=heads,
            concat=False,
            edge_dim=1,
            dropout=dropout,
            add_self_loops=True,
        )
        self.norm1 = nn.BatchNorm1d(hidden_dim)
        self.norm2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Batch) -> torch.Tensor:
        edge_attr = data.edge_attr.view(-1, 1)
        x = self.conv1(data.x, data.edge_index, edge_attr=edge_attr)
        x = self.norm1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.conv2(x, data.edge_index, edge_attr=edge_attr)
        x = self.norm2(x)
        x = F.elu(x)

        batch_size = int(data.batch.max().item()) + 1 if data.batch.numel() > 0 else 0
        nodes = x.view(batch_size, self.n_nodes, self.hidden_dim)
        virtual = nodes[:, self.n_electrodes :, :]
        frontal = nodes[:, self.n_electrodes + REGION_KEYS.index("frontal"), :]
        left = nodes[:, self.n_electrodes + REGION_KEYS.index("left"), :]
        right = nodes[:, self.n_electrodes + REGION_KEYS.index("right"), :]
        global_node = nodes[:, self.n_electrodes + REGION_KEYS.index("global"), :]
        return torch.cat(
            [
                nodes.mean(dim=1),
                nodes.max(dim=1).values,
                virtual.mean(dim=1),
                frontal,
                left - right,
                global_node,
            ],
            dim=1,
        )


class RegionGATSpatioTemporalModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        n_nodes: int,
        n_electrodes: int,
        hidden_dim: int = 32,
        temporal_hidden: int = 32,
        heads: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.enc_pcc = EdgeGATRegionEncoder(in_dim, hidden_dim, n_nodes, n_electrodes, heads, dropout)
        self.enc_theta = EdgeGATRegionEncoder(in_dim, hidden_dim, n_nodes, n_electrodes, heads, dropout)
        self.enc_alpha = EdgeGATRegionEncoder(in_dim, hidden_dim, n_nodes, n_electrodes, heads, dropout)
        self.enc_beta = EdgeGATRegionEncoder(in_dim, hidden_dim, n_nodes, n_electrodes, heads, dropout)
        readout_dim = hidden_dim * 6
        self.band_proj = nn.Sequential(
            nn.Linear(readout_dim, hidden_dim * 2),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.band_gate = nn.Sequential(
            nn.Linear(readout_dim * 4, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        self.temporal = nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=temporal_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.temporal_attn = nn.Linear(temporal_hidden * 2, 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(temporal_hidden * 2, 2),
        )

    def forward(self, seq_windows: list[list[dict[str, Data]]]) -> tuple[torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
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

        padded = pad_sequence(seq_feats, batch_first=True)
        packed = pack_padded_sequence(padded, lengths=lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.temporal(packed)
        temporal_out, _ = pad_packed_sequence(packed_out, batch_first=True)
        max_len = int(temporal_out.shape[1])
        mask = torch.arange(max_len, device=device).unsqueeze(0) < torch.tensor(lengths, device=device).unsqueeze(1)
        attn_logits = self.temporal_attn(temporal_out).squeeze(-1).masked_fill(~mask, -1e9)
        attn = F.softmax(attn_logits, dim=1)
        summary = torch.sum(attn.unsqueeze(-1) * temporal_out, dim=1)
        return self.classifier(summary), band_weight


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repeated CV for report-inspired region-GAT pure spatiotemporal GNN.")
    p.add_argument("--split-csv", type=Path, default=Path("data/splits/task_split_subject_level.csv"))
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--temporal-hidden", type=int, default=32)
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--max-seconds", type=int, default=120)
    p.add_argument("--window-seconds", type=float, default=16.0)
    p.add_argument("--step-seconds", type=float, default=8.0)
    p.add_argument("--max-windows", type=int, default=12)
    p.add_argument("--items-cache-path", type=Path, default=Path("outputs/cache/spatiotemporal_items/ms120_w16p0_s8p0_mw12_q0p8_me30_k4.pt"))
    p.add_argument("--pcc-quantile", type=float, default=0.8)
    p.add_argument("--min-edges", type=int, default=30)
    p.add_argument("--top-k-per-node", type=int, default=4)
    p.add_argument("--threshold-mode", type=str, default="fixed", choices=["fixed", "val_opt"])
    p.add_argument("--threshold-default", type=float, default=0.5)
    p.add_argument("--threshold-metric", type=str, default="balanced_accuracy", choices=["balanced_accuracy", "f1", "accuracy"])
    p.add_argument("--threshold-grid-step", type=float, default=0.01)
    p.add_argument("--threshold-grid-low", type=float, default=0.05)
    p.add_argument("--threshold-grid-high", type=float, default=0.95)
    p.add_argument("--calibration", type=str, default="none", choices=["none", "temperature", "platt"])
    p.add_argument("--loss-class-weight", type=str, default="none", choices=["none", "balanced"])
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--init-seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--select-top-k", type=int, default=3)
    p.add_argument("--selection-metric", type=str, default="balanced_accuracy", choices=["balanced_accuracy", "accuracy", "f1"])
    p.add_argument("--fold-ids", type=str, default="", help="Optional comma-separated outer fold ids to run, e.g. 5,7.")
    p.add_argument("--ensemble-space", type=str, default="score", choices=["score", "prob"])
    p.add_argument("--ensemble-reduction", type=str, default="mean", choices=["mean", "median"])
    p.add_argument("--ensemble-weighting", type=str, default="uniform", choices=["uniform", "val_metric", "val_metric_softmax"])
    p.add_argument("--weighting-scale", type=float, default=20.0)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/metrics/runs/spatiotemporal_regiongat_multiinit_5x10"))
    p.add_argument("--experiment-name", type=str, default="spatiotemporal_regiongat_multiinit_5x10")
    return p.parse_args()


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


def load_or_build_items(mod, args: argparse.Namespace):
    if args.items_cache_path.exists():
        cache_obj = torch.load(args.items_cache_path, map_location="cpu", weights_only=False)
        if isinstance(cache_obj, dict) and cache_obj.get("meta") == expected_cache_meta(args) and "items" in cache_obj:
            print(f"Loaded cache: {args.items_cache_path}")
            return cache_obj["items"]

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


def parse_optional_fold_ids(text: str) -> set[int] | None:
    values = [x.strip() for x in str(text).split(",") if x.strip()]
    if not values:
        return None
    return {int(x) for x in values}


def serialize_args(args: argparse.Namespace, seed: int, seeds: list[int]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    out["seed"] = int(seed)
    out["seeds"] = [int(x) for x in seeds]
    out["region_keys"] = REGION_KEYS
    return out


def candidate_weights(selected: list[dict[str, object]], mode: str, scale: float) -> np.ndarray:
    if len(selected) == 1 or str(mode) == "uniform":
        return np.full(len(selected), 1.0 / max(len(selected), 1), dtype=np.float64)
    metrics = np.array([float(x["best_metric"]) for x in selected], dtype=np.float64)
    if str(mode) == "val_metric":
        metrics = np.clip(metrics, 1e-8, None)
        return metrics / metrics.sum()
    if str(mode) == "val_metric_softmax":
        logits = (metrics - float(np.max(metrics))) * float(scale)
        w = np.exp(logits)
        return w / np.sum(w)
    raise ValueError(f"Unsupported ensemble weighting mode: {mode}")


def reduce_candidate_arrays(arrays: list[np.ndarray], reduction: str, weights: np.ndarray | None) -> np.ndarray:
    stack = np.stack(arrays, axis=0)
    if str(reduction) == "mean":
        if weights is None:
            return np.mean(stack, axis=0)
        return np.average(stack, axis=0, weights=weights)
    if str(reduction) == "median":
        return np.median(stack, axis=0)
    raise ValueError(f"Unsupported ensemble reduction: {reduction}")


def make_loss_weight(labels: np.ndarray, mode: str, device: torch.device) -> torch.Tensor | None:
    if str(mode) == "none":
        return None
    counts = np.bincount(labels.astype(np.int64), minlength=2).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    weights = counts.sum() / (2.0 * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_weight: torch.Tensor | None,
) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss(weight=loss_weight, label_smoothing=float(getattr(model, "_label_smoothing", 0.0)))
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        y = batch["y"].to(device)
        optimizer.zero_grad()
        logits, _ = model(batch["seq_windows"])
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        n = int(y.shape[0])
        total_loss += float(loss.item()) * n
        total_n += n
    return total_loss / max(total_n, 1)


def run_one_seed(mod, args: argparse.Namespace, items: list[dict[str, object]], seed: int, device: torch.device) -> dict[str, object]:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    init_seeds = parse_seed_list(args.init_seeds)
    fold_filter = parse_optional_fold_ids(args.fold_ids)

    y_all = np.array([int(x["label"]) for x in items], dtype=np.int64)
    idx_all = np.arange(len(items))
    sample_graph = augment_region_graph(items[0]["windows"][0]["pcc"], ch_names=DEFAULT_CH_NAMES)
    in_dim = int(sample_graph.x.shape[1])
    n_nodes = int(sample_graph.x.shape[0])
    n_electrodes = int(items[0]["windows"][0]["pcc"].x.shape[0])
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=int(seed))
    fold_results: list[dict[str, object]] = []

    for fold_no, (trainval_idx, test_idx) in enumerate(skf.split(idx_all, y_all), start=1):
        if fold_filter is not None and fold_no not in fold_filter:
            continue
        trainval_items = [items[i] for i in trainval_idx]
        test_items = [items[i] for i in test_idx]
        trainval_y = np.array([int(x["label"]) for x in trainval_items], dtype=np.int64)
        tr_idx, val_idx = train_test_split(
            np.arange(len(trainval_items)),
            test_size=0.2,
            random_state=int(seed) + fold_no,
            stratify=trainval_y,
        )
        train_items = [trainval_items[i] for i in tr_idx]
        val_items = [trainval_items[i] for i in val_idx]

        train_ds = RegionTemporalDataset(train_items, ch_names=DEFAULT_CH_NAMES)
        val_ds = RegionTemporalDataset(val_items, ch_names=DEFAULT_CH_NAMES)
        test_ds = RegionTemporalDataset(test_items, ch_names=DEFAULT_CH_NAMES)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        train_y = np.array([int(x["label"]) for x in train_items], dtype=np.int64)
        loss_weight = make_loss_weight(train_y, mode=args.loss_class_weight, device=device)

        candidates: list[dict[str, object]] = []
        for init_seed in init_seeds:
            train_seed = (int(init_seed) * 1000) + fold_no
            np.random.seed(train_seed)
            torch.manual_seed(train_seed)
            model = RegionGATSpatioTemporalModel(
                in_dim=in_dim,
                n_nodes=n_nodes,
                n_electrodes=n_electrodes,
                hidden_dim=args.hidden_dim,
                temporal_hidden=args.temporal_hidden,
                heads=args.heads,
                dropout=args.dropout,
            ).to(device)
            model._label_smoothing = float(args.label_smoothing)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            best_state = None
            best_epoch = 1
            best_metric = -1.0
            for epoch in range(1, args.epochs + 1):
                _ = train_one_epoch(model, train_loader, optimizer, device, loss_weight)
                y_val, p_val, _, _ = mod.predict(model, val_loader, device)
                val_metric = metric_value(args.selection_metric, y_true=y_val, prob_pos=p_val)
                if val_metric > best_metric:
                    best_metric = val_metric
                    best_epoch = epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if best_state is None:
                continue
            model.load_state_dict(best_state)
            y_val_best, p_val_best, _, s_val_best = mod.predict(model, val_loader, device)
            y_test, p_test, w_test, s_test = mod.predict(model, test_loader, device)
            candidates.append(
                {
                    "init_seed": int(init_seed),
                    "train_seed": int(train_seed),
                    "best_epoch": int(best_epoch),
                    "best_metric": float(best_metric),
                    "y_val": y_val_best,
                    "p_val": p_val_best,
                    "s_val": s_val_best,
                    "y_test": y_test,
                    "p_test": p_test,
                    "s_test": s_test,
                    "w_test": w_test,
                }
            )

        if not candidates:
            raise RuntimeError(f"No candidates trained for seed={seed} fold={fold_no}")

        candidates.sort(key=lambda x: (float(x["best_metric"]), -abs(int(x["best_epoch"]) - (args.epochs // 2))), reverse=True)
        selected = candidates[: max(1, min(int(args.select_top_k), len(candidates)))]
        ens_weights = candidate_weights(selected, mode=args.ensemble_weighting, scale=args.weighting_scale)
        y_val_ref = selected[0]["y_val"]
        y_test_ref = selected[0]["y_test"]
        ens_w_test = reduce_candidate_arrays([x["w_test"] for x in selected], reduction=args.ensemble_reduction, weights=ens_weights)

        if args.ensemble_space == "score":
            ens_score_val = reduce_candidate_arrays([x["s_val"] for x in selected], reduction=args.ensemble_reduction, weights=ens_weights)
            ens_score_test = reduce_candidate_arrays([x["s_test"] for x in selected], reduction=args.ensemble_reduction, weights=ens_weights)
            p_val, p_test, calib_info = mod.calibrate_with_val(
                method=args.calibration,
                y_val=y_val_ref,
                score_val=ens_score_val,
                score_test=ens_score_test,
                seed=int(seed) + fold_no,
            )
        else:
            p_val = reduce_candidate_arrays([x["p_val"] for x in selected], reduction=args.ensemble_reduction, weights=ens_weights)
            p_test = reduce_candidate_arrays([x["p_test"] for x in selected], reduction=args.ensemble_reduction, weights=ens_weights)
            calib_info = {"method": "none_prob_ensemble"}

        if args.threshold_mode == "val_opt":
            test_thr, val_thr_score = mod.select_threshold_from_val(
                y_true=y_val_ref,
                prob_pos=p_val,
                metric=args.threshold_metric,
                step=args.threshold_grid_step,
                low=args.threshold_grid_low,
                high=args.threshold_grid_high,
            )
        else:
            test_thr = float(args.threshold_default)
            val_thr_score = float("nan")

        metrics = mod.calc_metrics(y_test_ref, p_test, threshold=test_thr)
        band_mean = ens_w_test.mean(axis=0)
        fold_results.append(
            {
                "fold": fold_no,
                "n_train": int(len(train_ds)),
                "n_val": int(len(val_ds)),
                "n_test": int(len(test_ds)),
                "selected_init_seeds": [int(x["init_seed"]) for x in selected],
                "selected_best_epochs": [int(x["best_epoch"]) for x in selected],
                "selected_best_metrics": [float(x["best_metric"]) for x in selected],
                "selected_ensemble_weights": [float(x) for x in ens_weights.tolist()],
                "test_threshold": float(test_thr),
                "calibration": calib_info,
                "val_threshold_metric": args.threshold_metric if args.threshold_mode == "val_opt" else None,
                "val_threshold_score": None if args.threshold_mode != "val_opt" else float(val_thr_score),
                "weights_mean": {
                    "pcc": float(band_mean[0]),
                    "theta": float(band_mean[1]),
                    "alpha": float(band_mean[2]),
                    "beta": float(band_mean[3]),
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
            f"thr={test_thr:.2f} "
            f"seeds={[int(x['init_seed']) for x in selected]}"
        )

    if not fold_results:
        raise RuntimeError(f"No folds were executed for seed={seed}; check --fold-ids.")

    return {
        "schema_version": "eeg_mdd_spatiotemporal_regiongat_multiinit_5x10_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "RegionGATSpatioTemporalModelMultiInitEnsemble",
        "device": str(device),
        "params": serialize_args(args, seed=seed, seeds=parse_seed_list(args.seeds)),
        "aggregate": aggregate_fold_results(fold_results),
        "fold_results": fold_results,
    }


def main() -> int:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod = load_base_module()
    items = load_or_build_items(mod, args)

    run_payloads: list[dict[str, object]] = []
    for seed in seeds:
        payload = run_one_seed(mod=mod, args=args, items=items, seed=int(seed), device=device)
        out_path = args.out_dir / f"spatiotemporal_regiongat_multiinit_seed{seed}_cv10.json"
        write_json(out_path, payload)
        run_payloads.append({**payload, "_out_path": str(out_path)})
        print(f"saved: {out_path}")

    summary = summarize_run_payloads(run_payloads)
    summary_payload = {
        "schema_version": "eeg_mdd_spatiotemporal_regiongat_multiinit_5x10_summary_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "RegionGATSpatioTemporalModelMultiInitEnsemble",
        "params": {
            "folds": int(args.folds),
            "seeds": [int(x) for x in seeds],
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "temporal_hidden": int(args.temporal_hidden),
            "heads": int(args.heads),
            "max_seconds": int(args.max_seconds),
            "window_seconds": float(args.window_seconds),
            "step_seconds": float(args.step_seconds),
            "max_windows": int(args.max_windows),
            "pcc_quantile": float(args.pcc_quantile),
            "min_edges": int(args.min_edges),
            "top_k_per_node": int(args.top_k_per_node),
            "threshold_mode": args.threshold_mode,
            "threshold_default": float(args.threshold_default),
            "threshold_metric": args.threshold_metric,
            "threshold_grid_step": float(args.threshold_grid_step),
            "threshold_grid_low": float(args.threshold_grid_low),
            "threshold_grid_high": float(args.threshold_grid_high),
            "calibration": args.calibration,
            "loss_class_weight": args.loss_class_weight,
            "label_smoothing": float(args.label_smoothing),
            "init_seeds": [int(x) for x in parse_seed_list(args.init_seeds)],
            "select_top_k": int(args.select_top_k),
            "selection_metric": args.selection_metric,
            "fold_ids": sorted(parse_optional_fold_ids(args.fold_ids) or []),
            "ensemble_space": args.ensemble_space,
            "ensemble_reduction": args.ensemble_reduction,
            "ensemble_weighting": args.ensemble_weighting,
            "weighting_scale": float(args.weighting_scale),
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
