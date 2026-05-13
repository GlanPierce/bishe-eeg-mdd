from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from feature_model_utils import (
    aggregate_fold_results,
    build_static_graph_feature_table,
    calc_metrics,
    parse_seed_list,
    summarize_run_payloads,
    write_json,
)


REGION_KEYS = ["frontal", "central", "temporal", "parietal", "occipital", "left", "right", "midline", "cross_region", "global"]
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


def load_spatiotemporal_builder():
    src_dir = str(Path("src").resolve())
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    module_path = Path("src/__pycache__/train_gnn_dualgraph_multiband_spatiotemporal_cv10.cpython-311.pyc")
    loader = importlib.machinery.SourcelessFileLoader("region_temporal_builder", str(module_path))
    spec = importlib.util.spec_from_loader("region_temporal_builder", loader)
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


def load_or_build_temporal_items(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.items_cache_path.exists():
        cache_obj = torch.load(args.items_cache_path, map_location="cpu", weights_only=False)
        if isinstance(cache_obj, dict) and cache_obj.get("meta") == expected_cache_meta(args) and "items" in cache_obj:
            print(f"Loaded temporal cache: {args.items_cache_path}")
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


def _adj_from_graph(graph) -> np.ndarray:
    x = graph.x.detach().cpu().numpy().astype(np.float32)
    edge_index = graph.edge_index.detach().cpu().numpy().astype(np.int64)
    edge_weight = graph.edge_attr.detach().cpu().numpy().astype(np.float32)
    adj = np.zeros((x.shape[0], x.shape[0]), dtype=np.float32)
    for (src, dst), weight in zip(edge_index.T, edge_weight):
        adj[int(src), int(dst)] = float(weight)
    return adj


def window_to_feature_vector(window: dict[str, object]) -> np.ndarray:
    pcc_x = window["pcc"].x.detach().cpu().numpy().astype(np.float32)
    tri = np.triu_indices(pcc_x.shape[0], k=1)
    parts = [
        pcc_x.reshape(-1),
        _adj_from_graph(window["pcc"])[tri],
        _adj_from_graph(window["theta"])[tri],
        _adj_from_graph(window["alpha"])[tri],
        _adj_from_graph(window["beta"])[tri],
    ]
    return np.concatenate(parts).astype(np.float32)


def build_temporal_graph_feature_table(items: list[dict[str, object]]) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, object]] = []
    features: list[np.ndarray] = []
    for item in sorted(items, key=lambda x: str(x["subject_id"])):
        window_features = np.stack([window_to_feature_vector(w) for w in item["windows"]]).astype(np.float32)
        temporal_summary = np.concatenate(
            [
                window_features.mean(axis=0),
                window_features.std(axis=0),
                window_features[-1] - window_features[0],
            ]
        ).astype(np.float32)
        rows.append({"subject_id": str(item["subject_id"]), "label": int(item["label"]), "n_windows": int(item["n_windows"])})
        features.append(temporal_summary)
    return pd.DataFrame(rows), np.stack(features)


def build_temporal_node_feature_table(items: list[dict[str, object]]) -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[dict[str, object]] = []
    features: list[np.ndarray] = []
    for item in sorted(items, key=lambda x: str(x["subject_id"])):
        window_features = np.stack(
            [w["pcc"].x.detach().cpu().numpy().astype(np.float32).reshape(-1) for w in item["windows"]]
        ).astype(np.float32)
        temporal_summary = np.concatenate(
            [
                window_features.mean(axis=0),
                window_features.std(axis=0),
                window_features[-1] - window_features[0],
            ]
        ).astype(np.float32)
        rows.append({"subject_id": str(item["subject_id"]), "label": int(item["label"]), "n_windows": int(item["n_windows"])})
        features.append(temporal_summary)
    return pd.DataFrame(rows), np.stack(features)


def _norm_ch_name(name: str) -> str:
    s = str(name).upper().strip()
    if s.startswith("EEG "):
        s = s[4:]
    for suffix in ["-LE", "-REF"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s.strip()


def channel_regions(name: str) -> list[int]:
    ch = _norm_ch_name(name)
    regions: list[str] = []
    if ch.startswith(("FP", "F")):
        regions.append("frontal")
    elif ch.startswith("C"):
        regions.append("central")
    elif ch.startswith("T"):
        regions.append("temporal")
    elif ch.startswith("P"):
        regions.append("parietal")
    elif ch.startswith("O"):
        regions.append("occipital")
    if ch.endswith(("1", "3", "5", "7")):
        regions.append("left")
    elif ch.endswith(("2", "4", "6", "8")):
        regions.append("right")
    else:
        regions.append("midline")
    regions.append("global")
    return [REGION_KEYS.index(r) for r in dict.fromkeys(regions)]


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
            primary_i = {REGION_KEYS[r] for r in channel_map[int(i)] if REGION_KEYS[r] in {"frontal", "central", "temporal", "parietal", "occipital"}}
            primary_j = {REGION_KEYS[r] for r in channel_map[int(j)] if REGION_KEYS[r] in {"frontal", "central", "temporal", "parietal", "occipital"}}
            if primary_i != primary_j:
                regs.append(REGION_KEYS.index("cross_region"))
            feature_regions.append(sorted(set(regs)))
    return feature_regions


def node_feature_region_map(ch_names: list[str], n_bands: int = 5) -> list[list[int]]:
    feature_regions: list[list[int]] = []
    channel_map = [channel_regions(ch) for ch in ch_names]
    for channel_idx in range(len(ch_names)):
        for _ in range(n_bands):
            feature_regions.append(channel_map[channel_idx])
    return feature_regions


def combined_feature_region_map(base_map: list[list[int]], temporal_map: list[list[int]] | None) -> list[list[int]]:
    out = list(base_map)
    if temporal_map is not None:
        out.extend(temporal_map)
    return out


def build_region_graph_edges(feature_regions: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_features = len(feature_regions)
    n_regions = len(REGION_KEYS)
    region_offset = n_features
    global_node = n_features + n_regions
    src: list[int] = []
    dst: list[int] = []
    weights: list[float] = []
    membership = np.zeros((n_regions, n_features), dtype=np.float32)
    for feature_idx, regions in enumerate(feature_regions):
        unique_regions = sorted(set(int(r) for r in regions))
        if not unique_regions:
            unique_regions = [REGION_KEYS.index("global")]
        scale = 1.0 / float(len(unique_regions))
        for region_idx in unique_regions:
            region_node = region_offset + region_idx
            src.extend([feature_idx, region_node])
            dst.extend([region_node, feature_idx])
            weights.extend([scale, scale])
            membership[region_idx, feature_idx] = scale
    for region_idx in range(n_regions):
        region_node = region_offset + region_idx
        src.extend([region_node, global_node])
        dst.extend([global_node, region_node])
        weights.extend([1.0, 1.0])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight, torch.tensor(membership, dtype=torch.float32)


class RegionTemporalFeatureNodeDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        subject_ids: np.ndarray,
        feature_regions: list[list[int]],
        region_scale: float = 1.0,
    ) -> None:
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.subject_ids = np.asarray(subject_ids)
        self.n_features = int(self.x.shape[1])
        self.n_regions = len(REGION_KEYS)
        self.region_scale = float(region_scale)
        self.edge_index, self.edge_weight, self.region_membership = build_region_graph_edges(feature_regions)
        self.node_id = torch.arange(self.n_features + self.n_regions + 1, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Data:
        feature_values = torch.tensor(self.x[idx], dtype=torch.float32)
        region_den = torch.clamp(self.region_membership.sum(dim=1), min=1.0)
        region_values = ((self.region_membership @ feature_values) / region_den) * self.region_scale
        values = torch.cat([feature_values, region_values, torch.zeros(1, dtype=torch.float32)]).view(-1, 1)
        return Data(
            x=values,
            node_id=self.node_id.clone(),
            edge_index=self.edge_index.clone(),
            edge_attr=self.edge_weight.clone(),
            y=torch.tensor([int(self.y[idx])], dtype=torch.long),
            subject_id=str(self.subject_ids[idx]),
        )


class RegionTemporalFeatureNodeGNN(nn.Module):
    def __init__(self, n_features: int, n_regions: int) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.n_regions = int(n_regions)
        self.n_nodes = self.n_features + self.n_regions + 1
        self.feature_logits = nn.Embedding(self.n_features, 2)
        self.region_logits = nn.Embedding(self.n_regions, 2)
        self.bias = nn.Parameter(torch.zeros(2))
        nn.init.zeros_(self.feature_logits.weight)
        nn.init.zeros_(self.region_logits.weight)

    def forward(self, data: Batch) -> torch.Tensor:
        batch_size = int(data.batch.max().item()) + 1 if data.batch.numel() else 0
        values = data.x.view(batch_size, self.n_nodes)
        feature_values = values[:, : self.n_features]
        region_values = values[:, self.n_features : self.n_features + self.n_regions]
        return (feature_values @ self.feature_logits.weight) + (region_values @ self.region_logits.weight) + self.bias


def collate_graphs(batch: list[Data]) -> Batch:
    return Batch.from_data_list(batch)


def train_one_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        batch = batch.to(device)
        y = batch.y.view(-1).to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        n = int(y.shape[0])
        total_loss += float(loss.item()) * n
        total_n += n
    return total_loss / max(total_n, 1)


def train_lbfgs(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    lr: float,
    max_iter: int,
    l2: float,
) -> float:
    model.train()
    optimizer = torch.optim.LBFGS(
        model.parameters(),
        lr=float(lr),
        max_iter=int(max_iter),
        line_search_fn="strong_wolfe",
    )
    loss_fn = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        total_loss = torch.zeros((), dtype=torch.float32, device=device)
        total_n = 0
        for batch in loader:
            batch = batch.to(device)
            y = batch.y.view(-1).to(device)
            logits = model(batch)
            loss = loss_fn(logits, y)
            n = int(y.shape[0])
            total_loss = total_loss + (loss * n)
            total_n += n
        total_loss = total_loss / max(total_n, 1)
        if float(l2) > 0.0:
            reg = torch.zeros((), dtype=torch.float32, device=device)
            for name, param in model.named_parameters():
                if param.requires_grad and "bias" not in name:
                    reg = reg + torch.sum(param * param)
            total_loss = total_loss + (0.5 * float(l2) * reg)
        total_loss.backward()
        return total_loss

    loss = optimizer.step(closure)
    return float(loss.detach().cpu().item()) if torch.is_tensor(loss) else float(loss)


@torch.no_grad()
def predict(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        p = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        y_all.append(batch.y.view(-1).detach().cpu().numpy())
        p_all.append(p)
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


def parse_optional_fold_ids(text: str) -> set[int] | None:
    values = [x.strip() for x in str(text).split(",") if x.strip()]
    if not values:
        return None
    return {int(x) for x in values}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repeated 5x10 CV for pure GNN with temporal graph features and explicit region nodes.")
    p.add_argument("--pcc-manifest", type=Path, default=Path("data/processed/graphs_pcc_topk_task/manifest.csv"))
    p.add_argument("--theta-manifest", type=Path, default=Path("data/processed/graphs_plv_theta_topk_task/manifest.csv"))
    p.add_argument("--alpha-manifest", type=Path, default=Path("data/processed/graphs_plv_alpha_topk_task/manifest.csv"))
    p.add_argument("--beta-manifest", type=Path, default=Path("data/processed/graphs_plv_beta_topk_task/manifest.csv"))
    p.add_argument("--split-csv", type=Path, default=Path("data/splits/task_split_subject_level.csv"))
    p.add_argument("--items-cache-path", type=Path, default=Path("outputs/cache/spatiotemporal_items/ms120_w16p0_s8p0_mw12_q0p8_me30_k4.pt"))
    p.add_argument("--max-seconds", type=int, default=120)
    p.add_argument("--window-seconds", type=float, default=16.0)
    p.add_argument("--step-seconds", type=float, default=8.0)
    p.add_argument("--max-windows", type=int, default=12)
    p.add_argument("--pcc-quantile", type=float, default=0.8)
    p.add_argument("--min-edges", type=int, default=30)
    p.add_argument("--top-k-per-node", type=int, default=4)
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "lbfgs"])
    p.add_argument("--lbfgs-max-iter", type=int, default=100)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--patience", type=int, default=120)
    p.add_argument("--min-epochs", type=int, default=120)
    p.add_argument("--selection-metric", type=str, default="threshold_accuracy", choices=["accuracy", "threshold_accuracy", "auc", "neg_log_loss"])
    p.add_argument("--thr-low", type=float, default=0.05)
    p.add_argument("--thr-high", type=float, default=0.95)
    p.add_argument("--thr-step", type=float, default=0.01)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--fold-ids", type=str, default="")
    p.add_argument("--no-temporal", action="store_true")
    p.add_argument("--temporal-mode", type=str, default="graph", choices=["graph", "node"])
    p.add_argument("--temporal-scale", type=float, default=1.0)
    p.add_argument("--region-scale", type=float, default=1.0)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/metrics/runs/region_temporal_feature_node_gnn_5x10"))
    p.add_argument("--experiment-name", type=str, default="region_temporal_feature_node_gnn_5x10")
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
    static_meta, static_x = build_static_graph_feature_table(args.pcc_manifest, args.theta_manifest, args.alpha_manifest, args.beta_manifest)
    x_all = static_x
    temporal_meta = None
    temporal_x = None
    if not bool(args.no_temporal):
        temporal_items = load_or_build_temporal_items(args)
        if args.temporal_mode == "node":
            temporal_meta, temporal_x = build_temporal_node_feature_table(temporal_items)
        else:
            temporal_meta, temporal_x = build_temporal_graph_feature_table(temporal_items)
        temporal_x = temporal_x.astype(np.float32)
        if static_meta["subject_id"].tolist() != temporal_meta["subject_id"].tolist():
            raise ValueError("Static and temporal subject orders do not match.")
        x_all = np.concatenate([static_x, temporal_x], axis=1).astype(np.float32)

    y_all = static_meta["label"].to_numpy(dtype=np.int64)
    subjects = static_meta["subject_id"].to_numpy()
    base_map = base_feature_region_map(DEFAULT_CH_NAMES)
    temporal_map = None
    if not bool(args.no_temporal):
        if args.temporal_mode == "node":
            node_map = node_feature_region_map(DEFAULT_CH_NAMES)
            temporal_map = node_map + node_map + node_map
        else:
            temporal_map = base_map + base_map + base_map
    feature_region_map = combined_feature_region_map(base_map, temporal_map)
    if len(feature_region_map) != int(x_all.shape[1]):
        raise ValueError(f"Feature-region map length mismatch: {len(feature_region_map)} != {x_all.shape[1]}")

    run_payloads: list[dict[str, object]] = []
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        fold_results: list[dict[str, object]] = []
        for fold_no, (trainval_idx, test_idx) in enumerate(skf.split(np.arange(len(y_all)), y_all), start=1):
            if fold_filter is not None and fold_no not in fold_filter:
                continue
            y_trainval = y_all[trainval_idx]
            tr_idx, val_idx = train_test_split(
                np.arange(len(trainval_idx)),
                test_size=float(args.val_size),
                random_state=seed + fold_no,
                stratify=y_trainval,
            )
            train_idx = trainval_idx[tr_idx]
            val_abs_idx = trainval_idx[val_idx]
            x_train, x_val, x_test = standardize_features(x_all[train_idx], x_all[val_abs_idx], x_all[test_idx])
            if temporal_x is not None:
                static_dim = int(static_x.shape[1])
                x_train[:, static_dim:] *= float(args.temporal_scale)
                x_val[:, static_dim:] *= float(args.temporal_scale)
                x_test[:, static_dim:] *= float(args.temporal_scale)
            train_ds = RegionTemporalFeatureNodeDataset(x_train, y_all[train_idx], subjects[train_idx], feature_region_map, region_scale=args.region_scale)
            val_ds = RegionTemporalFeatureNodeDataset(x_val, y_all[val_abs_idx], subjects[val_abs_idx], feature_region_map, region_scale=args.region_scale)
            test_ds = RegionTemporalFeatureNodeDataset(x_test, y_all[test_idx], subjects[test_idx], feature_region_map, region_scale=args.region_scale)
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_graphs)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)
            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)

            model = RegionTemporalFeatureNodeGNN(n_features=int(x_all.shape[1]), n_regions=len(REGION_KEYS)).to(device)
            best_state = None
            best_epoch = 1
            best_val_score = -1.0
            best_val_tiebreak = -float("inf")
            if args.optimizer == "lbfgs":
                _ = train_lbfgs(
                    model=model,
                    loader=train_loader,
                    device=device,
                    lr=args.lr,
                    max_iter=args.lbfgs_max_iter,
                    l2=args.weight_decay,
                )
                y_val, p_val = predict(model, val_loader, device)
                best_val_score = validation_score(args.selection_metric, y_val, p_val, args.thr_low, args.thr_high, args.thr_step)
                best_val_tiebreak = validation_score("neg_log_loss", y_val, p_val, args.thr_low, args.thr_high, args.thr_step)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                stale = 0
                for epoch in range(1, args.epochs + 1):
                    _ = train_one_epoch(model, train_loader, optimizer, device)
                    y_val, p_val = predict(model, val_loader, device)
                    val_score = validation_score(args.selection_metric, y_val, p_val, args.thr_low, args.thr_high, args.thr_step)
                    val_tiebreak = validation_score("neg_log_loss", y_val, p_val, args.thr_low, args.thr_high, args.thr_step)
                    if (val_score > best_val_score) or (abs(val_score - best_val_score) <= 1e-12 and val_tiebreak > best_val_tiebreak):
                        best_val_score = val_score
                        best_val_tiebreak = val_tiebreak
                        best_epoch = int(epoch)
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        stale = 0
                    else:
                        stale += 1
                    if epoch >= args.min_epochs and stale >= args.patience:
                        break

            if best_state is not None:
                model.load_state_dict(best_state)
            y_val, p_val = predict(model, val_loader, device)
            thr_sel, val_thr_acc = select_threshold(y_val, p_val, args.thr_low, args.thr_high, args.thr_step)
            y_test, p_test = predict(model, test_loader, device)
            metrics = calc_metrics(y_test, p_test, threshold=thr_sel)
            fold_results.append(
                {
                    "fold": fold_no,
                    "n_train": int(len(train_ds)),
                    "n_val": int(len(val_ds)),
                    "n_test": int(len(test_ds)),
                    "best_epoch": int(best_epoch),
                    "selection_metric": args.selection_metric,
                    "best_val_score": float(best_val_score),
                    "val_threshold_accuracy": float(val_thr_acc),
                    "test_threshold": float(thr_sel),
                    "test_subject_ids": subjects[test_idx].tolist(),
                    "test_metrics": metrics,
                }
            )
            print(
                f"seed={seed} fold={fold_no:02d} "
                f"acc={metrics['accuracy']:.4f} "
                f"bal_acc={metrics['balanced_accuracy']:.4f} "
                f"f1={metrics['f1']:.4f} "
                f"auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'NA'} "
                f"epoch={best_epoch} thr={thr_sel:.2f}"
            )

        if not fold_results:
            raise RuntimeError(f"No folds executed for seed={seed}; check --fold-ids.")
        payload = {
            "schema_version": "eeg_mdd_region_temporal_feature_node_gnn_5x10_v1",
            "experiment_name": args.experiment_name,
            "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model": "RegionTemporalFeatureNodeGNN",
            "device": str(device),
            "params": {
                "folds": int(args.folds),
                "seed": int(seed),
                "seeds": [int(x) for x in seeds],
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "optimizer": args.optimizer,
                "lbfgs_max_iter": int(args.lbfgs_max_iter),
                "patience": int(args.patience),
                "min_epochs": int(args.min_epochs),
                "selection_metric": args.selection_metric,
                "thr_low": float(args.thr_low),
                "thr_high": float(args.thr_high),
                "thr_step": float(args.thr_step),
                "val_size": float(args.val_size),
                "fold_ids": sorted(fold_filter or []),
                "include_temporal": not bool(args.no_temporal),
                "temporal_mode": args.temporal_mode,
                "temporal_scale": float(args.temporal_scale),
                "region_scale": float(args.region_scale),
                "region_keys": REGION_KEYS,
                "split_csv": str(args.split_csv),
                "items_cache_path": str(args.items_cache_path),
                "window_seconds": float(args.window_seconds),
                "step_seconds": float(args.step_seconds),
                "max_windows": int(args.max_windows),
            },
            "feature_shape": {
                "n_subjects": int(x_all.shape[0]),
                "n_static_features": int(static_x.shape[1]),
                "n_temporal_features": 0 if temporal_x is None else int(temporal_x.shape[1]),
                "n_total_features": int(x_all.shape[1]),
                "n_region_nodes": int(len(REGION_KEYS)),
            },
            "aggregate": aggregate_fold_results(fold_results),
            "fold_results": fold_results,
        }
        out_path = args.out_dir / f"region_temporal_feature_node_gnn_seed{seed}_cv10.json"
        write_json(out_path, payload)
        run_payloads.append({**payload, "_out_path": str(out_path)})
        print(f"saved: {out_path}")

    summary = summarize_run_payloads(run_payloads)
    summary_payload = {
        "schema_version": "eeg_mdd_region_temporal_feature_node_gnn_5x10_summary_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "RegionTemporalFeatureNodeGNN",
        "params": {
            "folds": int(args.folds),
            "seeds": [int(x) for x in seeds],
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "optimizer": args.optimizer,
            "lbfgs_max_iter": int(args.lbfgs_max_iter),
            "patience": int(args.patience),
            "min_epochs": int(args.min_epochs),
            "selection_metric": args.selection_metric,
            "thr_low": float(args.thr_low),
            "thr_high": float(args.thr_high),
            "thr_step": float(args.thr_step),
            "val_size": float(args.val_size),
            "fold_ids": sorted(fold_filter or []),
            "include_temporal": not bool(args.no_temporal),
            "temporal_mode": args.temporal_mode,
            "temporal_scale": float(args.temporal_scale),
            "region_scale": float(args.region_scale),
            "region_keys": REGION_KEYS,
        },
        "feature_shape": {
            "n_subjects": int(x_all.shape[0]),
            "n_static_features": int(static_x.shape[1]),
            "n_temporal_features": 0 if temporal_x is None else int(temporal_x.shape[1]),
            "n_total_features": int(x_all.shape[1]),
            "n_region_nodes": int(len(REGION_KEYS)),
        },
        **summary,
    }
    summary_path = args.out_dir / "summary_5x10.json"
    write_json(summary_path, summary_payload)
    print(f"\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
