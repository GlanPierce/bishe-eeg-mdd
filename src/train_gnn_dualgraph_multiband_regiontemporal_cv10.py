from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool

from build_graphs import build_edges, extract_node_features


REGION_GROUPS: dict[str, str] = {
    "FP1": "frontal",
    "FP2": "frontal",
    "F3": "frontal",
    "F4": "frontal",
    "F7": "frontal",
    "F8": "frontal",
    "FZ": "frontal",
    "C3": "central",
    "C4": "central",
    "CZ": "central",
    "T3": "temporal",
    "T4": "temporal",
    "T5": "temporal",
    "T6": "temporal",
    "P3": "parietal",
    "P4": "parietal",
    "PZ": "parietal",
    "O1": "occipital",
    "O2": "occipital",
}
REGION_ORDER = ["frontal", "central", "temporal", "parietal", "occipital", "misc"]
REGION_TO_INDEX = {name: idx for idx, name in enumerate(REGION_ORDER)}


def normalize_channel_name(name: str) -> str:
    s = str(name).upper().strip()
    if s.startswith("EEG "):
        s = s[4:]
    if s.endswith("-LE"):
        s = s[:-3]
    if s.endswith("-REF"):
        s = s[:-4]
    return s.strip()


def infer_region_index(channel_name: str) -> int:
    norm = normalize_channel_name(channel_name)
    region = REGION_GROUPS.get(norm, "misc")
    return int(REGION_TO_INDEX[region])


def read_channel_names_from_edf(file_path: Path) -> list[str]:
    raw = mne.io.read_raw_edf(file_path, preload=False, verbose="ERROR")
    raw.pick("eeg")
    return list(raw.ch_names)


class SignedGCNNodeEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1_pos = GCNConv(in_dim, hidden_dim)
        self.conv1_neg = GCNConv(in_dim, hidden_dim)
        self.conv2_pos = GCNConv(hidden_dim, hidden_dim)
        self.conv2_neg = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
        ew_pos = torch.clamp(edge_weight, min=0.0)
        ew_neg = torch.clamp(-edge_weight, min=0.0)

        x = self.conv1_pos(x, edge_index, edge_weight=ew_pos) - self.conv1_neg(x, edge_index, edge_weight=ew_neg)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2_pos(x, edge_index, edge_weight=ew_pos) - self.conv2_neg(x, edge_index, edge_weight=ew_neg)
        x = F.relu(x)
        graph_emb = global_mean_pool(x, batch)
        return x, graph_emb


class RegionAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int, region_ids: list[int]) -> None:
        super().__init__()
        self.n_regions = len(REGION_ORDER)
        self.register_buffer("region_ids", torch.tensor(region_ids, dtype=torch.long))
        self.scorer = nn.Linear(hidden_dim, 1)

    def forward(self, node_emb: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nodes_per_graph = int(self.region_ids.numel())
        local_idx = torch.arange(node_emb.shape[0], device=node_emb.device) % nodes_per_graph
        node_region = self.region_ids[local_idx]
        group_index = (batch * self.n_regions) + node_region
        region_emb = global_mean_pool(node_emb, group_index)
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        region_emb = region_emb.view(batch_size, self.n_regions, node_emb.shape[1])
        logits = self.scorer(region_emb).squeeze(-1)
        weights = F.softmax(logits, dim=1)
        summary = torch.sum(weights.unsqueeze(-1) * region_emb, dim=1)
        return summary, weights


class BandWindowEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float, region_ids: list[int]) -> None:
        super().__init__()
        self.node_encoder = SignedGCNNodeEncoder(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.region_pool = RegionAttentionPool(hidden_dim=hidden_dim, region_ids=region_ids)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, data: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        node_emb, graph_emb = self.node_encoder(data)
        region_summary, region_weights = self.region_pool(node_emb=node_emb, batch=data.batch)
        fused = self.fuse(torch.cat([graph_emb, region_summary], dim=1))
        return fused, region_weights


class DualGraphRegionTemporalModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        region_ids: list[int],
        hidden_dim: int = 32,
        temporal_hidden: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        kwargs = {
            "in_dim": in_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "region_ids": region_ids,
        }
        self.enc_pcc = BandWindowEncoder(**kwargs)
        self.enc_theta = BandWindowEncoder(**kwargs)
        self.enc_alpha = BandWindowEncoder(**kwargs)
        self.enc_beta = BandWindowEncoder(**kwargs)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        self.temporal = nn.GRU(
            input_size=hidden_dim,
            hidden_size=temporal_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.temporal_attn = nn.Linear(temporal_hidden * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(temporal_hidden * 2, 2)

    def _encode_windows(
        self,
        pcc_batch: Batch,
        theta_batch: Batch,
        alpha_batch: Batch,
        beta_batch: Batch,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        h_pcc, r_pcc = self.enc_pcc(pcc_batch)
        h_theta, r_theta = self.enc_theta(theta_batch)
        h_alpha, r_alpha = self.enc_alpha(alpha_batch)
        h_beta, r_beta = self.enc_beta(beta_batch)

        h_cat = torch.cat([h_pcc, h_theta, h_alpha, h_beta], dim=1)
        band_weights = F.softmax(self.gate(h_cat), dim=1)
        h_stack = torch.stack([h_pcc, h_theta, h_alpha, h_beta], dim=1)
        win_emb = torch.sum(band_weights.unsqueeze(-1) * h_stack, dim=1)
        region_weights = {
            "pcc": r_pcc,
            "theta": r_theta,
            "alpha": r_alpha,
            "beta": r_beta,
        }
        return win_emb, {"band": band_weights}, region_weights

    def forward(self, seq_windows: list[list[dict[str, Data]]]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = next(self.parameters()).device

        lengths: list[int] = []
        pcc_list: list[Data] = []
        theta_list: list[Data] = []
        alpha_list: list[Data] = []
        beta_list: list[Data] = []

        for ws in seq_windows:
            lengths.append(len(ws))
            for w in ws:
                pcc_list.append(w["pcc"])
                theta_list.append(w["theta"])
                alpha_list.append(w["alpha"])
                beta_list.append(w["beta"])

        pcc_batch = Batch.from_data_list(pcc_list).to(device)
        theta_batch = Batch.from_data_list(theta_list).to(device)
        alpha_batch = Batch.from_data_list(alpha_list).to(device)
        beta_batch = Batch.from_data_list(beta_list).to(device)

        win_emb, band_aux, region_aux = self._encode_windows(
            pcc_batch=pcc_batch,
            theta_batch=theta_batch,
            alpha_batch=alpha_batch,
            beta_batch=beta_batch,
        )

        seq_feats: list[torch.Tensor] = []
        st = 0
        for length in lengths:
            seq_feats.append(win_emb[st : st + length])
            st += length

        padded = pad_sequence(seq_feats, batch_first=True)
        packed = pack_padded_sequence(padded, lengths=lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.temporal(packed)
        temporal_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        max_len = int(temporal_out.shape[1])
        mask = torch.arange(max_len, device=device).unsqueeze(0) < torch.tensor(lengths, device=device).unsqueeze(1)
        attn_logits = self.temporal_attn(temporal_out).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask, -1e9)
        temporal_weights = F.softmax(attn_logits, dim=1)
        temporal_summary = torch.sum(temporal_weights.unsqueeze(-1) * temporal_out, dim=1)
        logits = self.classifier(self.dropout(temporal_summary))

        aux = {
            "band": band_aux["band"],
            "temporal": temporal_weights,
            "region_pcc": region_aux["pcc"],
            "region_theta": region_aux["theta"],
            "region_alpha": region_aux["alpha"],
            "region_beta": region_aux["beta"],
        }
        return logits, aux


class TemporalDualGraphDataset(Dataset):
    def __init__(self, items: list[dict[str, object]]) -> None:
        self.items = items

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


def calc_metrics(y_true: np.ndarray, prob_pos: np.ndarray, threshold: float = 0.5) -> dict[str, object]:
    pred = (prob_pos >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, prob_pos)) if len(np.unique(y_true)) == 2 else None,
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


def fit_temperature_from_val(score_val: np.ndarray, y_val: np.ndarray) -> float:
    eps = 1e-8
    temps = np.concatenate(
        [
            np.linspace(0.5, 3.0, 51, dtype=np.float64),
            np.linspace(3.1, 8.0, 50, dtype=np.float64),
        ]
    )
    y = y_val.astype(np.float64)
    best_t = 1.0
    best_nll = float("inf")
    for t in temps:
        p = np.clip(_sigmoid(score_val / t), eps, 1.0 - eps)
        nll = float(-np.mean((y * np.log(p)) + ((1.0 - y) * np.log(1.0 - p))))
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t


def calibrate_with_val(
    method: str,
    y_val: np.ndarray,
    score_val: np.ndarray,
    score_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    method = str(method).lower()
    if method == "none":
        return _sigmoid(score_val), _sigmoid(score_test), {"method": "none"}

    if np.unique(y_val).size < 2:
        return _sigmoid(score_val), _sigmoid(score_test), {"method": "none_fallback_single_class_val"}

    if method == "temperature":
        temp = fit_temperature_from_val(score_val=score_val, y_val=y_val)
        return _sigmoid(score_val / temp), _sigmoid(score_test / temp), {"method": "temperature", "temperature": float(temp)}

    if method == "platt":
        clf = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=int(seed),
        )
        clf.fit(score_val.reshape(-1, 1), y_val.astype(np.int64))
        p_val = clf.predict_proba(score_val.reshape(-1, 1))[:, 1]
        p_test = clf.predict_proba(score_test.reshape(-1, 1))[:, 1]
        return p_val, p_test, {"method": "platt"}

    raise ValueError(f"Unknown calibration method: {method}")


def select_threshold_from_val(
    y_true: np.ndarray,
    prob_pos: np.ndarray,
    metric: str = "balanced_accuracy",
    step: float = 0.01,
    low: float = 0.05,
    high: float = 0.95,
) -> tuple[float, float]:
    step = float(max(1e-4, step))
    low = float(np.clip(low, 0.0, 1.0))
    high = float(np.clip(high, 0.0, 1.0))
    if high < low:
        low, high = high, low

    thresholds = np.arange(low, high + (step * 0.5), step, dtype=np.float64)
    if thresholds.size == 0:
        thresholds = np.array([0.5], dtype=np.float64)

    best_thr = 0.5
    best_score = -1.0
    metric = str(metric).lower()
    for thr in thresholds:
        pred = (prob_pos >= float(thr)).astype(np.int64)
        if metric == "balanced_accuracy":
            score = float(balanced_accuracy_score(y_true, pred))
        elif metric == "f1":
            score = float(f1_score(y_true, pred))
        elif metric == "accuracy":
            score = float(accuracy_score(y_true, pred))
        else:
            raise ValueError(f"Unsupported threshold metric: {metric}")
        if (score > best_score) or (abs(score - best_score) <= 1e-12 and abs(thr - 0.5) < abs(best_thr - 0.5)):
            best_score = score
            best_thr = float(thr)
    return best_thr, best_score


def _window_starts(n_samples: int, win: int, step: int) -> list[int]:
    if n_samples <= win:
        return [0]
    starts = list(range(0, n_samples - win + 1, step))
    if starts[-1] != n_samples - win:
        starts.append(n_samples - win)
    return starts


def _to_graph(x: np.ndarray, edge_index: np.ndarray, edge_weight: np.ndarray, label: int) -> Data:
    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_weight, dtype=torch.float32),
        y=torch.tensor([label], dtype=torch.long),
    )


def build_window_graphs(
    segment: np.ndarray,
    sfreq: float,
    label: int,
    pcc_quantile: float,
    min_edges: int,
    top_k_per_node: int,
) -> dict[str, Data]:
    x = extract_node_features(segment, sfreq).astype(np.float32)

    def build_one(edge_mode: str, band: str, signed_topk_split: bool) -> Data:
        edge_index, edge_weight = build_edges(
            data=segment,
            sfreq=sfreq,
            edge_mode=edge_mode,
            quantile=pcc_quantile,
            min_edges=min_edges,
            edge_selection="per_node_topk",
            top_k_per_node=top_k_per_node,
            signed_topk_split=signed_topk_split,
            fusion_alpha=0.5,
            plv_band=band,
        )
        return _to_graph(x=x, edge_index=edge_index, edge_weight=edge_weight, label=label)

    return {
        "pcc": build_one(edge_mode="pcc", band="broad", signed_topk_split=True),
        "theta": build_one(edge_mode="plv", band="theta", signed_topk_split=False),
        "alpha": build_one(edge_mode="plv", band="alpha", signed_topk_split=False),
        "beta": build_one(edge_mode="plv", band="beta", signed_topk_split=False),
    }


def _cache_meta_matches(meta: dict[str, object], args: argparse.Namespace) -> bool:
    expected = {
        "split_csv": str(args.split_csv),
        "max_seconds": int(args.max_seconds),
        "window_seconds": float(args.window_seconds),
        "step_seconds": float(args.step_seconds),
        "max_windows": int(args.max_windows),
        "pcc_quantile": float(args.pcc_quantile),
        "min_edges": int(args.min_edges),
        "top_k_per_node": int(args.top_k_per_node),
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            return False
    return True


def build_subject_sequences(args: argparse.Namespace) -> list[dict[str, object]]:
    if args.items_cache_path is not None and args.items_cache_path.exists():
        cache_obj = torch.load(args.items_cache_path, map_location="cpu", weights_only=False)
        if isinstance(cache_obj, dict) and "meta" in cache_obj and "items" in cache_obj:
            if _cache_meta_matches(cache_obj["meta"], args):
                print(f"Loaded cache: {args.items_cache_path}")
                return cache_obj["items"]

    df = pd.read_csv(args.split_csv)
    need = {"file_path", "subject_id", "label"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"split csv missing columns: {sorted(miss)}")

    df = df.sort_values("subject_id").drop_duplicates(subset=["subject_id"], keep="first").reset_index(drop=True)
    items: list[dict[str, object]] = []

    for i, row in df.iterrows():
        fp = Path(str(row["file_path"]))
        label = int(row["label"])
        print(f"[{i+1}/{len(df)}] build temporal multiband graphs: {fp.name}")

        raw = mne.io.read_raw_edf(fp, preload=True, verbose="ERROR")
        raw.pick("eeg")
        raw.filter(l_freq=0.5, h_freq=45.0, verbose="ERROR")

        sfreq = float(raw.info["sfreq"])
        n_stop = min(raw.n_times, int(args.max_seconds * sfreq))
        data = raw.get_data(start=0, stop=n_stop)

        win = max(1, int(round(args.window_seconds * sfreq)))
        step = max(1, int(round(args.step_seconds * sfreq)))
        starts = _window_starts(data.shape[1], win, step)[: args.max_windows]

        windows: list[dict[str, Data]] = []
        for start in starts:
            stop = min(start + win, data.shape[1])
            seg = data[:, start:stop]
            if seg.shape[1] < 8:
                continue
            windows.append(
                build_window_graphs(
                    segment=seg,
                    sfreq=sfreq,
                    label=label,
                    pcc_quantile=args.pcc_quantile,
                    min_edges=args.min_edges,
                    top_k_per_node=args.top_k_per_node,
                )
            )

        if not windows:
            windows.append(
                build_window_graphs(
                    segment=data,
                    sfreq=sfreq,
                    label=label,
                    pcc_quantile=args.pcc_quantile,
                    min_edges=args.min_edges,
                    top_k_per_node=args.top_k_per_node,
                )
            )

        items.append(
            {
                "subject_id": str(row["subject_id"]),
                "label": label,
                "windows": windows,
                "n_windows": int(len(windows)),
                "file_path": str(fp),
            }
        )

    if args.items_cache_path is not None:
        args.items_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "meta": {
                    "split_csv": str(args.split_csv),
                    "max_seconds": int(args.max_seconds),
                    "window_seconds": float(args.window_seconds),
                    "step_seconds": float(args.step_seconds),
                    "max_windows": int(args.max_windows),
                    "pcc_quantile": float(args.pcc_quantile),
                    "min_edges": int(args.min_edges),
                    "top_k_per_node": int(args.top_k_per_node),
                },
                "items": items,
            },
            args.items_cache_path,
        )
        print(f"Saved subject window cache: {args.items_cache_path}")
    return items


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
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


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], np.ndarray]:
    model.eval()
    y_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    score_all: list[np.ndarray] = []
    aux_band: list[np.ndarray] = []
    aux_temporal_peak: list[np.ndarray] = []
    aux_region_pcc: list[np.ndarray] = []
    aux_region_theta: list[np.ndarray] = []
    aux_region_alpha: list[np.ndarray] = []
    aux_region_beta: list[np.ndarray] = []

    for batch in loader:
        y = batch["y"].to(device)
        logits, aux = model(batch["seq_windows"])
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        scores = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()
        y_all.append(y.detach().cpu().numpy())
        p_all.append(probs)
        score_all.append(scores)
        aux_band.append(aux["band"].detach().cpu().numpy())
        aux_temporal_peak.append(aux["temporal"].amax(dim=1).detach().cpu().numpy())
        aux_region_pcc.append(aux["region_pcc"].detach().cpu().numpy())
        aux_region_theta.append(aux["region_theta"].detach().cpu().numpy())
        aux_region_alpha.append(aux["region_alpha"].detach().cpu().numpy())
        aux_region_beta.append(aux["region_beta"].detach().cpu().numpy())

    return (
        np.concatenate(y_all),
        np.concatenate(p_all),
        {
            "band": np.concatenate(aux_band, axis=0),
            "temporal_peak": np.concatenate(aux_temporal_peak, axis=0),
            "region_pcc": np.concatenate(aux_region_pcc, axis=0),
            "region_theta": np.concatenate(aux_region_theta, axis=0),
            "region_alpha": np.concatenate(aux_region_alpha, axis=0),
            "region_beta": np.concatenate(aux_region_beta, axis=0),
        },
        np.concatenate(score_all),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="10-fold CV: region-aware dual-graph multiband GNN + temporal attention")
    p.add_argument("--split-csv", type=Path, default=Path("data/splits/task_split_subject_level.csv"))
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temporal-hidden", type=int, default=32)
    p.add_argument("--max-seconds", type=int, default=120)
    p.add_argument("--window-seconds", type=float, default=16.0)
    p.add_argument("--step-seconds", type=float, default=8.0)
    p.add_argument("--max-windows", type=int, default=12)
    p.add_argument("--items-cache-path", type=Path, default=None)
    p.add_argument("--pcc-quantile", type=float, default=0.8)
    p.add_argument("--min-edges", type=int, default=30)
    p.add_argument("--top-k-per-node", type=int, default=4)
    p.add_argument(
        "--threshold-mode",
        type=str,
        default="fixed",
        choices=["fixed", "val_opt"],
    )
    p.add_argument("--threshold-default", type=float, default=0.5)
    p.add_argument(
        "--threshold-metric",
        type=str,
        default="balanced_accuracy",
        choices=["balanced_accuracy", "f1", "accuracy"],
    )
    p.add_argument("--threshold-grid-step", type=float, default=0.01)
    p.add_argument("--threshold-grid-low", type=float, default=0.05)
    p.add_argument("--threshold-grid-high", type=float, default=0.95)
    p.add_argument(
        "--calibration",
        type=str,
        default="none",
        choices=["none", "temperature", "platt"],
    )
    p.add_argument(
        "--out-path",
        type=Path,
        default=Path("outputs/metrics/runs/regiontemporal_cv10/gnn_dualgraph_multiband_regiontemporal_cv10.json"),
    )
    p.add_argument("--experiment-name", type=str, default="gnn_dualgraph_multiband_regiontemporal_cv10")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items = build_subject_sequences(args)
    y_all = np.array([int(x["label"]) for x in items], dtype=np.int64)
    idx_all = np.arange(len(items))

    first_file = Path(str(items[0]["file_path"]))
    channel_names = read_channel_names_from_edf(first_file)
    region_ids = [infer_region_index(ch) for ch in channel_names]
    in_dim = int(items[0]["windows"][0]["pcc"].x.shape[1])

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results: list[dict[str, object]] = []

    for fold_no, (trainval_idx, test_idx) in enumerate(skf.split(idx_all, y_all), start=1):
        trainval_items = [items[i] for i in trainval_idx]
        test_items = [items[i] for i in test_idx]

        trainval_y = np.array([int(x["label"]) for x in trainval_items], dtype=np.int64)
        tr_idx, val_idx = train_test_split(
            np.arange(len(trainval_items)),
            test_size=0.2,
            random_state=args.seed + fold_no,
            stratify=trainval_y,
        )
        train_items = [trainval_items[i] for i in tr_idx]
        val_items = [trainval_items[i] for i in val_idx]

        train_ds = TemporalDualGraphDataset(train_items)
        val_ds = TemporalDualGraphDataset(val_items)
        test_ds = TemporalDualGraphDataset(test_items)

        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        model = DualGraphRegionTemporalModel(
            in_dim=in_dim,
            region_ids=region_ids,
            hidden_dim=args.hidden_dim,
            temporal_hidden=args.temporal_hidden,
            dropout=args.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_state = None
        best_val_bal = -1.0
        for _ in range(args.epochs):
            _ = train_one_epoch(model, train_loader, optimizer, device)
            y_val, p_val, _, _ = predict(model, val_loader, device)
            val_bal = float(balanced_accuracy_score(y_val, (p_val >= 0.5).astype(np.int64)))
            if val_bal > best_val_bal:
                best_val_bal = val_bal
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        y_val_best, _, _, s_val_best = predict(model, val_loader, device)
        y_test, _, aux_test, s_test = predict(model, test_loader, device)
        p_val_best, p_test, calib_info = calibrate_with_val(
            method=args.calibration,
            y_val=y_val_best,
            score_val=s_val_best,
            score_test=s_test,
            seed=args.seed + fold_no,
        )
        if args.threshold_mode == "val_opt":
            test_thr, val_thr_score = select_threshold_from_val(
                y_true=y_val_best,
                prob_pos=p_val_best,
                metric=args.threshold_metric,
                step=args.threshold_grid_step,
                low=args.threshold_grid_low,
                high=args.threshold_grid_high,
            )
        else:
            test_thr = float(args.threshold_default)
            val_thr_score = float("nan")

        metrics = calc_metrics(y_test, p_test, threshold=test_thr)
        band_mean = aux_test["band"].mean(axis=0)
        temporal_mean = float(aux_test["temporal_peak"].mean())
        region_payload = {}
        for key, label in [
            ("region_pcc", "pcc"),
            ("region_theta", "theta"),
            ("region_alpha", "alpha"),
            ("region_beta", "beta"),
        ]:
            region_payload[label] = {
                region: float(aux_test[key].mean(axis=0)[idx])
                for idx, region in enumerate(REGION_ORDER)
            }

        fold_results.append(
            {
                "fold": fold_no,
                "n_train": int(len(train_ds)),
                "n_val": int(len(val_ds)),
                "n_test": int(len(test_ds)),
                "best_val_balanced_accuracy": best_val_bal,
                "test_threshold": float(test_thr),
                "calibration": calib_info,
                "val_threshold_metric": args.threshold_metric if args.threshold_mode == "val_opt" else None,
                "val_threshold_score": None if args.threshold_mode != "val_opt" else float(val_thr_score),
                "avg_windows_test": float(np.mean([int(x["n_windows"]) for x in test_items])),
                "band_weights_mean": {
                    "pcc": float(band_mean[0]),
                    "theta": float(band_mean[1]),
                    "alpha": float(band_mean[2]),
                    "beta": float(band_mean[3]),
                },
                "temporal_attention_mean": temporal_mean,
                "region_attention_mean": region_payload,
                "test_metrics": metrics,
            }
        )

        print(
            f"fold={fold_no:02d} "
            f"acc={metrics['accuracy']:.4f} "
            f"bal_acc={metrics['balanced_accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'NA'} "
            f"thr={test_thr:.2f} "
            f"band=[{band_mean[0]:.3f},{band_mean[1]:.3f},{band_mean[2]:.3f},{band_mean[3]:.3f}]"
        )

    aggregate: dict[str, dict[str, float]] = {}
    for metric_name in ["accuracy", "balanced_accuracy", "f1", "roc_auc"]:
        vals = np.array([fr["test_metrics"][metric_name] for fr in fold_results], dtype=np.float64)
        aggregate[metric_name] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}

    band_agg: dict[str, dict[str, float]] = {}
    for band_name in ["pcc", "theta", "alpha", "beta"]:
        vals = np.array([fr["band_weights_mean"][band_name] for fr in fold_results], dtype=np.float64)
        band_agg[band_name] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}

    region_agg: dict[str, dict[str, dict[str, float]]] = {}
    for band_name in ["pcc", "theta", "alpha", "beta"]:
        band_payload: dict[str, dict[str, float]] = {}
        for region_name in REGION_ORDER:
            vals = np.array([fr["region_attention_mean"][band_name][region_name] for fr in fold_results], dtype=np.float64)
            band_payload[region_name] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
        region_agg[band_name] = band_payload

    payload = {
        "schema_version": "eeg_mdd_benchmark_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "DualGraphRegionTemporalModel",
        "device": str(device),
        "params": {
            "folds": args.folds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "seed": args.seed,
            "temporal_hidden": args.temporal_hidden,
            "max_seconds": args.max_seconds,
            "window_seconds": args.window_seconds,
            "step_seconds": args.step_seconds,
            "max_windows": args.max_windows,
            "items_cache_path": None if args.items_cache_path is None else str(args.items_cache_path),
            "pcc_quantile": args.pcc_quantile,
            "min_edges": args.min_edges,
            "top_k_per_node": args.top_k_per_node,
            "threshold_mode": args.threshold_mode,
            "calibration": args.calibration,
            "threshold_default": args.threshold_default,
            "threshold_metric": args.threshold_metric,
            "threshold_grid_step": args.threshold_grid_step,
            "threshold_grid_low": args.threshold_grid_low,
            "threshold_grid_high": args.threshold_grid_high,
            "regions": REGION_ORDER,
            "channel_names": channel_names,
        },
        "split_csv": str(args.split_csv),
        "dataset_window_summary": {
            "n_subjects": int(len(items)),
            "avg_windows": float(np.mean([int(x["n_windows"]) for x in items])),
            "min_windows": int(np.min([int(x["n_windows"]) for x in items])),
            "max_windows": int(np.max([int(x["n_windows"]) for x in items])),
        },
        "aggregate": aggregate,
        "fusion_weight_aggregate": band_agg,
        "region_attention_aggregate": region_agg,
        "fold_results": fold_results,
    }

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.out_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
