from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool


def load_base_module():
    src_dir = str(Path("src").resolve())
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import train_gnn_dualgraph_multiband_spatiotemporal_cv10 as mod

    return mod


def build_or_load_items(base_mod, args: argparse.Namespace):
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
    items = base_mod.build_subject_sequences(build_args)
    args.items_cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta_expect, "items": items}, args.items_cache_path)
    return items


def augment_graph_with_adj_row(graph: Data) -> Data:
    x = graph.x.detach().cpu().numpy().astype(np.float32)
    edge_index = graph.edge_index.detach().cpu().numpy().astype(np.int64)
    edge_weight = graph.edge_attr.detach().cpu().numpy().astype(np.float32)
    n_nodes = int(x.shape[0])
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for (src, dst), weight in zip(edge_index.T, edge_weight):
        adj[int(src), int(dst)] = float(weight)
    x_aug = np.concatenate([x, adj], axis=1).astype(np.float32)
    return Data(
        x=torch.tensor(x_aug, dtype=torch.float32),
        edge_index=graph.edge_index.clone(),
        edge_attr=graph.edge_attr.clone(),
        y=graph.y.clone(),
    )


class AugmentedTemporalDataset(Dataset):
    def __init__(self, items: list[dict[str, object]]) -> None:
        self.items: list[dict[str, object]] = []
        for item in items:
            new_windows: list[dict[str, Data]] = []
            for window in item["windows"]:
                new_windows.append(
                    {
                        "pcc": augment_graph_with_adj_row(window["pcc"]),
                        "theta": augment_graph_with_adj_row(window["theta"]),
                        "alpha": augment_graph_with_adj_row(window["alpha"]),
                        "beta": augment_graph_with_adj_row(window["beta"]),
                    }
                )
            self.items.append(
                {
                    "subject_id": str(item["subject_id"]),
                    "label": int(item["label"]),
                    "windows": new_windows,
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


class SignedGCNNodeGraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 16, dropout: float = 0.2) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
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
        g = global_mean_pool(x, batch)
        return x, g


class HybridWindowSpatioTemporalGNN(nn.Module):
    def __init__(self, in_dim: int, n_nodes: int, hidden_dim: int = 16, temporal_hidden: int = 32, dropout: float = 0.3) -> None:
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.hidden_dim = int(hidden_dim)
        self.enc_pcc = SignedGCNNodeGraphEncoder(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.enc_theta = SignedGCNNodeGraphEncoder(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.enc_alpha = SignedGCNNodeGraphEncoder(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.enc_beta = SignedGCNNodeGraphEncoder(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)

        self.band_gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        flat_dim = self.n_nodes * self.hidden_dim * 4
        self.flat_proj = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.window_proj = nn.Sequential(
            nn.Linear(hidden_dim * 7, hidden_dim * 3),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.temporal = nn.GRU(
            input_size=hidden_dim * 2,
            hidden_size=temporal_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.temporal_attn = nn.Linear(temporal_hidden * 2, 1)
        self.classifier = nn.Linear(temporal_hidden * 2, 2)

    def _reshape_nodes(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        return x.view(batch_size, self.n_nodes, self.hidden_dim)

    def forward(self, seq_windows: list[list[dict[str, Data]]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        x_pcc, g_pcc = self.enc_pcc(pcc_batch)
        x_theta, g_theta = self.enc_theta(theta_batch)
        x_alpha, g_alpha = self.enc_alpha(alpha_batch)
        x_beta, g_beta = self.enc_beta(beta_batch)

        g_cat = torch.cat([g_pcc, g_theta, g_alpha, g_beta], dim=1)
        gate = F.softmax(self.band_gate(g_cat), dim=1)

        g_stack = torch.stack([g_pcc, g_theta, g_alpha, g_beta], dim=1)
        g_weighted = (gate.unsqueeze(-1) * g_stack).sum(dim=1)

        x_pcc = self._reshape_nodes(x_pcc, pcc_batch.batch)
        x_theta = self._reshape_nodes(x_theta, theta_batch.batch)
        x_alpha = self._reshape_nodes(x_alpha, alpha_batch.batch)
        x_beta = self._reshape_nodes(x_beta, beta_batch.batch)
        node_stack = torch.stack([x_pcc, x_theta, x_alpha, x_beta], dim=1)
        weighted_nodes = node_stack * gate.unsqueeze(-1).unsqueeze(-1)
        flat_nodes = weighted_nodes.reshape(weighted_nodes.shape[0], -1)
        flat_feat = self.flat_proj(flat_nodes)
        window_feat = self.window_proj(torch.cat([g_weighted, g_cat, flat_feat], dim=1))

        seq_feats: list[torch.Tensor] = []
        start = 0
        for length in lengths:
            seq_feats.append(window_feat[start : start + length])
            start += length

        padded = pad_sequence(seq_feats, batch_first=True)
        packed = pack_padded_sequence(padded, lengths=lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.temporal(packed)
        temporal_out, _ = pad_packed_sequence(packed_out, batch_first=True)

        max_len = int(temporal_out.shape[1])
        mask = torch.arange(max_len, device=device).unsqueeze(0) < torch.tensor(lengths, device=device).unsqueeze(1)
        attn_logits = self.temporal_attn(temporal_out).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask, -1e9)
        attn = F.softmax(attn_logits, dim=1)
        summary = torch.sum(attn.unsqueeze(-1) * temporal_out, dim=1)
        logits = self.classifier(summary)
        return logits, gate, attn


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    label_smoothing: float,
) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        y = batch["y"].to(device)
        optimizer.zero_grad()
        logits, _, _ = model(batch["seq_windows"])
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        n = int(y.shape[0])
        total_loss += float(loss.item()) * n
        total_n += n
    return total_loss / max(total_n, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    g_all: list[np.ndarray] = []
    a_all: list[np.ndarray] = []
    for batch in loader:
        y = batch["y"].to(device)
        logits, gate, attn = model(batch["seq_windows"])
        p = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        y_all.append(y.detach().cpu().numpy())
        p_all.append(p)
        g_all.append(gate.detach().cpu().numpy())
        a_all.append(attn.detach().cpu().numpy())
    return np.concatenate(y_all), np.concatenate(p_all), np.concatenate(g_all, axis=0), np.concatenate(a_all, axis=0)


def calc_metrics(y_true: np.ndarray, prob_pos: np.ndarray, threshold: float = 0.5) -> dict[str, object]:
    pred = (prob_pos >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, prob_pos)) if len(np.unique(y_true)) == 2 else None,
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def select_threshold(y_true: np.ndarray, prob_pos: np.ndarray, low: float, high: float, step: float) -> tuple[float, float]:
    thresholds = np.arange(float(low), float(high) + (float(step) * 0.5), float(step), dtype=np.float64)
    best_thr = 0.5
    best_acc = -1.0
    for thr in thresholds:
        acc = float(accuracy_score(y_true, (prob_pos >= thr).astype(np.int64)))
        if (acc > best_acc) or (abs(acc - best_acc) <= 1e-12 and abs(thr - 0.5) < abs(best_thr - 0.5)):
            best_acc = acc
            best_thr = float(thr)
    return best_thr, best_acc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="10-fold CV for hybrid-window pure spatiotemporal GNN")
    p.add_argument("--split-csv", type=Path, default=Path("data/splits/task_split_subject_level.csv"))
    p.add_argument("--items-cache-path", type=Path, default=Path("outputs/cache/spatiotemporal_items/ms120_w16p0_s8p0_mw12_q0p8_me30_k4.pt"))
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--hidden-dim", type=int, default=16)
    p.add_argument("--temporal-hidden", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--min-epochs", type=int, default=40)
    p.add_argument("--thr-low", type=float, default=0.3)
    p.add_argument("--thr-high", type=float, default=0.7)
    p.add_argument("--thr-step", type=float, default=0.01)
    p.add_argument("--max-seconds", type=int, default=120)
    p.add_argument("--window-seconds", type=float, default=16.0)
    p.add_argument("--step-seconds", type=float, default=8.0)
    p.add_argument("--max-windows", type=int, default=12)
    p.add_argument("--pcc-quantile", type=float, default=0.8)
    p.add_argument("--min-edges", type=int, default=30)
    p.add_argument("--top-k-per-node", type=int, default=4)
    p.add_argument("--out-path", type=Path, default=Path("outputs/metrics/runs/hybridwin_spatiotemporal_cv10/hybridwin_spatiotemporal_cv10.json"))
    p.add_argument("--experiment-name", type=str, default="hybridwin_spatiotemporal_cv10")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_mod = load_base_module()
    items = build_or_load_items(base_mod, args)

    y_all = np.array([int(x["label"]) for x in items], dtype=np.int64)
    idx_all = np.arange(len(items))
    sample_graph = augment_graph_with_adj_row(items[0]["windows"][0]["pcc"])
    n_nodes = int(sample_graph.x.shape[0])
    in_dim = int(sample_graph.x.shape[1])

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

        train_ds = AugmentedTemporalDataset(train_items)
        val_ds = AugmentedTemporalDataset(val_items)
        test_ds = AugmentedTemporalDataset(test_items)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

        model = HybridWindowSpatioTemporalGNN(
            in_dim=in_dim,
            n_nodes=n_nodes,
            hidden_dim=args.hidden_dim,
            temporal_hidden=args.temporal_hidden,
            dropout=args.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_state = None
        best_epoch = 1
        best_val_acc = -1.0
        stale = 0
        for epoch in range(1, args.epochs + 1):
            _ = train_one_epoch(model, train_loader, optimizer, device, label_smoothing=args.label_smoothing)
            y_val, p_val, _, _ = predict(model, val_loader, device)
            val_acc = float(accuracy_score(y_val, (p_val >= 0.5).astype(np.int64)))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if epoch >= args.min_epochs and stale >= args.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        y_val, p_val, _, _ = predict(model, val_loader, device)
        thr_sel, val_thr_acc = select_threshold(y_val, p_val, low=args.thr_low, high=args.thr_high, step=args.thr_step)
        y_test, p_test, gate_test, attn_test = predict(model, test_loader, device)
        metrics = calc_metrics(y_test, p_test, threshold=thr_sel)
        gate_mean = gate_test.mean(axis=0)
        fold_results.append(
            {
                "fold": fold_no,
                "n_train": int(len(train_ds)),
                "n_val": int(len(val_ds)),
                "n_test": int(len(test_ds)),
                "best_epoch": int(best_epoch),
                "best_val_accuracy": float(best_val_acc),
                "val_threshold_accuracy": float(val_thr_acc),
                "test_threshold": float(thr_sel),
                "gate_mean": {
                    "pcc": float(gate_mean[0]),
                    "theta": float(gate_mean[1]),
                    "alpha": float(gate_mean[2]),
                    "beta": float(gate_mean[3]),
                },
                "temporal_attention_peak_mean": float(attn_test.max(axis=1).mean()),
                "test_metrics": metrics,
            }
        )
        print(
            f"fold={fold_no:02d} acc={metrics['accuracy']:.4f} bal_acc={metrics['balanced_accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'NA'} "
            f"epoch={best_epoch} thr={thr_sel:.2f}"
        )

    aggregate: dict[str, dict[str, float]] = {}
    for metric_name in ["accuracy", "balanced_accuracy", "f1", "roc_auc"]:
        vals = np.array([fr["test_metrics"][metric_name] for fr in fold_results], dtype=np.float64)
        aggregate[metric_name] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}

    payload = {
        "schema_version": "eeg_mdd_hybridwin_spatiotemporal_gnn_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "HybridWindowSpatioTemporalGNN",
        "device": str(device),
        "params": {
            "folds": args.folds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "hidden_dim": int(args.hidden_dim),
            "temporal_hidden": int(args.temporal_hidden),
            "dropout": float(args.dropout),
            "label_smoothing": float(args.label_smoothing),
            "seed": int(args.seed),
            "patience": int(args.patience),
            "min_epochs": int(args.min_epochs),
            "thr_low": float(args.thr_low),
            "thr_high": float(args.thr_high),
            "thr_step": float(args.thr_step),
            "split_csv": str(args.split_csv),
            "items_cache_path": str(args.items_cache_path),
        },
        "aggregate": aggregate,
        "fold_results": fold_results,
    }
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
