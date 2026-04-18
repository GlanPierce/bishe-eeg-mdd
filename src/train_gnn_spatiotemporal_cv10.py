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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool

from build_graphs import build_edges, extract_node_features


class SignedGCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1_pos = GCNConv(in_dim, hidden_dim)
        self.conv1_neg = GCNConv(in_dim, hidden_dim)
        self.conv2_pos = GCNConv(hidden_dim, hidden_dim)
        self.conv2_neg = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Batch) -> torch.Tensor:
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
        ew_pos = torch.clamp(edge_weight, min=0.0)
        ew_neg = torch.clamp(-edge_weight, min=0.0)
        x = self.conv1_pos(x, edge_index, edge_weight=ew_pos) - self.conv1_neg(x, edge_index, edge_weight=ew_neg)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2_pos(x, edge_index, edge_weight=ew_pos) - self.conv2_neg(x, edge_index, edge_weight=ew_neg)
        x = F.relu(x)
        return global_mean_pool(x, batch)


class SpatioTemporalGCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        gcn_hidden: int = 32,
        temporal_hidden: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = SignedGCNEncoder(in_dim=in_dim, hidden_dim=gcn_hidden, dropout=dropout)
        self.temporal = nn.GRU(
            input_size=gcn_hidden,
            hidden_size=temporal_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(temporal_hidden * 2, 2)

    def forward(self, seq_graphs: list[list[Data]]) -> torch.Tensor:
        lengths = [len(gs) for gs in seq_graphs]
        all_graphs: list[Data] = []
        for gs in seq_graphs:
            all_graphs.extend(gs)

        graph_batch = Batch.from_data_list(all_graphs)
        g_emb_all = self.encoder(graph_batch)

        seq_tensors: list[torch.Tensor] = []
        st = 0
        for L in lengths:
            seq_tensors.append(g_emb_all[st : st + L])
            st += L

        padded = pad_sequence(seq_tensors, batch_first=True)
        packed = pack_padded_sequence(padded, lengths=lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.temporal(packed)

        # bidirectional GRU => [2, B, H]
        h_last = torch.cat([h_n[0], h_n[1]], dim=1)
        h_last = self.dropout(h_last)
        return self.classifier(h_last)


class TemporalGraphDataset(Dataset):
    def __init__(self, items: list[dict[str, object]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.items[idx]


def collate_temporal(batch: list[dict[str, object]]) -> dict[str, object]:
    seq_graphs = [x["graphs"] for x in batch]
    y = torch.tensor([int(x["label"]) for x in batch], dtype=torch.long)
    subject_ids = [str(x["subject_id"]) for x in batch]
    return {"seq_graphs": seq_graphs, "y": y, "subject_ids": subject_ids}


def calc_metrics(y_true: np.ndarray, prob_pos: np.ndarray, threshold: float = 0.5) -> dict[str, object]:
    pred = (prob_pos >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, prob_pos)) if len(np.unique(y_true)) == 2 else None,
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def _window_starts(n_samples: int, win: int, step: int) -> list[int]:
    if n_samples <= win:
        return [0]
    starts = list(range(0, n_samples - win + 1, step))
    if starts[-1] != n_samples - win:
        starts.append(n_samples - win)
    return starts


def build_temporal_graphs_for_file(
    file_path: Path,
    max_seconds: int,
    window_seconds: float,
    step_seconds: float,
    max_windows: int,
    pcc_quantile: float,
    min_edges: int,
    top_k_per_node: int,
) -> list[Data]:
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose="ERROR")
    raw.pick("eeg")
    raw.filter(l_freq=0.5, h_freq=45.0, verbose="ERROR")

    sfreq = float(raw.info["sfreq"])
    n_stop = min(raw.n_times, int(max_seconds * sfreq))
    data = raw.get_data(start=0, stop=n_stop)

    win = max(1, int(round(window_seconds * sfreq)))
    step = max(1, int(round(step_seconds * sfreq)))
    starts = _window_starts(data.shape[1], win, step)[:max_windows]

    seq: list[Data] = []
    for s in starts:
        e = min(s + win, data.shape[1])
        x_win = data[:, s:e]
        if x_win.shape[1] < 8:
            continue

        x = extract_node_features(x_win, sfreq).astype(np.float32)
        edge_index, edge_weight = build_edges(
            data=x_win,
            sfreq=sfreq,
            edge_mode="pcc",
            quantile=pcc_quantile,
            min_edges=min_edges,
            edge_selection="per_node_topk",
            top_k_per_node=top_k_per_node,
            signed_topk_split=True,
            fusion_alpha=0.5,
            plv_band="broad",
        )
        seq.append(
            Data(
                x=torch.tensor(x, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_weight, dtype=torch.float32),
                y=torch.tensor([0], dtype=torch.long),
            )
        )

    if not seq:
        # fallback: whole segment as one graph
        x = extract_node_features(data, sfreq).astype(np.float32)
        edge_index, edge_weight = build_edges(
            data=data,
            sfreq=sfreq,
            edge_mode="pcc",
            quantile=pcc_quantile,
            min_edges=min_edges,
            edge_selection="per_node_topk",
            top_k_per_node=top_k_per_node,
            signed_topk_split=True,
            fusion_alpha=0.5,
            plv_band="broad",
        )
        seq.append(
            Data(
                x=torch.tensor(x, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
                edge_attr=torch.tensor(edge_weight, dtype=torch.float32),
                y=torch.tensor([0], dtype=torch.long),
            )
        )
    return seq


def build_subject_items(args: argparse.Namespace) -> list[dict[str, object]]:
    df = pd.read_csv(args.split_csv)
    required_cols = {"file_path", "subject_id", "label"}
    miss = required_cols - set(df.columns)
    if miss:
        raise ValueError(f"split csv missing columns: {sorted(miss)}")

    # one sample per subject
    df = df.sort_values("subject_id").drop_duplicates(subset=["subject_id"], keep="first").reset_index(drop=True)

    items: list[dict[str, object]] = []
    for i, row in df.iterrows():
        fp = Path(str(row["file_path"]))
        print(f"[{i+1}/{len(df)}] temporal graph build: {fp.name}")
        seq = build_temporal_graphs_for_file(
            file_path=fp,
            max_seconds=args.max_seconds,
            window_seconds=args.window_seconds,
            step_seconds=args.step_seconds,
            max_windows=args.max_windows,
            pcc_quantile=args.pcc_quantile,
            min_edges=args.min_edges,
            top_k_per_node=args.top_k_per_node,
        )
        items.append(
            {
                "subject_id": str(row["subject_id"]),
                "label": int(row["label"]),
                "graphs": seq,
                "n_windows": int(len(seq)),
                "file_path": str(fp),
            }
        )
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
        seq_graphs = [[g.to(device) for g in gs] for gs in batch["seq_graphs"]]
        y = batch["y"].to(device)

        optimizer.zero_grad()
        logits = model(seq_graphs)
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
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    for batch in loader:
        seq_graphs = [[g.to(device) for g in gs] for gs in batch["seq_graphs"]]
        y = batch["y"].to(device)
        logits = model(seq_graphs)
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs_all.append(probs)
        y_all.append(y.detach().cpu().numpy())
    return np.concatenate(y_all), np.concatenate(probs_all)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="10-fold CV for SpatioTemporal GNN (windowed PCC graphs + BiGRU)")
    p.add_argument("--split-csv", type=Path, default=Path("data/splits/task_split_subject_level.csv"))
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--gcn-hidden", type=int, default=32)
    p.add_argument("--temporal-hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--max-seconds", type=int, default=120)
    p.add_argument("--window-seconds", type=float, default=8.0)
    p.add_argument("--step-seconds", type=float, default=4.0)
    p.add_argument("--max-windows", type=int, default=24)
    p.add_argument("--pcc-quantile", type=float, default=0.8)
    p.add_argument("--min-edges", type=int, default=30)
    p.add_argument("--top-k-per-node", type=int, default=4)

    p.add_argument(
        "--out-path",
        type=Path,
        default=Path("outputs/metrics/runs/spatiotemporal_cv10/gnn_spatiotemporal_pcc_cv10.json"),
    )
    p.add_argument("--experiment-name", type=str, default="gnn_spatiotemporal_pcc_cv10")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items = build_subject_items(args)
    y_all = np.array([int(x["label"]) for x in items], dtype=np.int64)
    idx_all = np.arange(len(items))

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results: list[dict[str, object]] = []

    in_dim = int(items[0]["graphs"][0].x.shape[1])

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

        train_ds = TemporalGraphDataset(train_items)
        val_ds = TemporalGraphDataset(val_items)
        test_ds = TemporalGraphDataset(test_items)

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_temporal,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_temporal,
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_temporal,
        )

        model = SpatioTemporalGCN(
            in_dim=in_dim,
            gcn_hidden=args.gcn_hidden,
            temporal_hidden=args.temporal_hidden,
            dropout=args.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_state = None
        best_val_bal = -1.0
        for _ in range(args.epochs):
            _ = train_one_epoch(model, train_loader, optimizer, device)
            y_val, p_val = predict(model, val_loader, device)
            val_bal = float(balanced_accuracy_score(y_val, (p_val >= 0.5).astype(np.int64)))
            if val_bal > best_val_bal:
                best_val_bal = val_bal
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        y_test, p_test = predict(model, test_loader, device)
        metrics = calc_metrics(y_test, p_test)
        fold_results.append(
            {
                "fold": fold_no,
                "n_train": int(len(train_ds)),
                "n_val": int(len(val_ds)),
                "n_test": int(len(test_ds)),
                "best_val_balanced_accuracy": best_val_bal,
                "test_metrics": metrics,
                "avg_windows_test": float(np.mean([int(x["n_windows"]) for x in test_items])),
            }
        )
        print(
            f"fold={fold_no:02d} "
            f"acc={metrics['accuracy']:.4f} "
            f"bal_acc={metrics['balanced_accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'NA'}"
        )

    metric_names = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
    aggregate: dict[str, dict[str, float]] = {}
    for mn in metric_names:
        vals = np.array([fr["test_metrics"][mn] for fr in fold_results], dtype=np.float64)
        aggregate[mn] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}

    payload = {
        "schema_version": "eeg_mdd_benchmark_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "SpatioTemporalGCN(PCC-windowed)+BiGRU",
        "device": str(device),
        "params": {
            "folds": args.folds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "gcn_hidden": args.gcn_hidden,
            "temporal_hidden": args.temporal_hidden,
            "dropout": args.dropout,
            "seed": args.seed,
            "max_seconds": args.max_seconds,
            "window_seconds": args.window_seconds,
            "step_seconds": args.step_seconds,
            "max_windows": args.max_windows,
            "pcc_quantile": args.pcc_quantile,
            "min_edges": args.min_edges,
            "top_k_per_node": args.top_k_per_node,
        },
        "split_csv": str(args.split_csv),
        "aggregate": aggregate,
        "fold_results": fold_results,
        "dataset_window_summary": {
            "n_subjects": int(len(items)),
            "avg_windows": float(np.mean([int(x["n_windows"]) for x in items])),
            "min_windows": int(np.min([int(x["n_windows"]) for x in items])),
            "max_windows": int(np.max([int(x["n_windows"]) for x in items])),
        },
    }

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.out_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
