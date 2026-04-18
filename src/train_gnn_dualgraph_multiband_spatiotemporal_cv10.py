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


class DualGraphMultiBandSpatioTemporalModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 32,
        temporal_hidden: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.enc_pcc = SignedGCNEncoder(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.enc_theta = SignedGCNEncoder(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.enc_alpha = SignedGCNEncoder(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.enc_beta = SignedGCNEncoder(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)

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
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(temporal_hidden * 2, 2)

    def forward(self, seq_windows: list[list[dict[str, Data]]]) -> tuple[torch.Tensor, torch.Tensor]:
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

        h_pcc = self.enc_pcc(pcc_batch)
        h_theta = self.enc_theta(theta_batch)
        h_alpha = self.enc_alpha(alpha_batch)
        h_beta = self.enc_beta(beta_batch)

        h_cat = torch.cat([h_pcc, h_theta, h_alpha, h_beta], dim=1)
        w = F.softmax(self.gate(h_cat), dim=1)
        h_stack = torch.stack([h_pcc, h_theta, h_alpha, h_beta], dim=1)
        h_win = (w.unsqueeze(-1) * h_stack).sum(dim=1)

        seq_feats: list[torch.Tensor] = []
        st = 0
        for L in lengths:
            seq_feats.append(h_win[st : st + L])
            st += L

        padded = pad_sequence(seq_feats, batch_first=True)
        packed = pack_padded_sequence(padded, lengths=lengths, batch_first=True, enforce_sorted=False)
        _, h_n = self.temporal(packed)

        h_last = torch.cat([h_n[0], h_n[1]], dim=1)
        logits = self.classifier(self.dropout(h_last))
        return logits, w


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

    metric = str(metric).lower()
    thresholds = np.arange(low, high + (step * 0.5), step, dtype=np.float64)
    if thresholds.size == 0:
        thresholds = np.array([0.5], dtype=np.float64)

    best_thr = 0.5
    best_score = -1.0
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

        # Tie-break toward threshold close to 0.5 for stability.
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

    pcc_ei, pcc_ew = build_edges(
        data=segment,
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

    theta_ei, theta_ew = build_edges(
        data=segment,
        sfreq=sfreq,
        edge_mode="plv",
        quantile=pcc_quantile,
        min_edges=min_edges,
        edge_selection="per_node_topk",
        top_k_per_node=top_k_per_node,
        signed_topk_split=False,
        fusion_alpha=0.5,
        plv_band="theta",
    )

    alpha_ei, alpha_ew = build_edges(
        data=segment,
        sfreq=sfreq,
        edge_mode="plv",
        quantile=pcc_quantile,
        min_edges=min_edges,
        edge_selection="per_node_topk",
        top_k_per_node=top_k_per_node,
        signed_topk_split=False,
        fusion_alpha=0.5,
        plv_band="alpha",
    )

    beta_ei, beta_ew = build_edges(
        data=segment,
        sfreq=sfreq,
        edge_mode="plv",
        quantile=pcc_quantile,
        min_edges=min_edges,
        edge_selection="per_node_topk",
        top_k_per_node=top_k_per_node,
        signed_topk_split=False,
        fusion_alpha=0.5,
        plv_band="beta",
    )

    return {
        "pcc": _to_graph(x, pcc_ei, pcc_ew, label),
        "theta": _to_graph(x, theta_ei, theta_ew, label),
        "alpha": _to_graph(x, alpha_ei, alpha_ew, label),
        "beta": _to_graph(x, beta_ei, beta_ew, label),
    }


def build_subject_sequences(args: argparse.Namespace) -> list[dict[str, object]]:
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
        for s in starts:
            e = min(s + win, data.shape[1])
            seg = data[:, s:e]
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    w_all: list[np.ndarray] = []

    for batch in loader:
        y = batch["y"].to(device)
        logits, w = model(batch["seq_windows"])
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        y_all.append(y.detach().cpu().numpy())
        p_all.append(probs)
        w_all.append(w.detach().cpu().numpy())

    return np.concatenate(y_all), np.concatenate(p_all), np.concatenate(w_all, axis=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="10-fold CV: Static-best DualGraph Multiband + Temporal GRU")
    p.add_argument("--split-csv", type=Path, default=Path("data/splits/task_split_subject_level.csv"))

    # Keep training hyper-parameters aligned with static-best by default.
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
    p.add_argument("--window-seconds", type=float, default=8.0)
    p.add_argument("--step-seconds", type=float, default=4.0)
    p.add_argument("--max-windows", type=int, default=24)

    p.add_argument("--pcc-quantile", type=float, default=0.8)
    p.add_argument("--min-edges", type=int, default=30)
    p.add_argument("--top-k-per-node", type=int, default=4)
    p.add_argument(
        "--threshold-mode",
        type=str,
        default="fixed",
        choices=["fixed", "val_opt"],
        help="fixed: use --threshold-default. val_opt: select threshold on validation fold.",
    )
    p.add_argument(
        "--threshold-default",
        type=float,
        default=0.5,
        help="Used when threshold-mode=fixed.",
    )
    p.add_argument(
        "--threshold-metric",
        type=str,
        default="balanced_accuracy",
        choices=["balanced_accuracy", "f1", "accuracy"],
        help="Used when threshold-mode=val_opt.",
    )
    p.add_argument("--threshold-grid-step", type=float, default=0.01)
    p.add_argument("--threshold-grid-low", type=float, default=0.05)
    p.add_argument("--threshold-grid-high", type=float, default=0.95)

    p.add_argument(
        "--out-path",
        type=Path,
        default=Path("outputs/metrics/runs/spatiotemporal_cv10/gnn_dualgraph_multiband_spatiotemporal_cv10.json"),
    )
    p.add_argument("--experiment-name", type=str, default="gnn_dualgraph_multiband_spatiotemporal_cv10")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    items = build_subject_sequences(args)
    y_all = np.array([int(x["label"]) for x in items], dtype=np.int64)
    idx_all = np.arange(len(items))

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

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        model = DualGraphMultiBandSpatioTemporalModel(
            in_dim=in_dim,
            hidden_dim=args.hidden_dim,
            temporal_hidden=args.temporal_hidden,
            dropout=args.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_state = None
        best_val_bal = -1.0
        for _ in range(args.epochs):
            _ = train_one_epoch(model, train_loader, optimizer, device)
            y_val, p_val, _ = predict(model, val_loader, device)
            val_bal = float(balanced_accuracy_score(y_val, (p_val >= 0.5).astype(np.int64)))
            if val_bal > best_val_bal:
                best_val_bal = val_bal
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        y_val_best, p_val_best, _ = predict(model, val_loader, device)
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

        y_test, p_test, w_test = predict(model, test_loader, device)
        metrics = calc_metrics(y_test, p_test, threshold=test_thr)

        w_mean = w_test.mean(axis=0)
        fold_results.append(
            {
                "fold": fold_no,
                "n_train": int(len(train_ds)),
                "n_val": int(len(val_ds)),
                "n_test": int(len(test_ds)),
                "best_val_balanced_accuracy": best_val_bal,
                "test_threshold": float(test_thr),
                "val_threshold_metric": args.threshold_metric if args.threshold_mode == "val_opt" else None,
                "val_threshold_score": None if args.threshold_mode != "val_opt" else float(val_thr_score),
                "avg_windows_test": float(np.mean([int(x["n_windows"]) for x in test_items])),
                "weights_mean": {
                    "pcc": float(w_mean[0]),
                    "theta": float(w_mean[1]),
                    "alpha": float(w_mean[2]),
                    "beta": float(w_mean[3]),
                },
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
            f"w=[{w_mean[0]:.3f},{w_mean[1]:.3f},{w_mean[2]:.3f},{w_mean[3]:.3f}]"
        )

    metric_names = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
    aggregate: dict[str, dict[str, float]] = {}
    for mn in metric_names:
        vals = np.array([fr["test_metrics"][mn] for fr in fold_results], dtype=np.float64)
        aggregate[mn] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}

    weight_agg: dict[str, dict[str, float]] = {}
    for wk in ["pcc", "theta", "alpha", "beta"]:
        vals = np.array([fr["weights_mean"][wk] for fr in fold_results], dtype=np.float64)
        weight_agg[wk] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}

    payload = {
        "schema_version": "eeg_mdd_benchmark_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "DualGraphMultiBandSpatioTemporalModel",
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
            "pcc_quantile": args.pcc_quantile,
            "min_edges": args.min_edges,
            "top_k_per_node": args.top_k_per_node,
            "threshold_mode": args.threshold_mode,
            "threshold_default": args.threshold_default,
            "threshold_metric": args.threshold_metric,
            "threshold_grid_step": args.threshold_grid_step,
            "threshold_grid_low": args.threshold_grid_low,
            "threshold_grid_high": args.threshold_grid_high,
        },
        "split_csv": str(args.split_csv),
        "dataset_window_summary": {
            "n_subjects": int(len(items)),
            "avg_windows": float(np.mean([int(x["n_windows"]) for x in items])),
            "min_windows": int(np.min([int(x["n_windows"]) for x in items])),
            "max_windows": int(np.max([int(x["n_windows"]) for x in items])),
        },
        "aggregate": aggregate,
        "fusion_weight_aggregate": weight_agg,
        "fold_results": fold_results,
    }

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.out_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
