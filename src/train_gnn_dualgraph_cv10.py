from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

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
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool


class SignedGCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1_pos = GCNConv(in_dim, hidden_dim)
        self.conv1_neg = GCNConv(in_dim, hidden_dim)
        self.conv2_pos = GCNConv(hidden_dim, hidden_dim)
        self.conv2_neg = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
        ew_pos = torch.clamp(edge_weight, min=0.0)
        ew_neg = torch.clamp(-edge_weight, min=0.0)

        x = self.conv1_pos(x, edge_index, edge_weight=ew_pos) - self.conv1_neg(x, edge_index, edge_weight=ew_neg)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2_pos(x, edge_index, edge_weight=ew_pos) - self.conv2_neg(x, edge_index, edge_weight=ew_neg)
        x = F.relu(x)
        return global_mean_pool(x, batch)


class DualGraphFusionModel(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.enc_pcc = SignedGCNEncoder(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.enc_plv = SignedGCNEncoder(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, pcc: Data, plv: Data) -> tuple[torch.Tensor, torch.Tensor]:
        h_pcc = self.enc_pcc(pcc)
        h_plv = self.enc_plv(plv)
        h_cat = torch.cat([h_pcc, h_plv], dim=1)
        w = F.softmax(self.gate(h_cat), dim=1)
        h = (w[:, 0:1] * h_pcc) + (w[:, 1:2] * h_plv)
        h = self.dropout(h)
        logits = self.classifier(h)
        return logits, w


def load_graph_npz(npz_path: Path) -> Data:
    d = np.load(npz_path, allow_pickle=True)
    x = torch.tensor(d["x"], dtype=torch.float32)
    edge_index = torch.tensor(d["edge_index"], dtype=torch.long)
    edge_weight = torch.tensor(d["edge_weight"], dtype=torch.float32)
    y = torch.tensor(d["y"], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)


class DualGraphDataset(Dataset):
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.items: list[dict[str, object]] = []
        for r in rows:
            self.items.append(
                {
                    "subject_id": str(r["subject_id"]),
                    "label": int(r["label"]),
                    "pcc": load_graph_npz(Path(str(r["pcc_path"]))),
                    "plv": load_graph_npz(Path(str(r["plv_path"]))),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.items[idx]


def collate_dual_graph(batch: list[dict[str, object]]) -> dict[str, object]:
    pcc_batch = Batch.from_data_list([x["pcc"] for x in batch])
    plv_batch = Batch.from_data_list([x["plv"] for x in batch])
    y = torch.tensor([int(x["label"]) for x in batch], dtype=torch.long)
    subject_ids = [str(x["subject_id"]) for x in batch]
    return {
        "pcc": pcc_batch,
        "plv": plv_batch,
        "y": y,
        "subject_ids": subject_ids,
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
        pcc = batch["pcc"].to(device)
        plv = batch["plv"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()
        logits, _ = model(pcc, plv)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

        n = int(y.shape[0])
        total_loss += float(loss.item()) * n
        total_n += n
    return total_loss / max(total_n, 1)


@torch.no_grad()
def predict_with_weights(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    w_all: list[np.ndarray] = []
    for batch in loader:
        pcc = batch["pcc"].to(device)
        plv = batch["plv"].to(device)
        y = batch["y"].to(device)
        logits, w = model(pcc, plv)
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs_all.append(probs)
        y_all.append(y.detach().cpu().numpy())
        w_all.append(w.detach().cpu().numpy())
    return np.concatenate(y_all), np.concatenate(probs_all), np.concatenate(w_all, axis=0)


def build_master_table(pcc_manifest: Path, plv_manifest: Path) -> pd.DataFrame:
    pcc = pd.read_csv(pcc_manifest)[["subject_id", "label", "graph_path"]].rename(columns={"graph_path": "pcc_path"})
    plv = pd.read_csv(plv_manifest)[["subject_id", "graph_path"]].rename(columns={"graph_path": "plv_path"})
    df = pcc.merge(plv, on="subject_id")
    return df.sort_values("subject_id").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="10-fold CV for dual-graph (PCC + PLV) fusion GNN")
    p.add_argument("--pcc-manifest", type=Path, default=Path("data/processed/graphs_pcc_task/manifest.csv"))
    p.add_argument("--plv-manifest", type=Path, default=Path("data/processed/graphs_plv_broad_task/manifest.csv"))
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-path", type=Path, default=Path("outputs/metrics/gnn_dualgraph_pcc_plv_cv10_metrics.json"))
    p.add_argument("--experiment-name", type=str, default="gnn_dualgraph_pcc_plv_cv10")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    master = build_master_table(args.pcc_manifest, args.plv_manifest)
    y_all = master["label"].to_numpy()
    idx_all = np.arange(len(master))

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_results: list[dict[str, object]] = []
    for fold_no, (trainval_idx, test_idx) in enumerate(skf.split(idx_all, y_all), start=1):
        trainval_df = master.iloc[trainval_idx].reset_index(drop=True)
        test_df = master.iloc[test_idx].reset_index(drop=True)

        tr_idx, val_idx = train_test_split(
            np.arange(len(trainval_df)),
            test_size=0.2,
            random_state=args.seed + fold_no,
            stratify=trainval_df["label"].to_numpy(),
        )
        train_df = trainval_df.iloc[tr_idx].reset_index(drop=True)
        val_df = trainval_df.iloc[val_idx].reset_index(drop=True)

        train_ds = DualGraphDataset(train_df.to_dict(orient="records"))
        val_ds = DualGraphDataset(val_df.to_dict(orient="records"))
        test_ds = DualGraphDataset(test_df.to_dict(orient="records"))

        in_dim = int(train_ds[0]["pcc"].x.shape[1])
        model = DualGraphFusionModel(in_dim=in_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_dual_graph
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_dual_graph
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_dual_graph
        )

        best_state = None
        best_val_bal = -1.0
        for _ in range(args.epochs):
            _ = train_one_epoch(model=model, loader=train_loader, optimizer=optimizer, device=device)
            y_val, p_val, _ = predict_with_weights(model, val_loader, device)
            val_bal = float(balanced_accuracy_score(y_val, (p_val >= 0.5).astype(np.int64)))
            if val_bal > best_val_bal:
                best_val_bal = val_bal
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        y_test, p_test, w_test = predict_with_weights(model, test_loader, device)
        metrics = calc_metrics(y_test, p_test, threshold=0.5)

        w_mean = w_test.mean(axis=0)
        w_mdd = w_test[y_test == 1].mean(axis=0) if np.any(y_test == 1) else np.full((2,), np.nan)
        w_h = w_test[y_test == 0].mean(axis=0) if np.any(y_test == 0) else np.full((2,), np.nan)

        fold_result = {
            "fold": fold_no,
            "n_train": int(len(train_ds)),
            "n_val": int(len(val_ds)),
            "n_test": int(len(test_ds)),
            "best_val_balanced_accuracy": best_val_bal,
            "test_metrics": metrics,
            "weights_mean": {"pcc": float(w_mean[0]), "plv": float(w_mean[1])},
            "weights_mdd_mean": {"pcc": float(w_mdd[0]), "plv": float(w_mdd[1])},
            "weights_h_mean": {"pcc": float(w_h[0]), "plv": float(w_h[1])},
        }
        fold_results.append(fold_result)
        print(
            f"fold={fold_no:02d} "
            f"acc={metrics['accuracy']:.4f} "
            f"bal_acc={metrics['balanced_accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} "
            f"auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'NA'} "
            f"w=[{w_mean[0]:.3f},{w_mean[1]:.3f}]"
        )

    metric_names = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
    aggregate: dict[str, dict[str, float]] = {}
    for mn in metric_names:
        vals = np.array([fr["test_metrics"][mn] for fr in fold_results], dtype=np.float64)
        aggregate[mn] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}

    weight_agg: dict[str, dict[str, float]] = {}
    for wk in ["pcc", "plv"]:
        vals = np.array([fr["weights_mean"][wk] for fr in fold_results], dtype=np.float64)
        weight_agg[wk] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}

    payload = {
        "schema_version": "eeg_mdd_benchmark_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "DualGraphFusionModel",
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
        },
        "manifests": {
            "pcc": str(args.pcc_manifest),
            "plv": str(args.plv_manifest),
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
