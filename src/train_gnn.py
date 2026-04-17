from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


class GCNClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, dropout: float = 0.2) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        return self.fc(x)


def load_graph_npz(npz_path: Path) -> Data:
    d = np.load(npz_path, allow_pickle=True)
    x = torch.tensor(d["x"], dtype=torch.float32)
    edge_index = torch.tensor(d["edge_index"], dtype=torch.long)
    # GCN normalization assumes non-negative edge weights.
    edge_weight = torch.tensor(np.abs(d["edge_weight"]) + 1e-6, dtype=torch.float32)
    y = torch.tensor(d["y"], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = loss_fn(logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs
    return total_loss / max(total_graphs, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        labels = batch.y.detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)
    p = np.concatenate(all_probs) if all_probs else np.array([])
    y = np.concatenate(all_labels) if all_labels else np.array([])
    return y, p


def eval_metrics(y_true: np.ndarray, prob_pos: np.ndarray) -> dict[str, object]:
    pred = (prob_pos >= 0.5).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, prob_pos)) if len(np.unique(y_true)) == 2 else None,
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "classification_report": classification_report(
            y_true, pred, target_names=["H", "MDD"], output_dict=True, zero_division=0
        ),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train first GNN model from constructed PCC graphs")
    p.add_argument("--manifest", type=Path, default=Path("data/processed/graphs_pcc_task/manifest.csv"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    manifest = pd.read_csv(args.manifest)
    train_df = manifest[manifest["split"] == "train"].copy()
    test_df = manifest[manifest["split"] == "test"].copy()

    # make a validation slice from train split for model selection
    train_ids, val_ids = train_test_split(
        train_df["subject_id"].tolist(),
        test_size=0.2,
        random_state=args.seed,
        stratify=train_df["label"].tolist(),
    )
    train_set = train_df[train_df["subject_id"].isin(set(train_ids))]
    val_set = train_df[train_df["subject_id"].isin(set(val_ids))]

    train_graphs = [load_graph_npz(Path(p)) for p in train_set["graph_path"].tolist()]
    val_graphs = [load_graph_npz(Path(p)) for p in val_set["graph_path"].tolist()]
    test_graphs = [load_graph_npz(Path(p)) for p in test_df["graph_path"].tolist()]

    in_dim = int(train_graphs[0].x.shape[1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCNClassifier(in_dim=in_dim, hidden_dim=args.hidden_dim, dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    best_state = None
    best_val_bal_acc = -1.0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        y_val, p_val = predict(model, val_loader, device)
        val_metrics = eval_metrics(y_val, p_val)
        val_bal_acc = float(val_metrics["balanced_accuracy"])
        history.append({"epoch": epoch, "train_loss": train_loss, "val_balanced_accuracy": val_bal_acc})
        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0 or epoch == 1:
            print(f"epoch={epoch:03d} loss={train_loss:.4f} val_bal_acc={val_bal_acc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    y_test, p_test = predict(model, test_loader, device)
    test_metrics = eval_metrics(y_test, p_test)

    out_dir = Path("outputs/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "gnn_gcn_task_metrics.json"
    payload = {
        "model": "GCNClassifier",
        "device": str(device),
        "params": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "seed": args.seed,
        },
        "split_counts": {
            "train": int(len(train_graphs)),
            "val": int(len(val_graphs)),
            "test": int(len(test_graphs)),
        },
        "best_val_balanced_accuracy": best_val_bal_acc,
        "test_metrics": test_metrics,
        "history": history,
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {metrics_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
