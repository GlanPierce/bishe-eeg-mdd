from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, GCNConv

from feature_model_utils import (
    aggregate_fold_results,
    build_static_graph_feature_table,
    calc_metrics,
    parse_seed_list,
    summarize_run_payloads,
    write_json,
)


def standardize_features(train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((train_x - mean) / std).astype(np.float32), ((val_x - mean) / std).astype(np.float32), ((test_x - mean) / std).astype(np.float32)


def build_feature_graph_edges(n_features: int) -> tuple[torch.Tensor, torch.Tensor]:
    global_node = int(n_features)
    src: list[int] = []
    dst: list[int] = []
    weights: list[float] = []
    for idx in range(n_features):
        src.extend([idx, global_node])
        dst.extend([global_node, idx])
        weights.extend([1.0, 1.0])
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


class FeatureNodeDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, subject_ids: np.ndarray) -> None:
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.subject_ids = np.asarray(subject_ids)
        self.n_features = int(self.x.shape[1])
        self.edge_index, self.edge_weight = build_feature_graph_edges(self.n_features)
        self.node_id = torch.arange(self.n_features + 1, dtype=torch.long)

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


class FeatureNodeGNN(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 64,
        node_embed_dim: int = 32,
        dropout: float = 0.2,
        conv_type: str = "gatv2",
        heads: int = 2,
    ) -> None:
        super().__init__()
        self.n_nodes = int(n_features) + 1
        self.conv_type = str(conv_type)
        self.value_proj = nn.Linear(1, hidden_dim)
        self.node_embedding = nn.Embedding(self.n_nodes, node_embed_dim)
        self.id_proj = nn.Linear(node_embed_dim, hidden_dim)
        if self.conv_type == "weighted_star":
            self.feature_logits = nn.Embedding(int(n_features), 2)
            self.bias = nn.Parameter(torch.zeros(2))
            nn.init.zeros_(self.feature_logits.weight)
        elif self.conv_type == "gcn":
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        elif self.conv_type == "gatv2":
            self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=1, dropout=dropout)
            self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=1, dropout=dropout)
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")
        if self.conv_type != "weighted_star":
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )

    def forward(self, data: Batch) -> torch.Tensor:
        if self.conv_type == "weighted_star":
            batch_size = int(data.batch.max().item()) + 1 if data.batch.numel() else 0
            values = data.x.view(batch_size, self.n_nodes)[:, :-1]
            weights = self.feature_logits.weight
            return values @ weights + self.bias

        h = self.value_proj(data.x) + self.id_proj(self.node_embedding(data.node_id))
        h = F.relu(h)
        h = self.dropout(h)
        if self.conv_type == "gatv2":
            h = self.conv1(h, data.edge_index, edge_attr=data.edge_attr.view(-1, 1))
        else:
            h = self.conv1(h, data.edge_index, edge_weight=data.edge_attr)
        h = F.relu(h)
        h = self.dropout(h)
        if self.conv_type == "gatv2":
            h = self.conv2(h, data.edge_index, edge_attr=data.edge_attr.view(-1, 1))
        else:
            h = self.conv2(h, data.edge_index, edge_weight=data.edge_attr)
        h = F.relu(h)
        batch_size = int(data.batch.max().item()) + 1 if data.batch.numel() else 0
        nodes = h.view(batch_size, self.n_nodes, -1)
        global_node = nodes[:, -1, :]
        feature_mean = nodes[:, :-1, :].mean(dim=1)
        return self.classifier(torch.cat([global_node, feature_mean], dim=1))


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repeated 5x10 CV for pure GNN over static graph-derived feature nodes.")
    p.add_argument("--pcc-manifest", type=Path, default=Path("data/processed/graphs_pcc_topk_task/manifest.csv"))
    p.add_argument("--theta-manifest", type=Path, default=Path("data/processed/graphs_plv_theta_topk_task/manifest.csv"))
    p.add_argument("--alpha-manifest", type=Path, default=Path("data/processed/graphs_plv_alpha_topk_task/manifest.csv"))
    p.add_argument("--beta-manifest", type=Path, default=Path("data/processed/graphs_plv_beta_topk_task/manifest.csv"))
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--node-embed-dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--conv-type", type=str, default="weighted_star", choices=["weighted_star", "gatv2", "gcn"])
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--min-epochs", type=int, default=80)
    p.add_argument("--selection-metric", type=str, default="auc", choices=["accuracy", "threshold_accuracy", "auc", "neg_log_loss"])
    p.add_argument("--thr-low", type=float, default=0.2)
    p.add_argument("--thr-high", type=float, default=0.8)
    p.add_argument("--thr-step", type=float, default=0.01)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--fold-ids", type=str, default="")
    p.add_argument("--out-dir", type=Path, default=Path("outputs/metrics/runs/static_feature_node_gnn_5x10"))
    p.add_argument("--experiment-name", type=str, default="static_feature_node_gnn_5x10")
    return p.parse_args()


def parse_optional_fold_ids(text: str) -> set[int] | None:
    values = [x.strip() for x in str(text).split(",") if x.strip()]
    if not values:
        return None
    return {int(x) for x in values}


def main() -> int:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    fold_filter = parse_optional_fold_ids(args.fold_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta, x_all = build_static_graph_feature_table(
        pcc_manifest=args.pcc_manifest,
        theta_manifest=args.theta_manifest,
        alpha_manifest=args.alpha_manifest,
        beta_manifest=args.beta_manifest,
    )
    y_all = meta["label"].to_numpy(dtype=np.int64)
    subjects = meta["subject_id"].to_numpy()
    n_features = int(x_all.shape[1])
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
            train_ds = FeatureNodeDataset(x_train, y_all[train_idx], subjects[train_idx])
            val_ds = FeatureNodeDataset(x_val, y_all[val_abs_idx], subjects[val_abs_idx])
            test_ds = FeatureNodeDataset(x_test, y_all[test_idx], subjects[test_idx])
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_graphs)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)
            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)

            model = FeatureNodeGNN(
                n_features=n_features,
                hidden_dim=args.hidden_dim,
                node_embed_dim=args.node_embed_dim,
                dropout=args.dropout,
                conv_type=args.conv_type,
                heads=args.heads,
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            best_state = None
            best_epoch = 1
            best_val_acc = -1.0
            best_val_tiebreak = -float("inf")
            stale = 0
            for epoch in range(1, args.epochs + 1):
                _ = train_one_epoch(model, train_loader, optimizer, device)
                y_val, p_val = predict(model, val_loader, device)
                val_score = validation_score(args.selection_metric, y_val, p_val, args.thr_low, args.thr_high, args.thr_step)
                val_tiebreak = validation_score("neg_log_loss", y_val, p_val, args.thr_low, args.thr_high, args.thr_step)
                if (val_score > best_val_acc) or (abs(val_score - best_val_acc) <= 1e-12 and val_tiebreak > best_val_tiebreak):
                    best_val_acc = val_score
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
                    "best_val_accuracy": float(best_val_acc),
                    "selection_metric": args.selection_metric,
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
            "schema_version": "eeg_mdd_static_feature_node_gnn_5x10_v1",
            "experiment_name": args.experiment_name,
            "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model": "FeatureNodeGNN",
            "device": str(device),
            "params": {
                "folds": int(args.folds),
                "seed": int(seed),
                "seeds": [int(x) for x in seeds],
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "hidden_dim": int(args.hidden_dim),
                "node_embed_dim": int(args.node_embed_dim),
                "dropout": float(args.dropout),
                "conv_type": args.conv_type,
                "heads": int(args.heads),
                "patience": int(args.patience),
                "min_epochs": int(args.min_epochs),
                "selection_metric": args.selection_metric,
                "thr_low": float(args.thr_low),
                "thr_high": float(args.thr_high),
                "thr_step": float(args.thr_step),
                "val_size": float(args.val_size),
                "fold_ids": sorted(fold_filter or []),
            },
            "manifests": {
                "pcc": str(args.pcc_manifest),
                "theta": str(args.theta_manifest),
                "alpha": str(args.alpha_manifest),
                "beta": str(args.beta_manifest),
            },
            "feature_shape": {"n_subjects": int(x_all.shape[0]), "n_features": int(x_all.shape[1])},
            "aggregate": aggregate_fold_results(fold_results),
            "fold_results": fold_results,
        }
        out_path = args.out_dir / f"static_feature_node_gnn_seed{seed}_cv10.json"
        write_json(out_path, payload)
        run_payloads.append({**payload, "_out_path": str(out_path)})
        print(f"saved: {out_path}")

    summary = summarize_run_payloads(run_payloads)
    summary_payload = {
        "schema_version": "eeg_mdd_static_feature_node_gnn_5x10_summary_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "FeatureNodeGNN",
        "params": {
            "folds": int(args.folds),
            "seeds": [int(x) for x in seeds],
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "hidden_dim": int(args.hidden_dim),
            "node_embed_dim": int(args.node_embed_dim),
            "dropout": float(args.dropout),
            "conv_type": args.conv_type,
            "heads": int(args.heads),
            "patience": int(args.patience),
            "min_epochs": int(args.min_epochs),
            "selection_metric": args.selection_metric,
            "thr_low": float(args.thr_low),
            "thr_high": float(args.thr_high),
            "thr_step": float(args.thr_step),
            "val_size": float(args.val_size),
            "fold_ids": sorted(fold_filter or []),
        },
        "feature_shape": {"n_subjects": int(x_all.shape[0]), "n_features": int(x_all.shape[1])},
        **summary,
    }
    summary_path = args.out_dir / "summary_5x10.json"
    write_json(summary_path, summary_payload)
    print(f"\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
