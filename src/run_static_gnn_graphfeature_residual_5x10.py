from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool

from feature_model_utils import (
    aggregate_fold_results,
    build_static_graph_feature_table,
    calc_metrics,
    parse_seed_list,
    summarize_run_payloads,
    write_json,
)


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


class GraphFeatureResidualGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        aux_dim: int,
        hidden_dim: int = 32,
        aux_hidden: int = 128,
        dropout: float = 0.2,
        aux_scale: float = 0.5,
    ) -> None:
        super().__init__()
        self.aux_scale = float(aux_scale)
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
        self.aux_proj = nn.Sequential(
            nn.Linear(aux_dim, aux_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(aux_hidden, hidden_dim),
            nn.Tanh(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, pcc: Data, theta: Data, alpha: Data, beta: Data, aux: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_pcc = self.enc_pcc(pcc)
        h_theta = self.enc_theta(theta)
        h_alpha = self.enc_alpha(alpha)
        h_beta = self.enc_beta(beta)
        h_cat = torch.cat([h_pcc, h_theta, h_alpha, h_beta], dim=1)
        w = F.softmax(self.gate(h_cat), dim=1)
        h_stack = torch.stack([h_pcc, h_theta, h_alpha, h_beta], dim=1)
        h_gnn = (w.unsqueeze(-1) * h_stack).sum(dim=1)
        h_aux = self.aux_proj(aux)
        h_res = h_gnn + (self.aux_scale * h_aux)
        logits = self.fusion(torch.cat([h_gnn, h_res], dim=1))
        return logits, w


def load_graph_npz(npz_path: Path) -> Data:
    d = np.load(npz_path, allow_pickle=True)
    x = torch.tensor(d["x"], dtype=torch.float32)
    edge_index = torch.tensor(d["edge_index"], dtype=torch.long)
    edge_weight = torch.tensor(d["edge_weight"], dtype=torch.float32)
    y = torch.tensor(d["y"], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)


class MultiBandFeatureDataset(Dataset):
    def __init__(self, rows: list[dict[str, object]], aux_features: np.ndarray) -> None:
        self.items: list[dict[str, object]] = []
        for idx, r in enumerate(rows):
            self.items.append(
                {
                    "subject_id": str(r["subject_id"]),
                    "label": int(r["label"]),
                    "aux": np.asarray(aux_features[idx], dtype=np.float32),
                    "pcc": load_graph_npz(Path(str(r["pcc_path"]))),
                    "theta": load_graph_npz(Path(str(r["theta_path"]))),
                    "alpha": load_graph_npz(Path(str(r["alpha_path"]))),
                    "beta": load_graph_npz(Path(str(r["beta_path"]))),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.items[idx]


def collate_multiband(batch: list[dict[str, object]]) -> dict[str, object]:
    return {
        "pcc": Batch.from_data_list([x["pcc"] for x in batch]),
        "theta": Batch.from_data_list([x["theta"] for x in batch]),
        "alpha": Batch.from_data_list([x["alpha"] for x in batch]),
        "beta": Batch.from_data_list([x["beta"] for x in batch]),
        "aux": torch.tensor(np.stack([x["aux"] for x in batch]), dtype=torch.float32),
        "y": torch.tensor([int(x["label"]) for x in batch], dtype=torch.long),
        "subject_ids": [str(x["subject_id"]) for x in batch],
    }


def build_master_table(
    pcc_manifest: Path,
    theta_manifest: Path,
    alpha_manifest: Path,
    beta_manifest: Path,
) -> pd.DataFrame:
    pcc = pd.read_csv(pcc_manifest)[["subject_id", "label", "graph_path"]].rename(columns={"graph_path": "pcc_path"})
    theta = pd.read_csv(theta_manifest)[["subject_id", "graph_path"]].rename(columns={"graph_path": "theta_path"})
    alpha = pd.read_csv(alpha_manifest)[["subject_id", "graph_path"]].rename(columns={"graph_path": "alpha_path"})
    beta = pd.read_csv(beta_manifest)[["subject_id", "graph_path"]].rename(columns={"graph_path": "beta_path"})
    return pcc.merge(theta, on="subject_id").merge(alpha, on="subject_id").merge(beta, on="subject_id").sort_values("subject_id").reset_index(drop=True)


def standardize_features(train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((train_x - mean) / std).astype(np.float32), ((val_x - mean) / std).astype(np.float32), ((test_x - mean) / std).astype(np.float32)


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
        theta = batch["theta"].to(device)
        alpha = batch["alpha"].to(device)
        beta = batch["beta"].to(device)
        aux = batch["aux"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad()
        logits, _ = model(pcc, theta, alpha, beta, aux)
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
        theta = batch["theta"].to(device)
        alpha = batch["alpha"].to(device)
        beta = batch["beta"].to(device)
        aux = batch["aux"].to(device)
        y = batch["y"].to(device)
        logits, w = model(pcc, theta, alpha, beta, aux)
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs_all.append(probs)
        y_all.append(y.detach().cpu().numpy())
        w_all.append(w.detach().cpu().numpy())
    return np.concatenate(y_all), np.concatenate(probs_all), np.concatenate(w_all, axis=0)


def select_threshold(
    y_val: np.ndarray,
    prob_val: np.ndarray,
    thr_grid: np.ndarray,
) -> tuple[float, float]:
    best_thr = 0.5
    best_score = -1.0
    for thr in thr_grid:
        pred = (prob_val >= thr).astype(np.int64)
        score = float((pred == y_val).mean())
        if (score > best_score) or (abs(score - best_score) <= 1e-12 and abs(thr - 0.5) < abs(best_thr - 0.5)):
            best_score = score
            best_thr = float(thr)
    return best_thr, best_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repeated 5x10 CV for static GNN with residual graph-feature fusion.")
    p.add_argument("--pcc-manifest", type=Path, default=Path("data/processed/graphs_pcc_topk_task/manifest.csv"))
    p.add_argument("--theta-manifest", type=Path, default=Path("data/processed/graphs_plv_theta_topk_task/manifest.csv"))
    p.add_argument("--alpha-manifest", type=Path, default=Path("data/processed/graphs_plv_alpha_topk_task/manifest.csv"))
    p.add_argument("--beta-manifest", type=Path, default=Path("data/processed/graphs_plv_beta_topk_task/manifest.csv"))
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--aux-hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--aux-scale", type=float, default=0.5)
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--thr-low", type=float, default=0.3)
    p.add_argument("--thr-high", type=float, default=0.7)
    p.add_argument("--thr-step", type=float, default=0.01)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/metrics/runs/static_gnn_graphfeature_residual_5x10"))
    p.add_argument("--experiment-name", type=str, default="static_gnn_graphfeature_residual_5x10")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    master = build_master_table(
        pcc_manifest=args.pcc_manifest,
        theta_manifest=args.theta_manifest,
        alpha_manifest=args.alpha_manifest,
        beta_manifest=args.beta_manifest,
    )
    meta, X_aux = build_static_graph_feature_table(
        pcc_manifest=args.pcc_manifest,
        theta_manifest=args.theta_manifest,
        alpha_manifest=args.alpha_manifest,
        beta_manifest=args.beta_manifest,
    )
    if master["subject_id"].tolist() != meta["subject_id"].tolist():
        raise ValueError("Subject order mismatch between graph table and graphvector features.")

    y = master["label"].to_numpy(dtype=np.int64)
    subjects = master["subject_id"].to_numpy()
    thr_grid = np.arange(float(args.thr_low), float(args.thr_high) + (float(args.thr_step) * 0.5), float(args.thr_step), dtype=np.float64)

    run_payloads: list[dict[str, object]] = []
    for seed in seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        fold_results: list[dict[str, object]] = []

        for fold_no, (trainval_idx, test_idx) in enumerate(skf.split(np.arange(len(master)), y), start=1):
            trainval_df = master.iloc[trainval_idx].reset_index(drop=True)
            test_df = master.iloc[test_idx].reset_index(drop=True)
            X_trainval_aux = X_aux[trainval_idx]
            X_test_aux = X_aux[test_idx]
            y_trainval = y[trainval_idx]

            tr_idx, val_idx = train_test_split(
                np.arange(len(trainval_df)),
                test_size=float(args.val_size),
                random_state=seed + fold_no,
                stratify=y_trainval,
            )
            train_df = trainval_df.iloc[tr_idx].reset_index(drop=True)
            val_df = trainval_df.iloc[val_idx].reset_index(drop=True)
            X_train_aux = X_trainval_aux[tr_idx]
            X_val_aux = X_trainval_aux[val_idx]
            X_train_aux, X_val_aux, X_test_aux_std = standardize_features(X_train_aux, X_val_aux, X_test_aux)

            train_ds = MultiBandFeatureDataset(train_df.to_dict(orient="records"), X_train_aux)
            val_ds = MultiBandFeatureDataset(val_df.to_dict(orient="records"), X_val_aux)
            test_ds = MultiBandFeatureDataset(test_df.to_dict(orient="records"), X_test_aux_std)
            in_dim = int(train_ds[0]["pcc"].x.shape[1])
            aux_dim = int(X_train_aux.shape[1])

            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_multiband)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multiband)
            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multiband)

            model = GraphFeatureResidualGNN(
                in_dim=in_dim,
                aux_dim=aux_dim,
                hidden_dim=args.hidden_dim,
                aux_hidden=args.aux_hidden,
                dropout=args.dropout,
                aux_scale=args.aux_scale,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            best_state = None
            best_val_acc = -1.0
            best_epoch = 1
            for epoch_no in range(1, args.epochs + 1):
                _ = train_one_epoch(model, train_loader, optimizer, device)
                y_val, p_val, _ = predict_with_weights(model, val_loader, device)
                val_acc = float(((p_val >= 0.5).astype(np.int64) == y_val).mean())
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = int(epoch_no)
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if best_state is not None:
                model.load_state_dict(best_state)

            y_val, p_val, _ = predict_with_weights(model, val_loader, device)
            y_test, p_test, w_test = predict_with_weights(model, test_loader, device)
            thr_sel, val_thr_acc = select_threshold(y_val=y_val, prob_val=p_val, thr_grid=thr_grid)
            metrics = calc_metrics(y_test, p_test, threshold=thr_sel)
            w_mean = w_test.mean(axis=0)

            fold_results.append(
                {
                    "fold": fold_no,
                    "n_train": int(len(train_ds)),
                    "n_val": int(len(val_ds)),
                    "n_test": int(len(test_ds)),
                    "best_val_accuracy": float(best_val_acc),
                    "best_epoch": int(best_epoch),
                    "test_threshold": float(thr_sel),
                    "val_threshold_accuracy": float(val_thr_acc),
                    "test_subject_ids": subjects[test_idx].tolist(),
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
                f"seed={seed} fold={fold_no:02d} "
                f"acc={metrics['accuracy']:.4f} "
                f"bal_acc={metrics['balanced_accuracy']:.4f} "
                f"f1={metrics['f1']:.4f} "
                f"auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'NA'} "
                f"thr={thr_sel:.2f}"
            )

        payload = {
            "schema_version": "eeg_mdd_static_gnn_graphfeature_residual_v1",
            "experiment_name": args.experiment_name,
            "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model": "GraphFeatureResidualGNN",
            "params": {
                "folds": args.folds,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "hidden_dim": int(args.hidden_dim),
                "aux_hidden": int(args.aux_hidden),
                "dropout": float(args.dropout),
                "aux_scale": float(args.aux_scale),
                "seed": int(seed),
                "seeds": seeds,
                "thr_low": float(args.thr_low),
                "thr_high": float(args.thr_high),
                "thr_step": float(args.thr_step),
                "val_size": float(args.val_size),
            },
            "manifests": {
                "pcc": str(args.pcc_manifest),
                "theta": str(args.theta_manifest),
                "alpha": str(args.alpha_manifest),
                "beta": str(args.beta_manifest),
            },
            "feature_shape": {
                "n_subjects": int(X_aux.shape[0]),
                "n_features": int(X_aux.shape[1]),
            },
            "aggregate": aggregate_fold_results(fold_results),
            "fold_results": fold_results,
        }
        out_path = args.out_dir / f"static_gnn_graphfeature_residual_seed{seed}_cv10.json"
        write_json(out_path, payload)
        run_payloads.append({**payload, "_out_path": str(out_path)})
        print(f"saved: {out_path}")

    summary = summarize_run_payloads(run_payloads)
    summary_payload = {
        "schema_version": "eeg_mdd_static_gnn_graphfeature_residual_summary_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "GraphFeatureResidualGNN",
        "params": {
            "folds": args.folds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "hidden_dim": int(args.hidden_dim),
            "aux_hidden": int(args.aux_hidden),
            "dropout": float(args.dropout),
            "aux_scale": float(args.aux_scale),
            "seeds": seeds,
            "thr_low": float(args.thr_low),
            "thr_high": float(args.thr_high),
            "thr_step": float(args.thr_step),
            "val_size": float(args.val_size),
        },
        "manifests": {
            "pcc": str(args.pcc_manifest),
            "theta": str(args.theta_manifest),
            "alpha": str(args.alpha_manifest),
            "beta": str(args.beta_manifest),
        },
        "feature_shape": {
            "n_subjects": int(X_aux.shape[0]),
            "n_features": int(X_aux.shape[1]),
        },
        **summary,
    }
    summary_path = args.out_dir / "summary_5x10.json"
    write_json(summary_path, summary_payload)
    print(f"\nsummary: {summary_path}")
    print("done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
