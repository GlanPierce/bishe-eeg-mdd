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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv, global_mean_pool


REGION_KEYS = ["frontal", "central", "temporal", "parietal", "occipital", "left", "right", "midline", "global"]


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
        if ch.startswith(("C",)):
            out["central"].append(i)
        if ch.startswith(("T",)):
            out["temporal"].append(i)
        if ch.startswith(("P",)):
            out["parietal"].append(i)
        if ch.startswith(("O",)):
            out["occipital"].append(i)
        if ch.endswith(("1", "3", "5", "7")):
            out["left"].append(i)
        elif ch.endswith(("2", "4", "6", "8")):
            out["right"].append(i)
        else:
            out["midline"].append(i)
        out["global"].append(i)
    return out


def _adj_from_edges(n_nodes: int, edge_index: np.ndarray, edge_weight: np.ndarray) -> np.ndarray:
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for (src, dst), weight in zip(edge_index.T, edge_weight):
        adj[int(src), int(dst)] = float(weight)
    return adj


def load_region_prior_graph(npz_path: Path, append_adj_row: bool = True) -> Data:
    d = np.load(npz_path, allow_pickle=True)
    x_base = np.asarray(d["x"], dtype=np.float32)
    edge_index = np.asarray(d["edge_index"], dtype=np.int64)
    edge_weight = np.asarray(d["edge_weight"], dtype=np.float32)
    ch_names = [str(x) for x in d["ch_names"].tolist()]
    y = torch.tensor(d["y"], dtype=torch.long)

    n_elec = int(x_base.shape[0])
    adj = _adj_from_edges(n_elec, edge_index, edge_weight)
    regions = _region_indices(ch_names)
    region_feats: list[np.ndarray] = []
    for region in REGION_KEYS:
        idx = regions[region]
        if idx:
            feat = x_base[idx].mean(axis=0)
        else:
            feat = np.zeros((x_base.shape[1],), dtype=np.float32)
        region_feats.append(feat.astype(np.float32))

    x_all = np.concatenate([x_base, np.stack(region_feats, axis=0)], axis=0).astype(np.float32)
    n_total = int(x_all.shape[0])
    adj_ext = np.zeros((n_total, n_total), dtype=np.float32)
    adj_ext[:n_elec, :n_elec] = adj

    # Region-prior edges encode the report narrative: frontal/hemisphere/global coupling.
    virtual_edges: list[tuple[int, int, float]] = []
    for region_pos, region in enumerate(REGION_KEYS):
        region_node = n_elec + region_pos
        idx = regions[region]
        for elec_idx in idx:
            virtual_edges.append((elec_idx, region_node, 1.0))
            virtual_edges.append((region_node, elec_idx, 1.0))
        if region != "global":
            global_node = n_elec + REGION_KEYS.index("global")
            virtual_edges.append((region_node, global_node, 1.0))
            virtual_edges.append((global_node, region_node, 1.0))

    if virtual_edges:
        vi = np.array([[a, b] for a, b, _ in virtual_edges], dtype=np.int64).T
        vw = np.array([w for _, _, w in virtual_edges], dtype=np.float32)
        edge_index = np.concatenate([edge_index, vi], axis=1)
        edge_weight = np.concatenate([edge_weight, vw], axis=0)
        for a, b, w in virtual_edges:
            adj_ext[a, b] = w

    if append_adj_row:
        x_all = np.concatenate([x_all, adj_ext], axis=1).astype(np.float32)

    region_id = np.zeros((n_total, len(REGION_KEYS) + 1), dtype=np.float32)
    region_id[:n_elec, 0] = 1.0
    for i in range(len(REGION_KEYS)):
        region_id[n_elec + i, i + 1] = 1.0
    x_all = np.concatenate([x_all, region_id], axis=1).astype(np.float32)

    return Data(
        x=torch.tensor(x_all, dtype=torch.float32),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_weight, dtype=torch.float32),
        y=y,
    )


class MultiBandRegionDataset(Dataset):
    def __init__(self, rows: list[dict[str, object]], append_adj_row: bool = True) -> None:
        self.items: list[dict[str, object]] = []
        for r in rows:
            self.items.append(
                {
                    "subject_id": str(r["subject_id"]),
                    "label": int(r["label"]),
                    "pcc": load_region_prior_graph(Path(str(r["pcc_path"])), append_adj_row=append_adj_row),
                    "theta": load_region_prior_graph(Path(str(r["theta_path"])), append_adj_row=append_adj_row),
                    "alpha": load_region_prior_graph(Path(str(r["alpha_path"])), append_adj_row=append_adj_row),
                    "beta": load_region_prior_graph(Path(str(r["beta_path"])), append_adj_row=append_adj_row),
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
        "y": torch.tensor([int(x["label"]) for x in batch], dtype=torch.long),
        "subject_ids": [str(x["subject_id"]) for x in batch],
    }


class SignedGCNNodeEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv1_pos = GCNConv(in_dim, hidden_dim)
        self.conv1_neg = GCNConv(in_dim, hidden_dim)
        self.conv2_pos = GCNConv(hidden_dim, hidden_dim)
        self.conv2_neg = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data: Data) -> tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
        ew_pos = torch.clamp(edge_weight, min=0.0)
        ew_neg = torch.clamp(-edge_weight, min=0.0)
        x = self.conv1_pos(x, edge_index, edge_weight=ew_pos) - self.conv1_neg(x, edge_index, edge_weight=ew_neg)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2_pos(x, edge_index, edge_weight=ew_pos) - self.conv2_neg(x, edge_index, edge_weight=ew_neg)
        x = F.relu(x)
        return x, global_mean_pool(x, batch)


class RegionPriorFlatGNN(nn.Module):
    def __init__(self, in_dim: int, n_nodes: int, hidden_dim: int = 16, dropout: float = 0.2) -> None:
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.hidden_dim = int(hidden_dim)
        self.enc_pcc = SignedGCNNodeEncoder(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.enc_theta = SignedGCNNodeEncoder(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.enc_alpha = SignedGCNNodeEncoder(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.enc_beta = SignedGCNNodeEncoder(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        graph_dim = hidden_dim * 4
        flat_dim = n_nodes * hidden_dim * 4
        self.band_gate = nn.Sequential(
            nn.Linear(graph_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        self.classifier = nn.Sequential(
            nn.Linear(flat_dim + graph_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def _reshape_nodes(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        return x.view(batch_size, self.n_nodes, self.hidden_dim)

    def forward(self, pcc: Data, theta: Data, alpha: Data, beta: Data) -> tuple[torch.Tensor, torch.Tensor]:
        x_pcc, g_pcc = self.enc_pcc(pcc)
        x_theta, g_theta = self.enc_theta(theta)
        x_alpha, g_alpha = self.enc_alpha(alpha)
        x_beta, g_beta = self.enc_beta(beta)

        g_cat = torch.cat([g_pcc, g_theta, g_alpha, g_beta], dim=1)
        w = F.softmax(self.band_gate(g_cat), dim=1)
        x_pcc = self._reshape_nodes(x_pcc, pcc.batch)
        x_theta = self._reshape_nodes(x_theta, theta.batch)
        x_alpha = self._reshape_nodes(x_alpha, alpha.batch)
        x_beta = self._reshape_nodes(x_beta, beta.batch)
        node_stack = torch.stack([x_pcc, x_theta, x_alpha, x_beta], dim=1)
        weighted_nodes = node_stack * w.unsqueeze(-1).unsqueeze(-1)
        flat_weighted = weighted_nodes.reshape(weighted_nodes.shape[0], -1)
        logits = self.classifier(torch.cat([flat_weighted, g_cat], dim=1))
        return logits, w


def build_master_table(pcc_manifest: Path, theta_manifest: Path, alpha_manifest: Path, beta_manifest: Path) -> pd.DataFrame:
    pcc = pd.read_csv(pcc_manifest)[["subject_id", "label", "graph_path"]].rename(columns={"graph_path": "pcc_path"})
    theta = pd.read_csv(theta_manifest)[["subject_id", "graph_path"]].rename(columns={"graph_path": "theta_path"})
    alpha = pd.read_csv(alpha_manifest)[["subject_id", "graph_path"]].rename(columns={"graph_path": "alpha_path"})
    beta = pd.read_csv(beta_manifest)[["subject_id", "graph_path"]].rename(columns={"graph_path": "beta_path"})
    return pcc.merge(theta, on="subject_id").merge(alpha, on="subject_id").merge(beta, on="subject_id").sort_values("subject_id").reset_index(drop=True)


def calc_metrics(y_true: np.ndarray, prob_pos: np.ndarray, threshold: float) -> dict[str, object]:
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


def train_one_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_n = 0
    for batch in loader:
        pcc = batch["pcc"].to(device)
        theta = batch["theta"].to(device)
        alpha = batch["alpha"].to(device)
        beta = batch["beta"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad()
        logits, _ = model(pcc, theta, alpha, beta)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        n = int(y.shape[0])
        total_loss += float(loss.item()) * n
        total_n += n
    return total_loss / max(total_n, 1)


@torch.no_grad()
def predict(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    y_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    w_all: list[np.ndarray] = []
    for batch in loader:
        pcc = batch["pcc"].to(device)
        theta = batch["theta"].to(device)
        alpha = batch["alpha"].to(device)
        beta = batch["beta"].to(device)
        y = batch["y"].to(device)
        logits, w = model(pcc, theta, alpha, beta)
        p = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        y_all.append(y.detach().cpu().numpy())
        p_all.append(p)
        w_all.append(w.detach().cpu().numpy())
    return np.concatenate(y_all), np.concatenate(p_all), np.concatenate(w_all, axis=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="10-fold CV for region-prior flat-readout pure GNN")
    p.add_argument("--pcc-manifest", type=Path, default=Path("data/processed/graphs_pcc_topk_task/manifest.csv"))
    p.add_argument("--theta-manifest", type=Path, default=Path("data/processed/graphs_plv_theta_topk_task/manifest.csv"))
    p.add_argument("--alpha-manifest", type=Path, default=Path("data/processed/graphs_plv_alpha_topk_task/manifest.csv"))
    p.add_argument("--beta-manifest", type=Path, default=Path("data/processed/graphs_plv_beta_topk_task/manifest.csv"))
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--min-epochs", type=int, default=30)
    p.add_argument("--thr-low", type=float, default=0.3)
    p.add_argument("--thr-high", type=float, default=0.7)
    p.add_argument("--thr-step", type=float, default=0.01)
    p.add_argument("--no-adj-row", action="store_true")
    p.add_argument("--out-path", type=Path, default=Path("outputs/metrics/runs/regionprior_flatreadout_cv10/regionprior_flatreadout_cv10.json"))
    p.add_argument("--experiment-name", type=str, default="regionprior_flatreadout_cv10")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    master = build_master_table(args.pcc_manifest, args.theta_manifest, args.alpha_manifest, args.beta_manifest)
    y_all = master["label"].to_numpy(dtype=np.int64)
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

        append_adj_row = not bool(args.no_adj_row)
        train_ds = MultiBandRegionDataset(train_df.to_dict(orient="records"), append_adj_row=append_adj_row)
        val_ds = MultiBandRegionDataset(val_df.to_dict(orient="records"), append_adj_row=append_adj_row)
        test_ds = MultiBandRegionDataset(test_df.to_dict(orient="records"), append_adj_row=append_adj_row)
        sample_graph = train_ds[0]["pcc"]
        model = RegionPriorFlatGNN(
            in_dim=int(sample_graph.x.shape[1]),
            n_nodes=int(sample_graph.x.shape[0]),
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_multiband)
        val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multiband)
        test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multiband)

        best_state = None
        best_epoch = 1
        best_val_acc = -1.0
        stale = 0
        for epoch in range(1, args.epochs + 1):
            _ = train_one_epoch(model, train_loader, optimizer, device)
            y_val, p_val, _ = predict(model, val_loader, device)
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
        y_val, p_val, _ = predict(model, val_loader, device)
        thr_sel, val_thr_acc = select_threshold(y_val, p_val, low=args.thr_low, high=args.thr_high, step=args.thr_step)
        y_test, p_test, w_test = predict(model, test_loader, device)
        metrics = calc_metrics(y_test, p_test, threshold=thr_sel)
        w_mean = w_test.mean(axis=0)
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
            f"fold={fold_no:02d} acc={metrics['accuracy']:.4f} bal_acc={metrics['balanced_accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'NA'} "
            f"epoch={best_epoch} thr={thr_sel:.2f}"
        )

    aggregate: dict[str, dict[str, float]] = {}
    for metric_name in ["accuracy", "balanced_accuracy", "f1", "roc_auc"]:
        vals = np.array([fr["test_metrics"][metric_name] for fr in fold_results], dtype=np.float64)
        aggregate[metric_name] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}

    payload = {
        "schema_version": "eeg_mdd_regionprior_flatreadout_gnn_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "RegionPriorFlatGNN",
        "device": str(device),
        "params": {
            "folds": int(args.folds),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "seed": int(args.seed),
            "patience": int(args.patience),
            "min_epochs": int(args.min_epochs),
            "thr_low": float(args.thr_low),
            "thr_high": float(args.thr_high),
            "thr_step": float(args.thr_step),
            "append_adj_row": append_adj_row,
            "region_keys": REGION_KEYS,
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
