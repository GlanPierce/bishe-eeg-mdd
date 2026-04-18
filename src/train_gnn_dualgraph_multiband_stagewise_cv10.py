from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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


@dataclass(frozen=True)
class StageConfig:
    name: str
    use_reweight: bool
    use_spatial_prior: bool
    use_mask: bool


STAGES: dict[str, StageConfig] = {
    "baseline": StageConfig("baseline", use_reweight=False, use_spatial_prior=False, use_mask=False),
    "reweight": StageConfig("reweight", use_reweight=True, use_spatial_prior=False, use_mask=False),
    "prior_reweight": StageConfig("prior_reweight", use_reweight=True, use_spatial_prior=True, use_mask=False),
    "prior_reweight_mask": StageConfig("prior_reweight_mask", use_reweight=True, use_spatial_prior=True, use_mask=True),
}


# Approximate 2D electrode coordinates (old naming included).
ELECTRODE_POS: dict[str, tuple[float, float]] = {
    "FP1": (-0.6, 1.0),
    "FP2": (0.6, 1.0),
    "F7": (-1.0, 0.6),
    "F3": (-0.5, 0.55),
    "FZ": (0.0, 0.55),
    "F4": (0.5, 0.55),
    "F8": (1.0, 0.6),
    "T3": (-1.15, 0.0),
    "C3": (-0.5, 0.0),
    "CZ": (0.0, 0.0),
    "C4": (0.5, 0.0),
    "T4": (1.15, 0.0),
    "T5": (-1.0, -0.6),
    "P3": (-0.5, -0.55),
    "PZ": (0.0, -0.55),
    "P4": (0.5, -0.55),
    "T6": (1.0, -0.6),
    "O1": (-0.5, -1.0),
    "O2": (0.5, -1.0),
}


class EdgeAdaptation(nn.Module):
    def __init__(
        self,
        stage: StageConfig,
        prior_lambda: float,
        mask_tau: float,
    ) -> None:
        super().__init__()
        self.stage = stage
        self.prior_lambda = float(np.clip(prior_lambda, 0.0, 1.0))
        self.mask_tau = float(np.clip(mask_tau, 0.0, 1.0))

        # Learnable scalar gates to avoid overfitting on tiny dataset.
        self.rw_abs = nn.Parameter(torch.tensor(1.0))
        self.rw_sign = nn.Parameter(torch.tensor(0.0))
        self.rw_prior = nn.Parameter(torch.tensor(0.0))
        self.rw_bias = nn.Parameter(torch.tensor(0.0))

        self.mk_abs = nn.Parameter(torch.tensor(1.0))
        self.mk_prior = nn.Parameter(torch.tensor(0.0))
        self.mk_bias = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        edge_weight: torch.Tensor,
        edge_prior: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ew = edge_weight
        stats: dict[str, torch.Tensor] = {}

        if self.stage.use_spatial_prior:
            prior_scale = (1.0 - self.prior_lambda) + (self.prior_lambda * edge_prior)
            ew = ew * prior_scale
            stats["prior_mean"] = edge_prior.mean()

        if self.stage.use_reweight:
            rw_logit = (
                self.rw_abs * torch.abs(ew)
                + self.rw_sign * torch.sign(ew)
                + self.rw_prior * edge_prior
                + self.rw_bias
            )
            rw_gate = torch.sigmoid(rw_logit)
            # Scale in [0, 2], so it can suppress or amplify.
            ew = ew * (2.0 * rw_gate)
            stats["rw_gate_mean"] = rw_gate.mean()

        if self.stage.use_mask:
            mk_logit = self.mk_abs * torch.abs(ew) + self.mk_prior * edge_prior + self.mk_bias
            mk_prob = torch.sigmoid(mk_logit)
            if self.training:
                hard = (mk_prob >= self.mask_tau).float()
                mask = hard + (mk_prob - mk_prob.detach())
            else:
                mask = (mk_prob >= self.mask_tau).float()
            ew = ew * mask
            stats["mask_prob_mean"] = mk_prob.mean()
            stats["mask_keep_ratio"] = (mask > 0.0).float().mean()

        return ew, stats


class SignedGCNEncoderAdaptive(nn.Module):
    def __init__(
        self,
        in_dim: int,
        prior_matrix: torch.Tensor,
        stage: StageConfig,
        hidden_dim: int = 32,
        dropout: float = 0.2,
        prior_lambda: float = 0.35,
        mask_tau: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv1_pos = GCNConv(in_dim, hidden_dim)
        self.conv1_neg = GCNConv(in_dim, hidden_dim)
        self.conv2_pos = GCNConv(hidden_dim, hidden_dim)
        self.conv2_neg = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.adapt = EdgeAdaptation(stage=stage, prior_lambda=prior_lambda, mask_tau=mask_tau)
        self.register_buffer("prior_matrix", prior_matrix.float())

    def _edge_prior(self, edge_index: torch.Tensor) -> torch.Tensor:
        n = int(self.prior_matrix.shape[0])
        src_local = torch.remainder(edge_index[0], n)
        dst_local = torch.remainder(edge_index[1], n)
        return self.prior_matrix[src_local, dst_local]

    def forward(self, data: Data) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x, edge_index, edge_weight, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_prior = self._edge_prior(edge_index)
        ew, stats = self.adapt(edge_weight=edge_weight, edge_prior=edge_prior)

        ew_pos = torch.clamp(ew, min=0.0)
        ew_neg = torch.clamp(-ew, min=0.0)

        x = self.conv1_pos(x, edge_index, edge_weight=ew_pos) - self.conv1_neg(x, edge_index, edge_weight=ew_neg)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2_pos(x, edge_index, edge_weight=ew_pos) - self.conv2_neg(x, edge_index, edge_weight=ew_neg)
        x = F.relu(x)
        return global_mean_pool(x, batch), stats


class DualGraphMultiBandAdaptiveModel(nn.Module):
    def __init__(
        self,
        in_dim: int,
        prior_matrix: torch.Tensor,
        stage: StageConfig,
        hidden_dim: int = 32,
        dropout: float = 0.2,
        prior_lambda: float = 0.35,
        mask_tau: float = 0.5,
    ) -> None:
        super().__init__()
        kwargs = {
            "in_dim": in_dim,
            "prior_matrix": prior_matrix,
            "stage": stage,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "prior_lambda": prior_lambda,
            "mask_tau": mask_tau,
        }
        self.enc_pcc = SignedGCNEncoderAdaptive(**kwargs)
        self.enc_theta = SignedGCNEncoderAdaptive(**kwargs)
        self.enc_alpha = SignedGCNEncoderAdaptive(**kwargs)
        self.enc_beta = SignedGCNEncoderAdaptive(**kwargs)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, pcc: Data, theta: Data, alpha: Data, beta: Data) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        h_pcc, s_pcc = self.enc_pcc(pcc)
        h_theta, s_theta = self.enc_theta(theta)
        h_alpha, s_alpha = self.enc_alpha(alpha)
        h_beta, s_beta = self.enc_beta(beta)

        h_cat = torch.cat([h_pcc, h_theta, h_alpha, h_beta], dim=1)
        w = F.softmax(self.gate(h_cat), dim=1)
        h_stack = torch.stack([h_pcc, h_theta, h_alpha, h_beta], dim=1)
        h = (w.unsqueeze(-1) * h_stack).sum(dim=1)
        h = self.dropout(h)
        logits = self.classifier(h)

        aux = {}
        all_stats = [s_pcc, s_theta, s_alpha, s_beta]
        keys = sorted(set().union(*[set(s.keys()) for s in all_stats]))
        for k in keys:
            vals = [s[k] for s in all_stats if k in s]
            aux[k] = torch.stack(vals).mean()
        return logits, w, aux


def load_graph_npz(npz_path: Path) -> Data:
    d = np.load(npz_path, allow_pickle=True)
    x = torch.tensor(d["x"], dtype=torch.float32)
    edge_index = torch.tensor(d["edge_index"], dtype=torch.long)
    edge_weight = torch.tensor(d["edge_weight"], dtype=torch.float32)
    y = torch.tensor(d["y"], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)


def read_channel_names(npz_path: Path) -> list[str]:
    d = np.load(npz_path, allow_pickle=True)
    return [str(x) for x in d["ch_names"].tolist()]


def _norm_ch_name(name: str) -> str:
    s = name.upper().strip()
    if s.startswith("EEG "):
        s = s[4:]
    if s.endswith("-LE"):
        s = s[:-3]
    if s.endswith("-REF"):
        s = s[:-4]
    return s.strip()


def build_spatial_prior(channel_names: list[str], sigma: float = 0.8, unknown_prior: float = 0.85) -> np.ndarray:
    n = len(channel_names)
    xy: list[tuple[float, float] | None] = []
    for ch in channel_names:
        key = _norm_ch_name(ch)
        xy.append(ELECTRODE_POS.get(key))

    prior = np.ones((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if i == j:
                prior[i, j] = 1.0
                continue
            if xy[i] is None or xy[j] is None:
                prior[i, j] = float(unknown_prior)
                continue
            dx = xy[i][0] - xy[j][0]
            dy = xy[i][1] - xy[j][1]
            dist = float(np.sqrt((dx * dx) + (dy * dy)))
            prior[i, j] = float(np.exp(-((dist * dist) / (2.0 * sigma * sigma))))
    return prior


class MultiBandDataset(Dataset):
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.items: list[dict[str, object]] = []
        for r in rows:
            self.items.append(
                {
                    "subject_id": str(r["subject_id"]),
                    "label": int(r["label"]),
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
    pcc_batch = Batch.from_data_list([x["pcc"] for x in batch])
    theta_batch = Batch.from_data_list([x["theta"] for x in batch])
    alpha_batch = Batch.from_data_list([x["alpha"] for x in batch])
    beta_batch = Batch.from_data_list([x["beta"] for x in batch])
    y = torch.tensor([int(x["label"]) for x in batch], dtype=torch.long)
    subject_ids = [str(x["subject_id"]) for x in batch]
    return {
        "pcc": pcc_batch,
        "theta": theta_batch,
        "alpha": alpha_batch,
        "beta": beta_batch,
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
    df = pcc.merge(theta, on="subject_id").merge(alpha, on="subject_id").merge(beta, on="subject_id")
    return df.sort_values("subject_id").reset_index(drop=True)


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mask_sparsity_lambda: float,
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
        y = batch["y"].to(device)

        optimizer.zero_grad()
        logits, _, aux = model(pcc, theta, alpha, beta)
        loss = loss_fn(logits, y)
        if "mask_prob_mean" in aux and mask_sparsity_lambda > 0.0:
            loss = loss + (mask_sparsity_lambda * aux["mask_prob_mean"])
        loss.backward()
        optimizer.step()

        n = int(y.shape[0])
        total_loss += float(loss.item()) * n
        total_n += n
    return total_loss / max(total_n, 1)


@torch.no_grad()
def predict_with_stats(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    model.eval()
    probs_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []
    w_all: list[np.ndarray] = []
    stats_buf: dict[str, list[float]] = {}

    for batch in loader:
        pcc = batch["pcc"].to(device)
        theta = batch["theta"].to(device)
        alpha = batch["alpha"].to(device)
        beta = batch["beta"].to(device)
        y = batch["y"].to(device)
        logits, w, aux = model(pcc, theta, alpha, beta)
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        probs_all.append(probs)
        y_all.append(y.detach().cpu().numpy())
        w_all.append(w.detach().cpu().numpy())

        for k, v in aux.items():
            stats_buf.setdefault(k, []).append(float(v.detach().cpu().item()))

    stats_mean = {k: float(np.mean(vs)) for k, vs in stats_buf.items()}
    return np.concatenate(y_all), np.concatenate(probs_all), np.concatenate(w_all, axis=0), stats_mean


def run_single_stage(master: pd.DataFrame, stage: StageConfig, args: argparse.Namespace) -> dict[str, object]:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sample_pcc = Path(str(master.iloc[0]["pcc_path"]))
    ch_names = read_channel_names(sample_pcc)
    prior_np = build_spatial_prior(ch_names, sigma=args.prior_sigma, unknown_prior=args.unknown_prior)
    prior = torch.tensor(prior_np, dtype=torch.float32)

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

        train_ds = MultiBandDataset(train_df.to_dict(orient="records"))
        val_ds = MultiBandDataset(val_df.to_dict(orient="records"))
        test_ds = MultiBandDataset(test_df.to_dict(orient="records"))

        in_dim = int(train_ds[0]["pcc"].x.shape[1])
        model = DualGraphMultiBandAdaptiveModel(
            in_dim=in_dim,
            prior_matrix=prior,
            stage=stage,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            prior_lambda=args.prior_lambda,
            mask_tau=args.mask_tau,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_multiband
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multiband
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_multiband
        )

        best_state = None
        best_val_bal = -1.0
        for _ in range(args.epochs):
            _ = train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                mask_sparsity_lambda=args.mask_sparsity_lambda,
            )
            y_val, p_val, _, _ = predict_with_stats(model, val_loader, device)
            val_bal = float(balanced_accuracy_score(y_val, (p_val >= 0.5).astype(np.int64)))
            if val_bal > best_val_bal:
                best_val_bal = val_bal
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        y_test, p_test, w_test, aux_stats = predict_with_stats(model, test_loader, device)
        metrics = calc_metrics(y_test, p_test, threshold=0.5)

        w_mean = w_test.mean(axis=0)
        fold_result = {
            "fold": fold_no,
            "n_train": int(len(train_ds)),
            "n_val": int(len(val_ds)),
            "n_test": int(len(test_ds)),
            "best_val_balanced_accuracy": best_val_bal,
            "test_metrics": metrics,
            "weights_mean": {
                "pcc": float(w_mean[0]),
                "theta": float(w_mean[1]),
                "alpha": float(w_mean[2]),
                "beta": float(w_mean[3]),
            },
            "aux_stats": aux_stats,
        }
        fold_results.append(fold_result)

        aux_text = ""
        if aux_stats:
            aux_text = " " + " ".join([f"{k}={v:.3f}" for k, v in sorted(aux_stats.items())])
        print(
            f"[{stage.name}] fold={fold_no:02d} "
            f"acc={metrics['accuracy']:.4f} bal_acc={metrics['balanced_accuracy']:.4f} "
            f"f1={metrics['f1']:.4f} auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'NA'}"
            + aux_text
        )

    metric_names = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
    aggregate: dict[str, dict[str, float]] = {}
    for mn in metric_names:
        vals = np.array([fr["test_metrics"][mn] for fr in fold_results], dtype=np.float64)
        aggregate[mn] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}

    aux_aggregate: dict[str, dict[str, float]] = {}
    aux_keys = sorted(set().union(*[set(fr.get("aux_stats", {}).keys()) for fr in fold_results]))
    for k in aux_keys:
        vals = np.array([fr.get("aux_stats", {}).get(k, np.nan) for fr in fold_results], dtype=np.float64)
        aux_aggregate[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}

    return {
        "stage": stage.name,
        "stage_config": {
            "use_reweight": stage.use_reweight,
            "use_spatial_prior": stage.use_spatial_prior,
            "use_mask": stage.use_mask,
        },
        "aggregate": aggregate,
        "aux_aggregate": aux_aggregate,
        "fold_results": fold_results,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-wise CV: baseline -> reweight -> +prior -> +mask")
    p.add_argument("--pcc-manifest", type=Path, default=Path("data/processed/graphs_pcc_task/manifest.csv"))
    p.add_argument("--theta-manifest", type=Path, default=Path("data/processed/graphs_plv_theta_task/manifest.csv"))
    p.add_argument("--alpha-manifest", type=Path, default=Path("data/processed/graphs_plv_alpha_task/manifest.csv"))
    p.add_argument("--beta-manifest", type=Path, default=Path("data/processed/graphs_plv_beta_task/manifest.csv"))
    p.add_argument("--stages", nargs="+", default=["baseline", "reweight", "prior_reweight", "prior_reweight_mask"])
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prior-lambda", type=float, default=0.35)
    p.add_argument("--prior-sigma", type=float, default=0.8)
    p.add_argument("--unknown-prior", type=float, default=0.85)
    p.add_argument("--mask-tau", type=float, default=0.5)
    p.add_argument("--mask-sparsity-lambda", type=float, default=0.01)
    p.add_argument(
        "--out-path",
        type=Path,
        default=Path("outputs/metrics/runs/learnable_prior_mask_cv10/stagewise_prior_learnable_cv10.json"),
    )
    p.add_argument("--experiment-name", type=str, default="gnn_dualgraph_multiband_stagewise_prior_learnable_cv10")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    requested = []
    for s in args.stages:
        if s not in STAGES:
            raise ValueError(f"Unknown stage: {s}. valid={sorted(STAGES.keys())}")
        requested.append(STAGES[s])

    master = build_master_table(
        args.pcc_manifest,
        args.theta_manifest,
        args.alpha_manifest,
        args.beta_manifest,
    )

    stage_results: list[dict[str, object]] = []
    for stage in requested:
        print(f"\n=== Running stage: {stage.name} ===")
        result = run_single_stage(master=master, stage=stage, args=args)
        stage_results.append(result)

    payload = {
        "schema_version": "eeg_mdd_benchmark_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "params": {
            "folds": args.folds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "seed": args.seed,
            "prior_lambda": args.prior_lambda,
            "prior_sigma": args.prior_sigma,
            "unknown_prior": args.unknown_prior,
            "mask_tau": args.mask_tau,
            "mask_sparsity_lambda": args.mask_sparsity_lambda,
        },
        "manifests": {
            "pcc": str(args.pcc_manifest),
            "theta": str(args.theta_manifest),
            "alpha": str(args.alpha_manifest),
            "beta": str(args.beta_manifest),
        },
        "stage_results": stage_results,
    }

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {args.out_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
