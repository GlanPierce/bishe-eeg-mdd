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
from torch_geometric.data import Batch
from torch_geometric.nn import GATv2Conv, GCNConv

import run_explicit_region_temporal_dynamics_node_gnn_5x10 as dynmod
from feature_model_utils import aggregate_fold_results, calc_metrics, parse_seed_list, summarize_run_payloads, write_json


def standardize_features(train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return ((train_x - mean) / std).astype(np.float32), ((val_x - mean) / std).astype(np.float32), ((test_x - mean) / std).astype(np.float32)


class DynamicsGraphConvGNN(nn.Module):
    def __init__(
        self,
        n_feature_nodes: int,
        n_region_nodes: int,
        n_group_nodes: int,
        hidden_dim: int = 48,
        node_embed_dim: int = 24,
        dropout: float = 0.3,
        conv_type: str = "gatv2",
        heads: int = 2,
    ) -> None:
        super().__init__()
        self.n_feature_nodes = int(n_feature_nodes)
        self.n_region_nodes = int(n_region_nodes)
        self.n_group_nodes = int(n_group_nodes)
        self.n_nodes = self.n_feature_nodes + self.n_region_nodes + self.n_group_nodes + 1
        self.conv_type = str(conv_type)
        self.value_proj = nn.Linear(1, hidden_dim)
        self.node_embedding = nn.Embedding(self.n_nodes, node_embed_dim)
        self.id_proj = nn.Linear(node_embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        if self.conv_type == "gatv2":
            self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=1, dropout=dropout)
            self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=1, dropout=dropout)
        elif self.conv_type == "gcn":
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, data: Batch) -> torch.Tensor:
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
        feature_h = nodes[:, : self.n_feature_nodes, :]
        region_start = self.n_feature_nodes
        region_stop = region_start + self.n_region_nodes
        group_start = region_stop
        group_stop = group_start + self.n_group_nodes
        region_h = nodes[:, region_start:region_stop, :]
        group_h = nodes[:, group_start:group_stop, :]
        global_h = nodes[:, -1, :]
        pooled = torch.cat(
            [
                global_h,
                feature_h.mean(dim=1),
                region_h.mean(dim=1),
                group_h.mean(dim=1),
            ],
            dim=1,
        )
        return self.classifier(pooled)


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
    if str(metric) == "accuracy":
        return float(accuracy_score(y_val, (prob_val >= 0.5).astype(np.int64)))
    if str(metric) == "threshold_accuracy":
        _, acc = select_threshold(y_val, prob_val, low=low, high=high, step=step)
        return float(acc)
    if str(metric) == "auc":
        if len(np.unique(y_val)) < 2:
            return 0.5
        return float(roc_auc_score(y_val, prob_val))
    if str(metric) == "neg_log_loss":
        p = np.clip(prob_val, 1e-6, 1.0 - 1e-6)
        return -float(log_loss(y_val, p, labels=[0, 1]))
    raise ValueError(f"Unsupported selection metric: {metric}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repeated 5x10 CV for explicit region-temporal dynamics graph-conv GNN.")
    p.add_argument("--split-csv", type=Path, default=Path("data/splits/task_split_subject_level.csv"))
    p.add_argument("--pcc-manifest", type=Path, default=Path("data/processed/graphs_pcc_topk_task/manifest.csv"))
    p.add_argument("--theta-manifest", type=Path, default=Path("data/processed/graphs_plv_theta_topk_task/manifest.csv"))
    p.add_argument("--alpha-manifest", type=Path, default=Path("data/processed/graphs_plv_alpha_topk_task/manifest.csv"))
    p.add_argument("--beta-manifest", type=Path, default=Path("data/processed/graphs_plv_beta_topk_task/manifest.csv"))
    p.add_argument("--items-cache-path", type=Path, default=Path("outputs/cache/spatiotemporal_items/ms120_w8p0_s8p0_mw12_q0p8_me30_k4.pt"))
    p.add_argument("--max-seconds", type=int, default=120)
    p.add_argument("--window-seconds", type=float, default=8.0)
    p.add_argument("--step-seconds", type=float, default=8.0)
    p.add_argument("--max-windows", type=int, default=12)
    p.add_argument("--pcc-quantile", type=float, default=0.8)
    p.add_argument("--min-edges", type=int, default=30)
    p.add_argument("--top-k-per-node", type=int, default=4)
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--hidden-dim", type=int, default=48)
    p.add_argument("--node-embed-dim", type=int, default=24)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--conv-type", type=str, default="gatv2", choices=["gatv2", "gcn"])
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--patience", type=int, default=24)
    p.add_argument("--min-epochs", type=int, default=40)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--selection-metric", type=str, default="threshold_accuracy", choices=["accuracy", "threshold_accuracy", "auc", "neg_log_loss"])
    p.add_argument("--thr-low", type=float, default=0.1)
    p.add_argument("--thr-high", type=float, default=0.9)
    p.add_argument("--thr-step", type=float, default=0.01)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out-dir", type=Path, default=Path("outputs/metrics/runs/explicit_region_temporal_dynamics_graphconv_5x10"))
    p.add_argument("--experiment-name", type=str, default="explicit_region_temporal_dynamics_graphconv_5x10")
    return p.parse_args()


def build_expanded_table(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int, torch.Tensor, torch.Tensor]:
    static_meta, static_x = dynmod.build_static_graph_feature_table(
        pcc_manifest=args.pcc_manifest,
        theta_manifest=args.theta_manifest,
        alpha_manifest=args.alpha_manifest,
        beta_manifest=args.beta_manifest,
    )
    static_ratio_x = dynmod.build_static_ratio_table(static_x, dynmod.DEFAULT_CH_NAMES)
    temporal_items = dynmod.load_or_build_temporal_items(args)
    temporal_subject_ids, temporal_labels, temporal_node_x, temporal_conn_x, temporal_ratio_x = dynmod.build_temporal_dynamic_tables(
        temporal_items,
        dynmod.DEFAULT_CH_NAMES,
    )
    static_subject_ids = static_meta["subject_id"].to_numpy()
    static_labels = static_meta["label"].to_numpy(dtype=np.int64)
    if list(static_subject_ids) != list(temporal_subject_ids):
        raise ValueError("Static and temporal subject orders do not match.")
    if not np.array_equal(static_labels, temporal_labels):
        raise ValueError("Static and temporal labels do not match.")

    feature_region_map: list[list[int]] = []
    feature_group_ids: list[int] = []
    blocks: list[np.ndarray] = []
    base_map = dynmod.base_feature_region_map(dynmod.DEFAULT_CH_NAMES)
    temporal_node_map = dynmod.node_feature_region_map(dynmod.DEFAULT_CH_NAMES)
    conn_map = dynmod.conn_window_feature_region_map()
    ratio_map = dynmod.ratio_feature_region_map()

    blocks.append(static_x.astype(np.float32))
    feature_region_map.extend(base_map)
    feature_group_ids.extend([dynmod.GROUP_KEYS.index("static_raw")] * int(static_x.shape[1]))

    for start, group_key in [(0, "temporal_node_mean"), (110, "temporal_node_std"), (220, "temporal_node_delta")]:
        blocks.append(temporal_node_x[:, start : start + 110].astype(np.float32))
        feature_region_map.extend(temporal_node_map)
        feature_group_ids.extend([dynmod.GROUP_KEYS.index(group_key)] * 110)

    for start, group_key in [(0, "temporal_conn_mean"), (75, "temporal_conn_std"), (150, "temporal_conn_delta")]:
        blocks.append(temporal_conn_x[:, start : start + 75].astype(np.float32))
        feature_region_map.extend(conn_map)
        feature_group_ids.extend([dynmod.GROUP_KEYS.index(group_key)] * 75)

    blocks.append(static_ratio_x.astype(np.float32))
    feature_region_map.extend(ratio_map)
    feature_group_ids.extend([dynmod.GROUP_KEYS.index("static_ratio")] * 10)

    for start, group_key in [(0, "temporal_ratio_mean"), (10, "temporal_ratio_std"), (20, "temporal_ratio_delta")]:
        blocks.append(temporal_ratio_x[:, start : start + 10].astype(np.float32))
        feature_region_map.extend(ratio_map)
        feature_group_ids.extend([dynmod.GROUP_KEYS.index(group_key)] * 10)

    base_x = np.concatenate(blocks, axis=1).astype(np.float32)
    region_x = dynmod.build_region_aggregates(base_x, feature_region_map)
    group_x = dynmod.build_group_aggregates(base_x, feature_group_ids)
    expanded_x = np.concatenate([base_x, region_x, group_x], axis=1).astype(np.float32)
    n_feature_nodes = int(base_x.shape[1])
    n_region_nodes = int(region_x.shape[1])
    n_group_nodes = int(group_x.shape[1])
    edge_index, edge_weight = dynmod.build_graph_edges(feature_region_map, feature_group_ids, n_feature_nodes=n_feature_nodes)
    return expanded_x, static_labels, static_subject_ids, n_feature_nodes, n_region_nodes, n_group_nodes, edge_index, edge_weight


def main() -> int:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    expanded_x, y_all, subject_ids, n_feature_nodes, n_region_nodes, n_group_nodes, edge_index, edge_weight = build_expanded_table(args)
    run_payloads: list[dict[str, object]] = []

    for seed in seeds:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=int(seed))
        fold_results: list[dict[str, object]] = []

        for fold_no, (trainval_idx, test_idx) in enumerate(skf.split(expanded_x, y_all), start=1):
            y_trainval = y_all[trainval_idx]
            tr_rel_idx, val_rel_idx = train_test_split(
                np.arange(len(trainval_idx)),
                test_size=float(args.val_size),
                random_state=int(seed) + fold_no,
                stratify=y_trainval,
            )
            train_idx = trainval_idx[tr_rel_idx]
            val_idx = trainval_idx[val_rel_idx]
            x_train, x_val, x_test = standardize_features(expanded_x[train_idx], expanded_x[val_idx], expanded_x[test_idx])

            train_ds = dynmod.ExplicitRegionTemporalDynamicsDataset(x_train, y_all[train_idx], subject_ids[train_idx], edge_index=edge_index, edge_weight=edge_weight)
            val_ds = dynmod.ExplicitRegionTemporalDynamicsDataset(x_val, y_all[val_idx], subject_ids[val_idx], edge_index=edge_index, edge_weight=edge_weight)
            test_ds = dynmod.ExplicitRegionTemporalDynamicsDataset(x_test, y_all[test_idx], subject_ids[test_idx], edge_index=edge_index, edge_weight=edge_weight)
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=dynmod.collate_graphs)
            val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=dynmod.collate_graphs)
            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=dynmod.collate_graphs)

            model = DynamicsGraphConvGNN(
                n_feature_nodes=n_feature_nodes,
                n_region_nodes=n_region_nodes,
                n_group_nodes=n_group_nodes,
                hidden_dim=args.hidden_dim,
                node_embed_dim=args.node_embed_dim,
                dropout=args.dropout,
                conv_type=args.conv_type,
                heads=args.heads,
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            best_state = None
            best_epoch = 1
            best_val_score = -1.0
            best_val_tiebreak = -float("inf")
            stale = 0
            for epoch in range(1, args.epochs + 1):
                _ = train_one_epoch(model, train_loader, optimizer, device)
                y_val, p_val = predict(model, val_loader, device)
                val_score = validation_score(args.selection_metric, y_val, p_val, args.thr_low, args.thr_high, args.thr_step)
                val_tiebreak = validation_score("neg_log_loss", y_val, p_val, args.thr_low, args.thr_high, args.thr_step)
                if (val_score > best_val_score) or (abs(val_score - best_val_score) <= 1e-12 and val_tiebreak > best_val_tiebreak):
                    best_val_score = val_score
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
                    "selection_metric": args.selection_metric,
                    "best_val_score": float(best_val_score),
                    "val_threshold_accuracy": float(val_thr_acc),
                    "test_threshold": float(thr_sel),
                    "test_subject_ids": subject_ids[test_idx].tolist(),
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

        payload = {
            "schema_version": "eeg_mdd_explicit_region_temporal_dynamics_graphconv_5x10_v1",
            "experiment_name": args.experiment_name,
            "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model": "DynamicsGraphConvGNN",
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
                "conv_type": str(args.conv_type),
                "heads": int(args.heads),
                "patience": int(args.patience),
                "min_epochs": int(args.min_epochs),
                "val_size": float(args.val_size),
                "selection_metric": str(args.selection_metric),
                "thr_low": float(args.thr_low),
                "thr_high": float(args.thr_high),
                "thr_step": float(args.thr_step),
                "window_seconds": float(args.window_seconds),
                "step_seconds": float(args.step_seconds),
                "max_windows": int(args.max_windows),
                "max_seconds": int(args.max_seconds),
            },
            "feature_shape": {
                "n_subjects": int(expanded_x.shape[0]),
                "n_feature_nodes": int(n_feature_nodes),
                "n_region_nodes": int(n_region_nodes),
                "n_group_nodes": int(n_group_nodes),
                "n_total_nodes_without_global": int(expanded_x.shape[1]),
            },
            "aggregate": aggregate_fold_results(fold_results),
            "fold_results": fold_results,
        }
        out_path = args.out_dir / f"explicit_region_temporal_dynamics_graphconv_seed{seed}_cv10.json"
        write_json(out_path, payload)
        run_payloads.append({**payload, "_out_path": str(out_path)})
        print(f"saved: {out_path}")

    summary = summarize_run_payloads(run_payloads)
    summary_payload = {
        "schema_version": "eeg_mdd_explicit_region_temporal_dynamics_graphconv_5x10_summary_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "DynamicsGraphConvGNN",
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
            "conv_type": str(args.conv_type),
            "heads": int(args.heads),
            "patience": int(args.patience),
            "min_epochs": int(args.min_epochs),
            "val_size": float(args.val_size),
            "selection_metric": str(args.selection_metric),
            "thr_low": float(args.thr_low),
            "thr_high": float(args.thr_high),
            "thr_step": float(args.thr_step),
            "window_seconds": float(args.window_seconds),
            "step_seconds": float(args.step_seconds),
            "max_windows": int(args.max_windows),
            "max_seconds": int(args.max_seconds),
        },
        "feature_shape": {
            "n_subjects": int(expanded_x.shape[0]),
            "n_feature_nodes": int(n_feature_nodes),
            "n_region_nodes": int(n_region_nodes),
            "n_group_nodes": int(n_group_nodes),
            "n_total_nodes_without_global": int(expanded_x.shape[1]),
        },
        **summary,
    }
    summary_path = args.out_dir / "summary_5x10.json"
    write_json(summary_path, summary_payload)
    print(f"\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
