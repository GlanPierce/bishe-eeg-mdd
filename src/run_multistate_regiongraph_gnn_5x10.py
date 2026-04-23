from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, GCNConv

import run_multistate_explicit_region_temporal_node_gnn_5x10 as msmod
from feature_model_utils import aggregate_fold_results, calc_metrics, parse_seed_list, summarize_run_payloads, write_json

PRIMARY_REGION_KEYS = ["frontal", "central", "temporal", "parietal", "occipital"]


def upper_to_matrix(vec: np.ndarray, n_nodes: int) -> np.ndarray:
    tri = np.triu_indices(n_nodes, k=1)
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    adj[tri] = vec.astype(np.float32)
    adj[(tri[1], tri[0])] = vec.astype(np.float32)
    return adj


def unpack_static_vec(static_vec: np.ndarray, n_channels: int = 22, n_bands: int = 5) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    node_dim = n_channels * n_bands
    tri_len = (n_channels * (n_channels - 1)) // 2
    node_x = np.asarray(static_vec[:node_dim], dtype=np.float32).reshape(n_channels, n_bands)
    mats: dict[str, np.ndarray] = {}
    start = node_dim
    for key in ["pcc", "theta", "alpha", "beta"]:
        mats[key] = upper_to_matrix(np.asarray(static_vec[start : start + tri_len], dtype=np.float32), n_channels)
        start += tri_len
    return node_x, mats


def primary_region_indices(ch_names: list[str]) -> dict[str, list[int]]:
    out = {k: [] for k in PRIMARY_REGION_KEYS}
    for idx, name in enumerate(ch_names):
        primary = msmod._channel_descriptor(name)["primary"]
        if primary in out:
            out[str(primary)].append(int(idx))
    return out


def pair_mean(mat: np.ndarray, idx_a: list[int], idx_b: list[int], absolute: bool = False) -> float:
    if not idx_a or not idx_b:
        return 0.0
    if absolute:
        mat = np.abs(mat)
    if idx_a == idx_b:
        if len(idx_a) < 2:
            return 0.0
        block = mat[np.ix_(idx_a, idx_a)]
        tri = np.triu_indices(len(idx_a), k=1)
        vals = block[tri]
    else:
        vals = mat[np.ix_(idx_a, idx_b)].reshape(-1)
    if vals.size == 0:
        return 0.0
    return float(np.mean(vals))


def region_band_block(block_flat: np.ndarray, region_map: dict[str, list[int]]) -> dict[str, np.ndarray]:
    node_block = np.asarray(block_flat, dtype=np.float32).reshape(len(msmod.DEFAULT_CH_NAMES), 5)
    out: dict[str, np.ndarray] = {}
    for region in PRIMARY_REGION_KEYS:
        idxs = region_map[region]
        if idxs:
            out[region] = node_block[idxs].mean(axis=0).astype(np.float32)
        else:
            out[region] = np.zeros((5,), dtype=np.float32)
    return out


def region_connectivity_summary(mats: dict[str, np.ndarray], region_map: dict[str, list[int]], region: str) -> np.ndarray:
    idxs = region_map[region]
    other = sorted({x for key, vals in region_map.items() if key != region for x in vals})
    return np.asarray(
        [
            pair_mean(mats["pcc"], idxs, other, absolute=True),
            pair_mean(mats["theta"], idxs, other),
            pair_mean(mats["alpha"], idxs, other),
            pair_mean(mats["beta"], idxs, other),
        ],
        dtype=np.float32,
    )


def within_state_edge_attr(mats: dict[str, np.ndarray], region_map: dict[str, list[int]], region_a: str, region_b: str) -> np.ndarray:
    idx_a = region_map[region_a]
    idx_b = region_map[region_b]
    return np.asarray(
        [
            pair_mean(mats["pcc"], idx_a, idx_b, absolute=True),
            pair_mean(mats["theta"], idx_a, idx_b),
            pair_mean(mats["alpha"], idx_a, idx_b),
            pair_mean(mats["beta"], idx_a, idx_b),
        ],
        dtype=np.float32,
    )


def load_subject_item_map(
    items_cache_path: Path,
    states: list[str],
    eligible_states: list[str],
) -> tuple[list[str], np.ndarray, dict[str, dict[str, dict[str, object]]]]:
    dummy_args = argparse.Namespace(
        raw_dir=Path("data/raw/mdd-patients-eeg-dataset"),
        cache_states=",".join(states),
        max_seconds=120,
        window_seconds=8.0,
        step_seconds=8.0,
        max_windows=12,
        top_k_per_node=4,
        items_cache_path=items_cache_path,
    )
    items = msmod.load_or_build_feature_cache(dummy_args)
    by_subject: dict[str, dict[str, dict[str, object]]] = {}
    labels: dict[str, int] = {}
    for item in items:
        subject_id = str(item["subject_id"])
        state = str(item["state"])
        by_subject.setdefault(subject_id, {})[state] = item
        labels[subject_id] = int(item["label"])
    subject_ids = sorted(
        [
            subject_id
            for subject_id, state_map in by_subject.items()
            if all(state in state_map for state in eligible_states) and all(state in state_map for state in states)
        ]
    )
    y = np.asarray([labels[sid] for sid in subject_ids], dtype=np.int64)
    return subject_ids, y, by_subject


def build_subject_graph(subject_id: str, state_items: dict[str, dict[str, object]], states: list[str]) -> Data:
    region_map = primary_region_indices(msmod.DEFAULT_CH_NAMES)
    n_regions = len(PRIMARY_REGION_KEYS)
    n_states = len(states)
    n_state_nodes = n_states
    global_idx = (n_states * n_regions) + n_state_nodes
    x_rows: list[np.ndarray] = []
    src: list[int] = []
    dst: list[int] = []
    edge_attr: list[np.ndarray] = []

    state_node_feats: list[np.ndarray] = []
    for state_idx, state in enumerate(states):
        item = state_items[state]
        node_x, mats = unpack_static_vec(np.asarray(item["static_vec"], dtype=np.float32))
        temporal = np.asarray(item["temporal_node"], dtype=np.float32)
        temporal_mean = temporal[:110]
        temporal_std = temporal[110:220]
        temporal_delta = temporal[220:330]
        ratio_feats = np.concatenate(
            [
                np.asarray(item["static_ratio"], dtype=np.float32),
                np.asarray(item["temporal_ratio"], dtype=np.float32),
            ]
        ).astype(np.float32)

        static_by_region = region_band_block(node_x.reshape(-1), region_map)
        temporal_mean_by_region = region_band_block(temporal_mean, region_map)
        temporal_std_by_region = region_band_block(temporal_std, region_map)
        temporal_delta_by_region = region_band_block(temporal_delta, region_map)

        region_feats: list[np.ndarray] = []
        for region in PRIMARY_REGION_KEYS:
            reg_feat = np.concatenate(
                [
                    static_by_region[region],
                    temporal_mean_by_region[region],
                    temporal_std_by_region[region],
                    temporal_delta_by_region[region],
                    region_connectivity_summary(mats, region_map, region),
                    ratio_feats,
                ]
            ).astype(np.float32)
            region_feats.append(reg_feat)
            x_rows.append(reg_feat)

        state_feat = np.stack(region_feats).mean(axis=0).astype(np.float32)
        state_node_feats.append(state_feat)

        for region_i in range(n_regions):
            node_i = (state_idx * n_regions) + region_i
            state_node = (n_states * n_regions) + state_idx
            src.extend([node_i, state_node, node_i, global_idx])
            dst.extend([state_node, node_i, global_idx, node_i])
            edge_attr.extend(
                [
                    np.zeros((4,), dtype=np.float32),
                    np.zeros((4,), dtype=np.float32),
                    np.zeros((4,), dtype=np.float32),
                    np.zeros((4,), dtype=np.float32),
                ]
            )

        for i, region_a in enumerate(PRIMARY_REGION_KEYS):
            for j in range(i + 1, n_regions):
                region_b = PRIMARY_REGION_KEYS[j]
                w = within_state_edge_attr(mats, region_map, region_a, region_b)
                node_i = (state_idx * n_regions) + i
                node_j = (state_idx * n_regions) + j
                src.extend([node_i, node_j])
                dst.extend([node_j, node_i])
                edge_attr.extend([w, w])

    for state_feat in state_node_feats:
        x_rows.append(state_feat)

    for region_i in range(n_regions):
        for state_a in range(n_states):
            for state_b in range(state_a + 1, n_states):
                node_a = (state_a * n_regions) + region_i
                node_b = (state_b * n_regions) + region_i
                w = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                src.extend([node_a, node_b])
                dst.extend([node_b, node_a])
                edge_attr.extend([w, w])

    global_feat = np.stack(state_node_feats).mean(axis=0).astype(np.float32)
    x_rows.append(global_feat)
    for state_idx in range(n_states):
        state_node = (n_states * n_regions) + state_idx
        src.extend([state_node, global_idx])
        dst.extend([global_idx, state_node])
        edge_attr.extend([np.zeros((4,), dtype=np.float32), np.zeros((4,), dtype=np.float32)])

    return Data(
        x=torch.tensor(np.stack(x_rows).astype(np.float32), dtype=torch.float32),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(np.stack(edge_attr).astype(np.float32), dtype=torch.float32),
        y=torch.tensor([int(state_items[states[0]]["label"])], dtype=torch.long),
        subject_id=str(subject_id),
    )


def standardize_graphs(graphs: list[Data], train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> tuple[list[Data], list[Data], list[Data]]:
    train_x = np.stack([graphs[i].x.numpy() for i in train_idx]).astype(np.float32)
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)

    def clone_graph(idx: int) -> Data:
        g = graphs[idx]
        x = ((g.x.numpy() - mean[0]) / std[0]).astype(np.float32)
        return Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=g.edge_index.clone(),
            edge_attr=g.edge_attr.clone(),
            y=g.y.clone(),
            subject_id=str(g.subject_id),
        )

    return [clone_graph(i) for i in train_idx], [clone_graph(i) for i in val_idx], [clone_graph(i) for i in test_idx]


def collate_graphs(batch: list[Data]) -> Batch:
    return Batch.from_data_list(batch)


class MultiStateRegionGraphGNN(nn.Module):
    def __init__(self, n_nodes: int, input_dim: int, hidden_dim: int = 48, dropout: float = 0.25, conv_type: str = "gatv2", heads: int = 2) -> None:
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.value_proj = nn.Linear(input_dim, hidden_dim)
        self.node_embedding = nn.Embedding(self.n_nodes, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.conv_type = str(conv_type)
        if self.conv_type == "gatv2":
            self.conv1 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=4, dropout=dropout)
            self.conv2 = GATv2Conv(hidden_dim, hidden_dim, heads=heads, concat=False, edge_dim=4, dropout=dropout)
        elif self.conv_type == "gcn":
            self.edge_proj = nn.Linear(4, 1)
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported conv_type: {conv_type}")
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, data: Batch) -> torch.Tensor:
        node_id = torch.arange(self.n_nodes, device=data.x.device).repeat(int(data.num_graphs))
        h = self.value_proj(data.x) + self.node_embedding(node_id)
        h = F.relu(h)
        h = self.dropout(h)
        if self.conv_type == "gatv2":
            h = self.conv1(h, data.edge_index, edge_attr=data.edge_attr)
        else:
            ew = torch.sigmoid(self.edge_proj(data.edge_attr)).view(-1)
            h = self.conv1(h, data.edge_index, edge_weight=ew)
        h = F.relu(h)
        h = self.dropout(h)
        if self.conv_type == "gatv2":
            h = self.conv2(h, data.edge_index, edge_attr=data.edge_attr)
        else:
            ew = torch.sigmoid(self.edge_proj(data.edge_attr)).view(-1)
            h = self.conv2(h, data.edge_index, edge_weight=ew)
        h = F.relu(h)

        batch_size = int(data.y.shape[0])
        nodes = h.view(batch_size, self.n_nodes, -1)
        global_h = nodes[:, -1, :]
        state_h = nodes[:, -4:-1, :].mean(dim=1)
        region_h = nodes[:, :-4, :].mean(dim=1)
        pooled = torch.cat([global_h, state_h, region_h], dim=1)
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
        total_loss += float(loss.item()) * int(y.shape[0])
        total_n += int(y.shape[0])
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
        if (acc > best_acc) or (abs(acc - best_acc) <= 1e-12 and abs(float(thr) - 0.5) < abs(best_thr - 0.5)):
            best_acc = acc
            best_thr = float(thr)
    return best_thr, best_acc


def validation_score(metric: str, y_val: np.ndarray, prob_val: np.ndarray, low: float, high: float, step: float) -> float:
    if str(metric) == "accuracy":
        return float(accuracy_score(y_val, (prob_val >= 0.5).astype(np.int64)))
    if str(metric) == "threshold_accuracy":
        _, acc = select_threshold(y_val, prob_val, low, high, step)
        return float(acc)
    if str(metric) == "auc":
        return float(roc_auc_score(y_val, prob_val)) if len(np.unique(y_val)) == 2 else 0.5
    raise ValueError(f"Unsupported selection metric: {metric}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repeated 5x10 CV for multistate region-graph pure GNN.")
    p.add_argument("--items-cache-path", type=Path, default=Path("outputs/cache/multistate_task_ec_eo_features_ms120_w8p0_s8p0_mw12_topk4_v2.pt"))
    p.add_argument("--states", type=str, default="TASK,EC,EO")
    p.add_argument("--eligible-states", type=str, default="TASK,EC,EO")
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--epochs", type=int, default=220)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=2e-4)
    p.add_argument("--hidden-dim", type=int, default=48)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--conv-type", type=str, default="gatv2", choices=["gatv2", "gcn"])
    p.add_argument("--heads", type=int, default=2)
    p.add_argument("--patience", type=int, default=28)
    p.add_argument("--min-epochs", type=int, default=50)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--selection-metric", type=str, default="threshold_accuracy", choices=["accuracy", "threshold_accuracy", "auc"])
    p.add_argument("--thr-low", type=float, default=0.1)
    p.add_argument("--thr-high", type=float, default=0.9)
    p.add_argument("--thr-step", type=float, default=0.01)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out-dir", type=Path, default=Path("outputs/metrics/runs/multistate_regiongraph_gnn_5x10"))
    p.add_argument("--experiment-name", type=str, default="multistate_regiongraph_gnn_5x10")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    states = [x.strip().upper() for x in str(args.states).split(",") if x.strip()]
    eligible_states = [x.strip().upper() for x in str(args.eligible_states).split(",") if x.strip()]

    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    subject_ids, y_all, by_subject = load_subject_item_map(args.items_cache_path, states, eligible_states)
    graphs = [build_subject_graph(subject_id=sid, state_items=by_subject[sid], states=states) for sid in subject_ids]
    n_nodes = int(graphs[0].x.shape[0])
    input_dim = int(graphs[0].x.shape[1])

    run_payloads: list[dict[str, object]] = []
    for seed in seeds:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=int(seed))
        fold_results: list[dict[str, object]] = []

        for fold_no, (trainval_idx, test_idx) in enumerate(skf.split(np.zeros((len(graphs), 1), dtype=np.float32), y_all), start=1):
            y_trainval = y_all[trainval_idx]
            tr_rel_idx, val_rel_idx = train_test_split(
                np.arange(len(trainval_idx)),
                test_size=float(args.val_size),
                random_state=int(seed) + fold_no,
                stratify=y_trainval,
            )
            train_idx = trainval_idx[tr_rel_idx]
            val_idx = trainval_idx[val_rel_idx]
            train_graphs, val_graphs, test_graphs = standardize_graphs(graphs, train_idx, val_idx, test_idx)
            train_loader = torch.utils.data.DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True, collate_fn=collate_graphs)
            val_loader = torch.utils.data.DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)
            test_loader = torch.utils.data.DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False, collate_fn=collate_graphs)

            model = MultiStateRegionGraphGNN(
                n_nodes=n_nodes,
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                conv_type=args.conv_type,
                heads=args.heads,
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            best_state = None
            best_epoch = 1
            best_val_score = -1.0
            stale = 0
            for epoch in range(1, args.epochs + 1):
                _ = train_one_epoch(model, train_loader, optimizer, device)
                y_val, p_val = predict(model, val_loader, device)
                val_score = validation_score(args.selection_metric, y_val, p_val, args.thr_low, args.thr_high, args.thr_step)
                if val_score > best_val_score:
                    best_val_score = float(val_score)
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
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                    "n_test": int(len(test_idx)),
                    "best_epoch": int(best_epoch),
                    "best_val_score": float(best_val_score),
                    "selected_threshold": float(thr_sel),
                    "val_threshold_accuracy": float(val_thr_acc),
                    "test_subject_ids": [subject_ids[i] for i in test_idx],
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
            "schema_version": "eeg_mdd_multistate_regiongraph_gnn_5x10_v1",
            "experiment_name": args.experiment_name,
            "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model": "MultiStateRegionGraphGNN",
            "device": str(device),
            "params": {
                "folds": int(args.folds),
                "seed": int(seed),
                "seeds": [int(x) for x in seeds],
                "states": states,
                "eligible_states": eligible_states,
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "hidden_dim": int(args.hidden_dim),
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
            },
            "feature_shape": {
                "n_subjects": int(len(graphs)),
                "n_nodes": int(n_nodes),
                "input_dim": int(input_dim),
            },
            "aggregate": aggregate_fold_results(fold_results),
            "fold_results": fold_results,
        }
        out_path = args.out_dir / f"multistate_regiongraph_gnn_seed{seed}_cv10.json"
        write_json(out_path, payload)
        run_payloads.append({**payload, "_out_path": str(out_path)})
        print(f"saved: {out_path}")

    summary = summarize_run_payloads(run_payloads)
    summary_payload = {
        "schema_version": "eeg_mdd_multistate_regiongraph_gnn_5x10_summary_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "MultiStateRegionGraphGNN",
        "params": {
            "folds": int(args.folds),
            "seeds": [int(x) for x in seeds],
            "states": states,
            "eligible_states": eligible_states,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "hidden_dim": int(args.hidden_dim),
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
        },
        "feature_shape": {
            "n_subjects": int(len(graphs)),
            "n_nodes": int(n_nodes),
            "input_dim": int(input_dim),
        },
        **summary,
    }
    summary_path = args.out_dir / "summary_5x10.json"
    write_json(summary_path, summary_payload)
    print(f"\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
