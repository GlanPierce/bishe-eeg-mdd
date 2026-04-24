from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data

from feature_model_utils import (
    aggregate_fold_results,
    build_clean_window_static_graph_feature_table,
    build_static_graph_feature_table,
    build_targeted_clean_temporal_summary_table,
    build_temporal_summary_table,
    calc_metrics,
    parse_seed_list,
    summarize_run_payloads,
    write_json,
)

REGION_KEYS = ["frontal", "central", "temporal", "parietal", "occipital", "left", "right", "midline", "cross_region", "global"]
DEFAULT_CH_NAMES = [
    "EEG Fp1-LE",
    "EEG F3-LE",
    "EEG C3-LE",
    "EEG P3-LE",
    "EEG O1-LE",
    "EEG F7-LE",
    "EEG T3-LE",
    "EEG T5-LE",
    "EEG Fz-LE",
    "EEG Fp2-LE",
    "EEG F4-LE",
    "EEG C4-LE",
    "EEG P4-LE",
    "EEG O2-LE",
    "EEG F8-LE",
    "EEG T4-LE",
    "EEG T6-LE",
    "EEG Cz-LE",
    "EEG Pz-LE",
    "EEG A2-A1",
    "EEG 23A-23R",
    "EEG 24A-24R",
]
TEMPORAL_GROUP_KEYS = ["temporal_mean", "temporal_std", "temporal_delta"]


def _norm_ch_name(name: str) -> str:
    s = str(name).upper().strip()
    if s.startswith("EEG "):
        s = s[4:]
    for suffix in ["-LE", "-REF"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s.strip()


def channel_regions(name: str) -> list[int]:
    ch = _norm_ch_name(name)
    regions: list[str] = []
    if ch.startswith(("FP", "F")):
        regions.append("frontal")
    elif ch.startswith("C"):
        regions.append("central")
    elif ch.startswith("T"):
        regions.append("temporal")
    elif ch.startswith("P"):
        regions.append("parietal")
    elif ch.startswith("O"):
        regions.append("occipital")
    if ch.endswith(("1", "3", "5", "7")):
        regions.append("left")
    elif ch.endswith(("2", "4", "6", "8")):
        regions.append("right")
    else:
        regions.append("midline")
    regions.append("global")
    return [REGION_KEYS.index(r) for r in dict.fromkeys(regions)]


def base_feature_region_map(ch_names: list[str], n_bands: int = 5) -> list[list[int]]:
    n_channels = len(ch_names)
    feature_regions: list[list[int]] = []
    channel_map = [channel_regions(ch) for ch in ch_names]

    for channel_idx in range(n_channels):
        for _ in range(n_bands):
            feature_regions.append(channel_map[channel_idx])

    tri_i, tri_j = np.triu_indices(n_channels, k=1)
    for _band_block in range(4):
        for i, j in zip(tri_i, tri_j):
            regs = list(channel_map[int(i)]) + list(channel_map[int(j)])
            primary_i = {REGION_KEYS[r] for r in channel_map[int(i)] if REGION_KEYS[r] in {"frontal", "central", "temporal", "parietal", "occipital"}}
            primary_j = {REGION_KEYS[r] for r in channel_map[int(j)] if REGION_KEYS[r] in {"frontal", "central", "temporal", "parietal", "occipital"}}
            if primary_i != primary_j:
                regs.append(REGION_KEYS.index("cross_region"))
            feature_regions.append(sorted(set(regs)))
    return feature_regions


def node_feature_region_map(ch_names: list[str], n_bands: int = 5) -> list[list[int]]:
    feature_regions: list[list[int]] = []
    channel_map = [channel_regions(ch) for ch in ch_names]
    for channel_idx in range(len(ch_names)):
        for _ in range(n_bands):
            feature_regions.append(channel_map[channel_idx])
    return feature_regions


def combined_feature_region_map(base_map: list[list[int]], temporal_map: list[list[int]] | None) -> list[list[int]]:
    out = list(base_map)
    if temporal_map is not None:
        out.extend(temporal_map)
    return out


def build_region_aggregates(x: np.ndarray, feature_region_map: list[list[int]]) -> np.ndarray:
    out = np.zeros((x.shape[0], len(REGION_KEYS)), dtype=np.float32)
    for feat_idx, regions in enumerate(feature_region_map):
        if not regions:
            continue
        scale = 1.0 / float(len(regions))
        for region_idx in regions:
            out[:, int(region_idx)] += x[:, feat_idx] * scale
    return out


def build_temporal_group_aggregates(temporal_x: np.ndarray) -> np.ndarray:
    if int(temporal_x.shape[1]) != 330:
        raise ValueError(f"Expected 330 temporal summary features, got {temporal_x.shape[1]}")
    return np.stack(
        [
            temporal_x[:, :110].mean(axis=1),
            temporal_x[:, 110:220].mean(axis=1),
            temporal_x[:, 220:330].mean(axis=1),
        ],
        axis=1,
    ).astype(np.float32)


def build_expanded_feature_table(
    static_x: np.ndarray,
    temporal_x: np.ndarray,
    feature_region_map: list[list[int]],
) -> tuple[np.ndarray, dict[str, tuple[int, int]]]:
    base_x = np.concatenate([static_x, temporal_x], axis=1).astype(np.float32)
    region_x = build_region_aggregates(base_x, feature_region_map)
    temporal_group_x = build_temporal_group_aggregates(temporal_x)
    expanded_x = np.concatenate([base_x, region_x, temporal_group_x], axis=1).astype(np.float32)
    slices = {
        "feature": (0, int(base_x.shape[1])),
        "region": (int(base_x.shape[1]), int(base_x.shape[1] + region_x.shape[1])),
        "temporal_group": (int(base_x.shape[1] + region_x.shape[1]), int(expanded_x.shape[1])),
    }
    return expanded_x, slices


def build_graph_edges(feature_region_map: list[list[int]], n_feature_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    n_regions = len(REGION_KEYS)
    n_temporal_groups = len(TEMPORAL_GROUP_KEYS)
    region_offset = int(n_feature_nodes)
    temporal_offset = int(region_offset + n_regions)
    global_node = int(temporal_offset + n_temporal_groups)

    src: list[int] = []
    dst: list[int] = []
    weights: list[float] = []

    for feature_idx, regions in enumerate(feature_region_map):
        unique_regions = sorted(set(int(r) for r in regions))
        if not unique_regions:
            unique_regions = [REGION_KEYS.index("global")]
        scale = 1.0 / float(len(unique_regions))
        for region_idx in unique_regions:
            region_node = region_offset + region_idx
            src.extend([feature_idx, region_node])
            dst.extend([region_node, feature_idx])
            weights.extend([scale, scale])

    temporal_start = 1034
    temporal_block = 110
    for group_idx in range(n_temporal_groups):
        temporal_node = temporal_offset + group_idx
        start = temporal_start + (group_idx * temporal_block)
        stop = start + temporal_block
        for feature_idx in range(start, stop):
            src.extend([feature_idx, temporal_node])
            dst.extend([temporal_node, feature_idx])
            weights.extend([1.0, 1.0])

    for region_idx in range(n_regions):
        region_node = region_offset + region_idx
        src.extend([region_node, global_node])
        dst.extend([global_node, region_node])
        weights.extend([1.0, 1.0])

    for group_idx in range(n_temporal_groups):
        temporal_node = temporal_offset + group_idx
        src.extend([temporal_node, global_node])
        dst.extend([global_node, temporal_node])
        weights.extend([1.0, 1.0])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


class ExplicitRegionTemporalNodeDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, subject_ids: np.ndarray, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> None:
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.subject_ids = np.asarray(subject_ids)
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.n_nodes = int(self.x.shape[1]) + 1
        self.node_id = torch.arange(self.n_nodes, dtype=torch.long)

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


class ExplicitRegionTemporalWeightedStarGNN(nn.Module):
    def __init__(self, n_feature_nodes: int, n_region_nodes: int, n_temporal_group_nodes: int) -> None:
        super().__init__()
        self.n_feature_nodes = int(n_feature_nodes)
        self.n_region_nodes = int(n_region_nodes)
        self.n_temporal_group_nodes = int(n_temporal_group_nodes)
        self.n_nodes = self.n_feature_nodes + self.n_region_nodes + self.n_temporal_group_nodes + 1
        self.feature_logits = nn.Embedding(self.n_feature_nodes, 2)
        self.region_logits = nn.Embedding(self.n_region_nodes, 2)
        self.temporal_group_logits = nn.Embedding(self.n_temporal_group_nodes, 2)
        self.bias = nn.Parameter(torch.zeros(2))
        nn.init.zeros_(self.feature_logits.weight)
        nn.init.zeros_(self.region_logits.weight)
        nn.init.zeros_(self.temporal_group_logits.weight)

    def load_logistic_parameters(self, coef: np.ndarray, intercept: np.ndarray) -> None:
        coef = np.asarray(coef, dtype=np.float32).reshape(-1)
        intercept = np.asarray(intercept, dtype=np.float32).reshape(-1)
        if int(coef.shape[0]) != int(self.n_feature_nodes + self.n_region_nodes + self.n_temporal_group_nodes):
            raise ValueError("Coefficient dimension mismatch.")
        feat_end = self.n_feature_nodes
        reg_end = feat_end + self.n_region_nodes
        tmp_end = reg_end + self.n_temporal_group_nodes

        feat_w = torch.zeros((self.n_feature_nodes, 2), dtype=torch.float32)
        reg_w = torch.zeros((self.n_region_nodes, 2), dtype=torch.float32)
        tmp_w = torch.zeros((self.n_temporal_group_nodes, 2), dtype=torch.float32)
        feat_w[:, 1] = torch.tensor(coef[:feat_end], dtype=torch.float32)
        reg_w[:, 1] = torch.tensor(coef[feat_end:reg_end], dtype=torch.float32)
        tmp_w[:, 1] = torch.tensor(coef[reg_end:tmp_end], dtype=torch.float32)

        self.feature_logits.weight.data.copy_(feat_w)
        self.region_logits.weight.data.copy_(reg_w)
        self.temporal_group_logits.weight.data.copy_(tmp_w)
        self.bias.data.zero_()
        self.bias.data[1] = float(intercept[0])

    def forward(self, data: Batch) -> torch.Tensor:
        batch_size = int(data.batch.max().item()) + 1 if data.batch.numel() else 0
        values = data.x.view(batch_size, self.n_nodes)
        feature_values = values[:, : self.n_feature_nodes]
        region_start = self.n_feature_nodes
        region_stop = region_start + self.n_region_nodes
        temporal_start = region_stop
        temporal_stop = temporal_start + self.n_temporal_group_nodes
        region_values = values[:, region_start:region_stop]
        temporal_group_values = values[:, temporal_start:temporal_stop]
        return (
            (feature_values @ self.feature_logits.weight)
            + (region_values @ self.region_logits.weight)
            + (temporal_group_values @ self.temporal_group_logits.weight)
            + self.bias
        )


def collate_graphs(batch: list[Data]) -> Batch:
    return Batch.from_data_list(batch)


@torch.no_grad()
def predict(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        y_all.append(batch.y.view(-1).detach().cpu().numpy())
        p_all.append(prob)
    return np.concatenate(y_all), np.concatenate(p_all)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repeated 5x10 CV for explicit-region temporal-summary pure weighted-star GNN.")
    p.add_argument("--split-csv", type=Path, default=Path("data/splits/task_split_subject_level.csv"))
    p.add_argument("--pcc-manifest", type=Path, default=Path("data/processed/graphs_pcc_topk_task/manifest.csv"))
    p.add_argument("--theta-manifest", type=Path, default=Path("data/processed/graphs_plv_theta_topk_task/manifest.csv"))
    p.add_argument("--alpha-manifest", type=Path, default=Path("data/processed/graphs_plv_alpha_topk_task/manifest.csv"))
    p.add_argument("--beta-manifest", type=Path, default=Path("data/processed/graphs_plv_beta_topk_task/manifest.csv"))
    p.add_argument("--window-seconds", type=float, default=8.0)
    p.add_argument("--step-seconds", type=float, default=8.0)
    p.add_argument("--max-windows", type=int, default=12)
    p.add_argument("--max-seconds", type=int, default=120)
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--c", type=float, default=1.0)
    p.add_argument("--targeted-clean-bad-abs-threshold", type=float, default=None)
    p.add_argument("--targeted-clean-bad-window-ratio", type=float, default=None)
    p.add_argument("--targeted-clean-summary-abs-threshold", type=float, default=None)
    p.add_argument("--targeted-static-clean-bad-window-ratio", type=float, default=None)
    p.add_argument("--targeted-temporal-zero-bad-window-ratio", type=float, default=None)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out-dir", type=Path, default=Path("outputs/metrics/runs/explicit_region_temporal_summary_node_gnn_5x10"))
    p.add_argument("--experiment-name", type=str, default="explicit_region_temporal_summary_node_gnn_5x10")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    seeds = parse_seed_list(args.seeds)
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    static_meta, static_x = build_static_graph_feature_table(
        pcc_manifest=args.pcc_manifest,
        theta_manifest=args.theta_manifest,
        alpha_manifest=args.alpha_manifest,
        beta_manifest=args.beta_manifest,
    )
    targeted_clean_enabled = (
        args.targeted_clean_bad_abs_threshold is not None
        and args.targeted_clean_bad_window_ratio is not None
    )
    if targeted_clean_enabled:
        temporal_meta, temporal_x = build_targeted_clean_temporal_summary_table(
            split_csv=args.split_csv,
            window_seconds=args.window_seconds,
            step_seconds=args.step_seconds,
            max_windows=args.max_windows,
            max_seconds=args.max_seconds,
            bad_window_abs_threshold=float(args.targeted_clean_bad_abs_threshold),
            replace_bad_window_ratio=float(args.targeted_clean_bad_window_ratio),
            clean_window_abs_threshold=args.targeted_clean_summary_abs_threshold,
        )
    else:
        temporal_meta, temporal_x = build_temporal_summary_table(
            split_csv=args.split_csv,
            window_seconds=args.window_seconds,
            step_seconds=args.step_seconds,
            max_windows=args.max_windows,
            max_seconds=args.max_seconds,
        )
    merged = static_meta.merge(temporal_meta, on=["subject_id", "label"], how="inner").sort_values("subject_id").reset_index(drop=True)
    if int(len(merged)) != int(len(static_meta)) or int(len(merged)) != int(len(temporal_meta)):
        raise ValueError("Static and temporal tables did not align on identical subject sets.")

    static_index = {sid: idx for idx, sid in enumerate(static_meta["subject_id"].tolist())}
    temporal_index = {sid: idx for idx, sid in enumerate(temporal_meta["subject_id"].tolist())}
    subject_order = merged["subject_id"].tolist()
    static_x = np.stack([static_x[static_index[sid]] for sid in subject_order]).astype(np.float32)
    temporal_x = np.stack([temporal_x[temporal_index[sid]] for sid in subject_order]).astype(np.float32)
    bad_window_ratio = merged["bad_window_ratio"].to_numpy(dtype=np.float32) if "bad_window_ratio" in merged.columns else None

    static_clean_enabled = args.targeted_static_clean_bad_window_ratio is not None
    if static_clean_enabled:
        clean_static_meta, clean_static_x = build_clean_window_static_graph_feature_table(
            split_csv=args.split_csv,
            window_seconds=args.window_seconds,
            step_seconds=args.step_seconds,
            max_windows=args.max_windows,
            max_seconds=args.max_seconds,
            clean_window_abs_threshold=float(
                args.targeted_clean_summary_abs_threshold
                if args.targeted_clean_summary_abs_threshold is not None
                else args.targeted_clean_bad_abs_threshold
            ),
        )
        clean_static_index = {sid: idx for idx, sid in enumerate(clean_static_meta["subject_id"].tolist())}
        clean_static_x = np.stack([clean_static_x[clean_static_index[sid]] for sid in subject_order]).astype(np.float32)
        if bad_window_ratio is None:
            raise ValueError("Static clean replacement requires bad_window_ratio diagnostics from temporal features.")
        static_replace_mask = bad_window_ratio >= float(args.targeted_static_clean_bad_window_ratio)
        static_x = static_x.copy()
        static_x[static_replace_mask] = clean_static_x[static_replace_mask]
    else:
        static_replace_mask = np.zeros((len(subject_order),), dtype=bool)

    temporal_zero_enabled = args.targeted_temporal_zero_bad_window_ratio is not None
    if temporal_zero_enabled:
        if bad_window_ratio is None:
            raise ValueError("Temporal zeroing requires bad_window_ratio diagnostics from temporal features.")
        temporal_zero_mask = bad_window_ratio >= float(args.targeted_temporal_zero_bad_window_ratio)
        temporal_x = temporal_x.copy()
        temporal_x[temporal_zero_mask] = 0.0
    else:
        temporal_zero_mask = np.zeros((len(subject_order),), dtype=bool)

    y = merged["label"].to_numpy(dtype=np.int64)
    subject_ids = merged["subject_id"].to_numpy()

    base_map = base_feature_region_map(DEFAULT_CH_NAMES)
    node_map = node_feature_region_map(DEFAULT_CH_NAMES)
    temporal_map = node_map + node_map + node_map
    feature_region_map = combined_feature_region_map(base_map, temporal_map)

    expanded_x, slices = build_expanded_feature_table(static_x, temporal_x, feature_region_map)
    n_feature_nodes = int(slices["feature"][1] - slices["feature"][0])
    n_region_nodes = int(slices["region"][1] - slices["region"][0])
    n_temporal_group_nodes = int(slices["temporal_group"][1] - slices["temporal_group"][0])
    edge_index, edge_weight = build_graph_edges(feature_region_map, n_feature_nodes=n_feature_nodes)

    run_payloads: list[dict[str, object]] = []
    for seed in seeds:
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=int(seed))
        fold_results: list[dict[str, object]] = []

        for fold_no, (train_idx, test_idx) in enumerate(skf.split(expanded_x, y), start=1):
            train_x = expanded_x[train_idx]
            test_x = expanded_x[test_idx]
            train_y = y[train_idx]
            test_y = y[test_idx]

            solver = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=5000,
                            class_weight="balanced",
                            C=float(args.c),
                            random_state=int(seed),
                        ),
                    ),
                ]
            )
            solver.fit(train_x, train_y)
            scaler: StandardScaler = solver.named_steps["scaler"]
            clf: LogisticRegression = solver.named_steps["clf"]

            train_z = scaler.transform(train_x).astype(np.float32)
            test_z = scaler.transform(test_x).astype(np.float32)

            model = ExplicitRegionTemporalWeightedStarGNN(
                n_feature_nodes=n_feature_nodes,
                n_region_nodes=n_region_nodes,
                n_temporal_group_nodes=n_temporal_group_nodes,
            ).to(device)
            model.load_logistic_parameters(clf.coef_[0], clf.intercept_)

            test_ds = ExplicitRegionTemporalNodeDataset(
                x=test_z,
                y=test_y,
                subject_ids=subject_ids[test_idx],
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_graphs)
            y_pred, prob_pred = predict(model, test_loader, device=device)
            metrics = calc_metrics(y_pred, prob_pred, threshold=0.5)

            region_coef = clf.coef_[0][slices["region"][0] : slices["region"][1]]
            temporal_group_coef = clf.coef_[0][slices["temporal_group"][0] : slices["temporal_group"][1]]
            fold_results.append(
                {
                    "fold": fold_no,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "test_subject_ids": subject_ids[test_idx].tolist(),
                    "region_coefficients": {name: float(val) for name, val in zip(REGION_KEYS, region_coef.tolist())},
                    "temporal_group_coefficients": {name: float(val) for name, val in zip(TEMPORAL_GROUP_KEYS, temporal_group_coef.tolist())},
                    "test_metrics": metrics,
                }
            )
            print(
                f"seed={seed} fold={fold_no:02d} "
                f"acc={metrics['accuracy']:.4f} "
                f"bal_acc={metrics['balanced_accuracy']:.4f} "
                f"f1={metrics['f1']:.4f} "
                f"auc={metrics['roc_auc'] if metrics['roc_auc'] is not None else 'NA'}"
            )

        payload = {
            "schema_version": "eeg_mdd_explicit_region_temporal_summary_node_gnn_5x10_v1",
            "experiment_name": args.experiment_name,
            "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model": "ExplicitRegionTemporalWeightedStarGNN",
            "device": str(device),
            "params": {
                "folds": int(args.folds),
                "seed": int(seed),
                "seeds": [int(x) for x in seeds],
                "c": float(args.c),
                "window_seconds": float(args.window_seconds),
                "step_seconds": float(args.step_seconds),
                "max_windows": int(args.max_windows),
                "max_seconds": int(args.max_seconds),
                "targeted_clean_enabled": bool(targeted_clean_enabled),
                "targeted_clean_bad_abs_threshold": float(args.targeted_clean_bad_abs_threshold) if args.targeted_clean_bad_abs_threshold is not None else None,
                "targeted_clean_bad_window_ratio": float(args.targeted_clean_bad_window_ratio) if args.targeted_clean_bad_window_ratio is not None else None,
                "targeted_clean_summary_abs_threshold": float(args.targeted_clean_summary_abs_threshold) if args.targeted_clean_summary_abs_threshold is not None else None,
                "targeted_static_clean_bad_window_ratio": float(args.targeted_static_clean_bad_window_ratio) if args.targeted_static_clean_bad_window_ratio is not None else None,
                "targeted_temporal_zero_bad_window_ratio": float(args.targeted_temporal_zero_bad_window_ratio) if args.targeted_temporal_zero_bad_window_ratio is not None else None,
                "region_keys": REGION_KEYS,
                "temporal_group_keys": TEMPORAL_GROUP_KEYS,
            },
            "feature_shape": {
                "n_subjects": int(expanded_x.shape[0]),
                "n_feature_nodes": int(n_feature_nodes),
                "n_region_nodes": int(n_region_nodes),
                "n_temporal_group_nodes": int(n_temporal_group_nodes),
                "n_total_nodes_without_global": int(expanded_x.shape[1]),
            },
            "temporal_cleaning": {
                "n_replaced_subjects": int(merged["temporal_replaced"].sum()) if "temporal_replaced" in merged.columns else 0,
                "mean_bad_window_ratio": float(merged["bad_window_ratio"].mean()) if "bad_window_ratio" in merged.columns else None,
                "max_bad_window_ratio": float(merged["bad_window_ratio"].max()) if "bad_window_ratio" in merged.columns else None,
                "n_static_clean_replaced_subjects": int(np.sum(static_replace_mask)),
                "n_temporal_zero_subjects": int(np.sum(temporal_zero_mask)),
            },
            "aggregate": aggregate_fold_results(fold_results),
            "fold_results": fold_results,
        }
        out_path = args.out_dir / f"explicit_region_temporal_summary_node_gnn_seed{seed}_cv10.json"
        write_json(out_path, payload)
        run_payloads.append({**payload, "_out_path": str(out_path)})
        print(f"saved: {out_path}")

    summary = summarize_run_payloads(run_payloads)
    summary_payload = {
        "schema_version": "eeg_mdd_explicit_region_temporal_summary_node_gnn_5x10_summary_v1",
        "experiment_name": args.experiment_name,
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "model": "ExplicitRegionTemporalWeightedStarGNN",
        "params": {
            "folds": int(args.folds),
            "seeds": [int(x) for x in seeds],
            "c": float(args.c),
            "window_seconds": float(args.window_seconds),
            "step_seconds": float(args.step_seconds),
            "max_windows": int(args.max_windows),
            "max_seconds": int(args.max_seconds),
            "targeted_clean_enabled": bool(targeted_clean_enabled),
            "targeted_clean_bad_abs_threshold": float(args.targeted_clean_bad_abs_threshold) if args.targeted_clean_bad_abs_threshold is not None else None,
            "targeted_clean_bad_window_ratio": float(args.targeted_clean_bad_window_ratio) if args.targeted_clean_bad_window_ratio is not None else None,
            "targeted_clean_summary_abs_threshold": float(args.targeted_clean_summary_abs_threshold) if args.targeted_clean_summary_abs_threshold is not None else None,
            "targeted_static_clean_bad_window_ratio": float(args.targeted_static_clean_bad_window_ratio) if args.targeted_static_clean_bad_window_ratio is not None else None,
            "targeted_temporal_zero_bad_window_ratio": float(args.targeted_temporal_zero_bad_window_ratio) if args.targeted_temporal_zero_bad_window_ratio is not None else None,
            "region_keys": REGION_KEYS,
            "temporal_group_keys": TEMPORAL_GROUP_KEYS,
        },
        "feature_shape": {
            "n_subjects": int(expanded_x.shape[0]),
            "n_feature_nodes": int(n_feature_nodes),
            "n_region_nodes": int(n_region_nodes),
            "n_temporal_group_nodes": int(n_temporal_group_nodes),
            "n_total_nodes_without_global": int(expanded_x.shape[1]),
        },
        "temporal_cleaning": {
            "n_replaced_subjects": int(merged["temporal_replaced"].sum()) if "temporal_replaced" in merged.columns else 0,
            "mean_bad_window_ratio": float(merged["bad_window_ratio"].mean()) if "bad_window_ratio" in merged.columns else None,
            "max_bad_window_ratio": float(merged["bad_window_ratio"].max()) if "bad_window_ratio" in merged.columns else None,
            "n_static_clean_replaced_subjects": int(np.sum(static_replace_mask)),
            "n_temporal_zero_subjects": int(np.sum(temporal_zero_mask)),
        },
        **summary,
    }
    summary_path = args.out_dir / "summary_5x10.json"
    write_json(summary_path, summary_payload)
    print(f"\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
