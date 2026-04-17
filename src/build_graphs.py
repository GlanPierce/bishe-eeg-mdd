from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import hilbert, welch


@dataclass(frozen=True)
class GraphConfig:
    max_seconds: int = 120
    l_freq: float = 0.5
    h_freq: float = 45.0
    pcc_quantile: float = 0.8
    min_edges: int = 30
    edge_mode: str = "pcc"  # pcc | plv | pcc_plv
    fusion_alpha: float = 0.5  # used when edge_mode == pcc_plv
    plv_band: str = "broad"  # broad | theta | alpha | beta


BANDS: dict[str, tuple[float, float]] = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

PLV_BANDS: dict[str, tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def band_power(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        return np.full(psd.shape[0], 1e-12, dtype=float)
    return psd[:, mask].mean(axis=1) + 1e-12


def extract_node_features(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Return node features as per-channel relative band powers.

    Shape: (n_channels, n_bands)
    """
    nperseg = min(1024, data.shape[1])
    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=1)
    total = band_power(psd, freqs, 0.5, 45.0)

    feats: list[np.ndarray] = []
    for fmin, fmax in BANDS.values():
        bp = band_power(psd, freqs, fmin, fmax)
        feats.append((bp / total).reshape(-1, 1))
    return np.concatenate(feats, axis=1)


def _build_sparse_edges(
    score_matrix: np.ndarray,
    quantile: float,
    min_edges: int,
    weight_matrix: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sparse undirected edges from a score matrix.

    score_matrix drives edge selection, weight_matrix provides actual edge weights.
    """
    score_matrix = np.nan_to_num(score_matrix, nan=0.0)
    np.fill_diagonal(score_matrix, 0.0)
    if weight_matrix is None:
        weight_matrix = score_matrix
    weight_matrix = np.nan_to_num(weight_matrix, nan=0.0)
    np.fill_diagonal(weight_matrix, 0.0)

    n = score_matrix.shape[0]
    triu_i, triu_j = np.triu_indices(n, k=1)
    vals = score_matrix[triu_i, triu_j]
    if vals.size == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    thr = float(np.quantile(vals, quantile))
    keep_mask = vals >= thr

    # Guardrail for tiny graphs where quantile can over-prune.
    if int(np.sum(keep_mask)) < min_edges:
        order = np.argsort(vals)[::-1]
        topk = order[: min(min_edges, vals.size)]
        keep_mask = np.zeros_like(keep_mask, dtype=bool)
        keep_mask[topk] = True

    src = triu_i[keep_mask]
    dst = triu_j[keep_mask]
    w = weight_matrix[src, dst]

    edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])]).astype(np.int64)
    edge_weight = np.concatenate([w, w]).astype(np.float32)
    return edge_index, edge_weight


def _pcc_matrix(data: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(data)
    return np.nan_to_num(corr, nan=0.0)


def _plv_matrix(data: np.ndarray) -> np.ndarray:
    """Compute PLV matrix from channel time-series.

    data shape: (n_channels, n_times)
    """
    analytic = hilbert(data, axis=1)
    phase = np.angle(analytic)
    phase_diff = phase[:, None, :] - phase[None, :, :]
    plv = np.abs(np.mean(np.exp(1j * phase_diff), axis=2))
    return np.nan_to_num(plv, nan=0.0)


def _plv_input_data(data: np.ndarray, sfreq: float, plv_band: str) -> np.ndarray:
    if plv_band == "broad":
        return data
    if plv_band not in PLV_BANDS:
        raise ValueError(f"Unknown plv_band: {plv_band}")
    fmin, fmax = PLV_BANDS[plv_band]
    return mne.filter.filter_data(data, sfreq=sfreq, l_freq=fmin, h_freq=fmax, verbose="ERROR")


def build_edges(
    data: np.ndarray,
    sfreq: float,
    edge_mode: str,
    quantile: float,
    min_edges: int,
    fusion_alpha: float,
    plv_band: str,
) -> tuple[np.ndarray, np.ndarray]:
    pcc = _pcc_matrix(data)
    # Use squared magnitude for sparse edge selection while keeping signed edge weights.
    pcc_score = pcc * pcc

    if edge_mode == "pcc":
        return _build_sparse_edges(pcc_score, quantile, min_edges, weight_matrix=pcc)

    plv_data = _plv_input_data(data, sfreq, plv_band)
    plv = _plv_matrix(plv_data)
    if edge_mode == "plv":
        # PLV naturally lies in [0,1], non-negative edge weights.
        return _build_sparse_edges(plv, quantile, min_edges, weight_matrix=plv)

    if edge_mode == "pcc_plv":
        a = float(np.clip(fusion_alpha, 0.0, 1.0))
        # Map PLV from [0,1] to [-1,1] to preserve signed fusion with PCC.
        plv_signed = (2.0 * plv) - 1.0
        fused_signed = (a * pcc) + ((1.0 - a) * plv_signed)
        fused_score = fused_signed * fused_signed
        return _build_sparse_edges(fused_score, quantile, min_edges, weight_matrix=fused_signed)

    raise ValueError(f"Unknown edge_mode: {edge_mode}")


def read_eeg(file_path: Path, cfg: GraphConfig) -> tuple[np.ndarray, float, list[str]]:
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose="ERROR")
    raw.pick("eeg")
    raw.filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq, verbose="ERROR")

    sfreq = float(raw.info["sfreq"])
    n_stop = min(raw.n_times, int(cfg.max_seconds * sfreq))
    data = raw.get_data(start=0, stop=n_stop)
    return data, sfreq, list(raw.ch_names)


def build_graph_for_row(row: pd.Series, cfg: GraphConfig) -> dict[str, object]:
    file_path = Path(str(row["file_path"]))
    data, sfreq, ch_names = read_eeg(file_path, cfg)
    x = extract_node_features(data, sfreq).astype(np.float32)
    edge_index, edge_weight = build_edges(
        data,
        sfreq,
        cfg.edge_mode,
        cfg.pcc_quantile,
        cfg.min_edges,
        cfg.fusion_alpha,
        cfg.plv_band,
    )

    return {
        "subject_id": str(row["subject_id"]),
        "label": int(row["label"]),
        "label_name": str(row["label_name"]),
        "split": str(row["split"]),
        "file_path": str(file_path),
        "n_nodes": int(x.shape[0]),
        "n_edges": int(edge_index.shape[1]),
        "x": x,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "ch_names": ch_names,
        "sfreq": sfreq,
    }


def save_graph(graph: dict[str, object], out_dir: Path) -> Path:
    subject_id = str(graph["subject_id"])
    out_path = out_dir / f"{subject_id}.npz"
    np.savez_compressed(
        out_path,
        x=graph["x"],
        edge_index=graph["edge_index"],
        edge_weight=graph["edge_weight"],
        y=np.array([graph["label"]], dtype=np.int64),
        split=np.array([graph["split"]], dtype=object),
        subject_id=np.array([subject_id], dtype=object),
        label_name=np.array([graph["label_name"]], dtype=object),
        file_path=np.array([graph["file_path"]], dtype=object),
        ch_names=np.array(graph["ch_names"], dtype=object),
        sfreq=np.array([graph["sfreq"]], dtype=np.float32),
    )
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PCC graph data for EEG TASK samples.")
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=Path("data/splits/task_split_subject_level.csv"),
        help="Input split CSV from baseline script.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed/graphs_pcc_task"),
        help="Output directory for graph npz files.",
    )
    parser.add_argument("--max-seconds", type=int, default=120)
    parser.add_argument("--pcc-quantile", type=float, default=0.8)
    parser.add_argument("--min-edges", type=int, default=30)
    parser.add_argument(
        "--edge-mode",
        type=str,
        default="pcc",
        choices=["pcc", "plv", "pcc_plv"],
        help="Connectivity mode for graph edges.",
    )
    parser.add_argument(
        "--fusion-alpha",
        type=float,
        default=0.5,
        help="Used when edge-mode=pcc_plv. fused_signed = alpha*PCC + (1-alpha)*(2*PLV-1).",
    )
    parser.add_argument(
        "--plv-band",
        type=str,
        default="broad",
        choices=["broad", "theta", "alpha", "beta"],
        help="Frequency band used to compute PLV when edge-mode includes PLV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = GraphConfig(
        max_seconds=args.max_seconds,
        pcc_quantile=args.pcc_quantile,
        min_edges=args.min_edges,
        edge_mode=args.edge_mode,
        fusion_alpha=args.fusion_alpha,
        plv_band=args.plv_band,
    )

    split_csv = Path(args.split_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not split_csv.exists():
        raise FileNotFoundError(f"Split CSV not found: {split_csv}")

    df = pd.read_csv(split_csv)
    required_cols = {"file_path", "subject_id", "label", "label_name", "split"}
    miss = required_cols - set(df.columns)
    if miss:
        raise ValueError(f"Split CSV missing columns: {sorted(miss)}")

    records: list[dict[str, object]] = []
    for i, row in df.iterrows():
        print(f"[{i+1}/{len(df)}] build graph: {Path(str(row['file_path'])).name}")
        graph = build_graph_for_row(row, cfg)
        save_path = save_graph(graph, out_dir)
        records.append(
            {
                "subject_id": graph["subject_id"],
                "label": graph["label"],
                "label_name": graph["label_name"],
                "split": graph["split"],
                "n_nodes": graph["n_nodes"],
                "n_edges": graph["n_edges"],
                "graph_path": str(save_path),
                "source_file": graph["file_path"],
            }
        )

    manifest = pd.DataFrame(records).sort_values(["split", "subject_id"])
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    summary = {
        "n_graphs": int(len(manifest)),
        "split_counts": manifest["split"].value_counts().to_dict(),
        "label_counts": manifest["label_name"].value_counts().to_dict(),
        "avg_nodes": float(manifest["n_nodes"].mean()),
        "avg_edges": float(manifest["n_edges"].mean()),
        "config": {
            "max_seconds": cfg.max_seconds,
            "l_freq": cfg.l_freq,
            "h_freq": cfg.h_freq,
            "pcc_quantile": cfg.pcc_quantile,
            "min_edges": cfg.min_edges,
            "edge_mode": cfg.edge_mode,
            "fusion_alpha": cfg.fusion_alpha,
            "plv_band": cfg.plv_band,
            "bands": BANDS,
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(f"- {manifest_path}")
    print(f"- {summary_path}")
    print(f"- {out_dir}/*.npz")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
