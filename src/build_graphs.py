from __future__ import annotations

"""
把原始 EEG 文件转换成“图数据”的脚本。

1. 读取一个受试者的 EDF 脑电文件。
2. 对每个 EEG 通道提取频带能量特征：
   - delta / theta / alpha / beta / gamma 五个频段。
   - 这些特征作为图里的“节点特征”，可以理解为“每个电极自己有什么频率活动”。
3. 计算通道和通道之间的连接强度：
   - PCC：相关系数，表示两个通道波形是否同步升降。
   - PLV：相位锁定值，表示两个通道相位是否同步。
   - wPLI / dwPLI：相位滞后同步指标，用于降低体积传导影响。
4. 根据连接强度筛选边，构成稀疏脑功能图。
5. 保存为 .npz 文件和 manifest.csv，供后续 GNN 或特征模型读取。

在图里：
- 节点 = EEG 通道/电极。
- 节点特征 = 每个通道的五个频段相对功率。
- 边 = 两个通道之间的功能连接。
- 边权重 = 连接强度，可能是正相关、负相关或相位同步强度。
"""

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
    """构图参数集中配置。

    frozen=True 表示创建后不允许随便修改，避免实验过程中参数被意外改掉。
    """

    # 每个 EDF 文件最多读取前 120 秒；这样不同受试者输入时长一致。
    max_seconds: int = 120

    # EEG 常用带通滤波范围，去掉太慢的漂移和高频噪声。
    l_freq: float = 0.5
    h_freq: float = 45.0

    # 用全局分位数筛边时，保留连接强度靠前的一部分。
    # 0.8 表示阈值取所有边分数的 80% 分位附近。
    pcc_quantile: float = 0.8

    # 最少保留多少条边，防止图太稀疏导致没有足够连接信息。
    min_edges: int = 30

    # 两种边筛选方式：
    # - global_quantile：全局统一阈值筛选强边。
    # - per_node_topk：每个节点保留若干最强邻居。
    edge_selection: str = "global_quantile"  # global_quantile | per_node_topk
    top_k_per_node: int = 4

    # PCC 有正负号。True 表示正相关和负相关分别选 top-k，避免负相关被忽略。
    signed_topk_split: bool = True  # keep positive/negative edges separately for signed matrices

    # 使用哪种连接指标构边。
    edge_mode: str = "pcc"  # pcc | plv | wpli | dwpli | pcc_plv

    # 当 edge_mode == pcc_plv 时，PCC 和 PLV 融合的权重。
    fusion_alpha: float = 0.5  # used when edge_mode == pcc_plv

    # 计算 PLV 时使用哪个频段。
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
    """计算每个通道在某个频段内的平均功率。

    psd 的含义：
    - PSD = Power Spectral Density，功率谱密度。
    - 可以理解为“一个通道在每个频率上有多少能量”。

    返回值形状是 (n_channels,)，每个通道一个数。
    """

    # 找出 freqs 中落在 [fmin, fmax) 这个频段的频率点。
    mask = (freqs >= fmin) & (freqs < fmax)
    if not np.any(mask):
        # 如果没有频率点落在该范围，返回一个极小值，避免后续除零或报错。
        return np.full(psd.shape[0], 1e-12, dtype=float)

    # 对频段内的功率求平均。加 1e-12 是数值保护，防止出现 0。
    return psd[:, mask].mean(axis=1) + 1e-12


def extract_node_features(data: np.ndarray, sfreq: float) -> np.ndarray:
    """Return node features as per-channel relative band powers.

    Shape: (n_channels, n_bands)
    """

    # Welch 方法把时间序列转换到频域，得到每个通道的 PSD。
    # nperseg 不能超过信号长度，所以取 1024 和实际长度中的较小值。
    nperseg = min(1024, data.shape[1])
    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=1)

    # 0.5-45Hz 总功率作为分母，用来计算“相对功率”。
    # 相对功率比原始功率更稳，因为不同受试者整体信号强弱可能不同。
    total = band_power(psd, freqs, 0.5, 45.0)

    feats: list[np.ndarray] = []
    for fmin, fmax in BANDS.values():
        # 对每个频段计算功率，再除以总功率。
        # 例如 alpha 相对功率 = alpha 频段功率 / 0.5-45Hz 总功率。
        bp = band_power(psd, freqs, fmin, fmax)
        feats.append((bp / total).reshape(-1, 1))

    # 拼成 (通道数, 5) 的矩阵。
    # 每一行是一个 EEG 通道，每一列是一个频段。
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

    # score_matrix 负责“选哪些边”。
    # weight_matrix 负责“被选中的边最终权重是多少”。
    # 对 PCC 来说，常用 |相关| 或相关平方选边，但保留原始正负相关作为边权。
    score_matrix = np.nan_to_num(score_matrix, nan=0.0)
    np.fill_diagonal(score_matrix, 0.0)
    if weight_matrix is None:
        weight_matrix = score_matrix
    weight_matrix = np.nan_to_num(weight_matrix, nan=0.0)
    np.fill_diagonal(weight_matrix, 0.0)

    n = score_matrix.shape[0]

    # 只取上三角，因为通道 i-j 和 j-i 在无向图中是同一条边。
    triu_i, triu_j = np.triu_indices(n, k=1)
    vals = score_matrix[triu_i, triu_j]
    if vals.size == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    thr = float(np.quantile(vals, quantile))
    keep_mask = vals >= thr

    # Guardrail for tiny graphs where quantile can over-prune.
    if int(np.sum(keep_mask)) < min_edges:
        # 如果按分位数筛完边太少，就强制保留分数最高的 min_edges 条。
        order = np.argsort(vals)[::-1]
        topk = order[: min(min_edges, vals.size)]
        keep_mask = np.zeros_like(keep_mask, dtype=bool)
        keep_mask[topk] = True

    src = triu_i[keep_mask]
    dst = triu_j[keep_mask]
    w = weight_matrix[src, dst]

    # PyG / DGL 等图框架通常用有向边列表表示图。
    # 无向边 i-j 要保存成 i->j 和 j->i 两条有向边。
    edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])]).astype(np.int64)
    edge_weight = np.concatenate([w, w]).astype(np.float32)
    return edge_index, edge_weight


def _build_topk_edges(
    weight_matrix: np.ndarray,
    top_k: int,
    signed_split: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sparse undirected edges by per-node top-k policy.

    - signed_split=True: keep top-k positive and top-k negative edges per node separately.
    - signed_split=False: keep top-k strongest edges per node (positive-only case like PLV).
    """

    # per-node top-k 的思路：
    # 每个通道都保留自己最重要的若干邻居，避免某些节点完全没有边。
    w = np.nan_to_num(weight_matrix, nan=0.0)
    np.fill_diagonal(w, 0.0)
    n = int(w.shape[0])
    if n <= 1:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    k = max(1, int(top_k))
    undirected: dict[tuple[int, int], float] = {}

    for i in range(n):
        row = w[i]
        if signed_split:
            # PCC 可能为正，也可能为负。
            # 正相关表示两个通道一起升降，负相关表示一个升另一个降。
            # 两者都可能有判别意义，所以分开保留。
            pos_idx = np.where(row > 0.0)[0]
            if pos_idx.size > 0:
                pos_vals = row[pos_idx]
                order = np.argsort(pos_vals)[::-1][:k]
                for j in pos_idx[order]:
                    a, b = (i, int(j)) if i < int(j) else (int(j), i)
                    if a != b:
                        undirected[(a, b)] = float(w[a, b])

            neg_idx = np.where(row < 0.0)[0]
            if neg_idx.size > 0:
                # 负相关按绝对值排序，越负代表反向同步越强。
                neg_vals = np.abs(row[neg_idx])
                order = np.argsort(neg_vals)[::-1][:k]
                for j in neg_idx[order]:
                    a, b = (i, int(j)) if i < int(j) else (int(j), i)
                    if a != b:
                        undirected[(a, b)] = float(w[a, b])
        else:
            # PLV / wPLI 等指标天然非负，只需要保留最大值即可。
            nz_idx = np.where(row > 0.0)[0]
            if nz_idx.size == 0:
                continue
            vals = row[nz_idx]
            order = np.argsort(vals)[::-1][:k]
            for j in nz_idx[order]:
                a, b = (i, int(j)) if i < int(j) else (int(j), i)
                if a != b:
                    undirected[(a, b)] = float(w[a, b])

    if not undirected:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    src = np.array([ab[0] for ab in undirected.keys()], dtype=np.int64)
    dst = np.array([ab[1] for ab in undirected.keys()], dtype=np.int64)
    ew = np.array([v for v in undirected.values()], dtype=np.float32)
    edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])]).astype(np.int64)
    edge_weight = np.concatenate([ew, ew]).astype(np.float32)
    return edge_index, edge_weight


def _pcc_matrix(data: np.ndarray) -> np.ndarray:
    """计算 PCC 连接矩阵。

    输入 data 形状是 (通道数, 时间点数)。
    输出矩阵形状是 (通道数, 通道数)。

    PCC 越接近 1：两个通道越同步升降。
    PCC 越接近 -1：两个通道越反向变化。
    PCC 接近 0：线性相关弱。
    """

    corr = np.corrcoef(data)
    return np.nan_to_num(corr, nan=0.0)


def _plv_matrix(data: np.ndarray) -> np.ndarray:
    """Compute PLV matrix from channel time-series.

    data shape: (n_channels, n_times)
    """

    # Hilbert 变换把实数信号转换成解析信号，从中可以取瞬时相位。
    analytic = hilbert(data, axis=1)
    phase = np.angle(analytic)

    # phase_diff[i, j, t] 表示第 t 个时间点上，通道 i 和通道 j 的相位差。
    phase_diff = phase[:, None, :] - phase[None, :, :]

    # exp(1j * phase_diff) 把相位差映射到单位圆。
    # 如果相位差长期稳定，平均后长度接近 1；如果相位差很乱，平均后接近 0。
    plv = np.abs(np.mean(np.exp(1j * phase_diff), axis=2))
    return np.nan_to_num(plv, nan=0.0)


def _wpli_matrix(data: np.ndarray, debiased: bool = False) -> np.ndarray:
    """Compute wPLI / dwPLI matrix from channel time-series.

    data shape: (n_channels, n_times)
    """

    # wPLI 也是相位同步指标，但更关注“非零相位滞后”的同步，
    # 可以一定程度降低体积传导造成的虚假零相位同步。
    analytic = hilbert(data, axis=1)
    n = int(analytic.shape[0])
    out = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        out[i, i] = 1.0
    for i in range(n):
        zi = analytic[i]
        for j in range(i + 1, n):
            zj = analytic[j]
            im = np.imag(zi * np.conj(zj)).astype(np.float64)
            abs_sum = float(np.sum(np.abs(im)))
            if abs_sum <= 1e-12:
                # 分母太小，说明没有可靠相位滞后信息。
                val = 0.0
            elif not debiased:
                # 普通 wPLI。
                val = float(abs(np.sum(im)) / abs_sum)
            else:
                # debiased wPLI，用于减少有限样本偏差。
                im_sum = float(np.sum(im))
                im_sq_sum = float(np.sum(im * im))
                num = (im_sum * im_sum) - im_sq_sum
                den = (abs_sum * abs_sum) - im_sq_sum
                if den <= 1e-12:
                    val = 0.0
                else:
                    # dwPLI can be slightly negative in finite samples; clip for graph weight stability.
                    val = float(np.clip(num / den, 0.0, 1.0))
            out[i, j] = val
            out[j, i] = val
    return np.nan_to_num(out, nan=0.0)


def _phase_input_data(data: np.ndarray, sfreq: float, plv_band: str) -> np.ndarray:
    """给 PLV / wPLI 准备输入信号。

    broad 表示直接使用宽频信号。
    theta/alpha/beta 表示先滤到对应频段，再计算相位同步。
    """

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
    edge_selection: str,
    top_k_per_node: int,
    signed_topk_split: bool,
    fusion_alpha: float,
    plv_band: str,
) -> tuple[np.ndarray, np.ndarray]:
    """根据指定连接指标构建图边。

    这个函数是“边构建”的总入口：
    - 先计算连接矩阵。
    - 再按 edge_selection 筛出稀疏边。
    - 最后返回 edge_index 和 edge_weight。
    """

    pcc = _pcc_matrix(data)
    # Use squared magnitude for sparse edge selection while keeping signed edge weights.
    # PCC 有正负号，选边时更关心强度，所以用平方作为 score。
    # 但真正保存的边权仍然是原始 PCC，保留正负信息。
    pcc_score = pcc * pcc

    if edge_mode == "pcc":
        if edge_selection == "per_node_topk":
            return _build_topk_edges(pcc, top_k=top_k_per_node, signed_split=signed_topk_split)
        return _build_sparse_edges(pcc_score, quantile, min_edges, weight_matrix=pcc)

    if edge_mode in {"plv", "wpli", "dwpli", "pcc_plv"}:
        # PLV / wPLI 都依赖相位，所以先准备相位输入数据。
        phase_data = _phase_input_data(data, sfreq, plv_band)

    if edge_mode == "plv":
        plv = _plv_matrix(phase_data)
        # PLV naturally lies in [0,1], non-negative edge weights.
        if edge_selection == "per_node_topk":
            return _build_topk_edges(plv, top_k=top_k_per_node, signed_split=False)
        return _build_sparse_edges(plv, quantile, min_edges, weight_matrix=plv)

    if edge_mode == "wpli":
        wpli = _wpli_matrix(phase_data, debiased=False)
        if edge_selection == "per_node_topk":
            return _build_topk_edges(wpli, top_k=top_k_per_node, signed_split=False)
        return _build_sparse_edges(wpli, quantile, min_edges, weight_matrix=wpli)

    if edge_mode == "dwpli":
        dwpli = _wpli_matrix(phase_data, debiased=True)
        if edge_selection == "per_node_topk":
            return _build_topk_edges(dwpli, top_k=top_k_per_node, signed_split=False)
        return _build_sparse_edges(dwpli, quantile, min_edges, weight_matrix=dwpli)

    if edge_mode == "pcc_plv":
        plv = _plv_matrix(phase_data)
        a = float(np.clip(fusion_alpha, 0.0, 1.0))
        # Map PLV from [0,1] to [-1,1] to preserve signed fusion with PCC.
        # PLV 本来是 0 到 1，没有负号；这里映射到 -1 到 1，便于和 PCC 融合。
        plv_signed = (2.0 * plv) - 1.0
        fused_signed = (a * pcc) + ((1.0 - a) * plv_signed)
        fused_score = fused_signed * fused_signed
        if edge_selection == "per_node_topk":
            return _build_topk_edges(fused_signed, top_k=top_k_per_node, signed_split=signed_topk_split)
        return _build_sparse_edges(fused_score, quantile, min_edges, weight_matrix=fused_signed)

    raise ValueError(f"Unknown edge_mode: {edge_mode}")


def read_eeg(file_path: Path, cfg: GraphConfig) -> tuple[np.ndarray, float, list[str]]:
    """读取并预处理单个 EDF 文件。

    返回：
    - data：形状为 (通道数, 时间点数) 的 EEG 数组。
    - sfreq：采样率，例如每秒多少个采样点。
    - ch_names：通道名称列表。
    """

    raw = mne.io.read_raw_edf(file_path, preload=True, verbose="ERROR")

    # 只保留 EEG 通道，排除非脑电通道。
    raw.pick("eeg")

    # 带通滤波，保留 EEG 常用频段。
    raw.filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq, verbose="ERROR")

    sfreq = float(raw.info["sfreq"])

    # 不同受试者文件长度可能不同，这里统一最多取前 max_seconds 秒。
    n_stop = min(raw.n_times, int(cfg.max_seconds * sfreq))
    data = raw.get_data(start=0, stop=n_stop)
    return data, sfreq, list(raw.ch_names)


def build_graph_for_row(row: pd.Series, cfg: GraphConfig) -> dict[str, object]:
    """把 split/manifest 表中的一行样本转换成图。

    row 里至少包含：
    - file_path：EDF 文件路径。
    - subject_id：受试者编号。
    - label：分类标签。
    - split：训练/测试等划分信息。
    """

    file_path = Path(str(row["file_path"]))
    data, sfreq, ch_names = read_eeg(file_path, cfg)

    # x 是节点特征矩阵，每一行对应一个 EEG 通道。
    x = extract_node_features(data, sfreq).astype(np.float32)

    # edge_index / edge_weight 描述通道之间如何连接。
    edge_index, edge_weight = build_edges(
        data,
        sfreq,
        cfg.edge_mode,
        cfg.pcc_quantile,
        cfg.min_edges,
        cfg.edge_selection,
        cfg.top_k_per_node,
        cfg.signed_topk_split,
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
    """把一个受试者的图保存成压缩 npz 文件。

    后续训练脚本不需要重新读取 EDF，可以直接读取这些 npz 图文件。
    """

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
    """命令行参数。

    例如可以通过 --edge-mode plv 改成 PLV 构图，
    或通过 --edge-selection per_node_topk 改成每节点 top-k 策略。
    """

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
        "--edge-selection",
        type=str,
        default="global_quantile",
        choices=["global_quantile", "per_node_topk"],
        help="Edge selection policy: global quantile threshold or per-node top-k.",
    )
    parser.add_argument(
        "--top-k-per-node",
        type=int,
        default=4,
        help="Used when edge-selection=per_node_topk.",
    )
    parser.add_argument(
        "--signed-topk-split",
        action="store_true",
        help="Used when edge-selection=per_node_topk on signed matrices: keep positive/negative edges separately.",
    )
    parser.add_argument(
        "--no-signed-topk-split",
        dest="signed_topk_split",
        action="store_false",
        help="Disable separate positive/negative top-k selection for signed matrices.",
    )
    parser.set_defaults(signed_topk_split=True)
    parser.add_argument(
        "--edge-mode",
        type=str,
        default="pcc",
        choices=["pcc", "plv", "wpli", "dwpli", "pcc_plv"],
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
    """脚本入口：读取 CSV，逐个样本构图，保存 manifest 和 summary。"""

    args = parse_args()
    cfg = GraphConfig(
        max_seconds=args.max_seconds,
        pcc_quantile=args.pcc_quantile,
        min_edges=args.min_edges,
        edge_selection=args.edge_selection,
        top_k_per_node=args.top_k_per_node,
        signed_topk_split=args.signed_topk_split,
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
        # 每一行对应一个受试者样本。
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

    # manifest 记录每个受试者的图文件路径，后续模型根据它加载数据。
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
            "edge_selection": cfg.edge_selection,
            "top_k_per_node": cfg.top_k_per_node,
            "signed_topk_split": cfg.signed_topk_split,
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
