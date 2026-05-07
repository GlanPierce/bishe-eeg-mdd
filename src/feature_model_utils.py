from __future__ import annotations

"""
特征模型和实验结果的工具函数。

这个文件连接了“图构建”和“模型训练”两个阶段：

1. 从 build_graphs.py 生成的 .npz 图文件中读取特征。
2. 把每个受试者的图转换成一行固定长度的特征向量。
3. 从原始 EEG 中按时间窗口提取动态摘要特征。
4. 计算 Accuracy / Balanced Accuracy / F1 / ROC-AUC 等指标。
5. 汇总 10 折交叉验证和 5 个随机种子的结果。

- .npz 图文件像“一个受试者的一张脑网络图”。
- 模型训练前，脚本会把每张图压平成一个数字向量。
- 静态特征描述“整体脑网络长什么样”。
- 时间摘要描述“脑电在多个时间窗口中怎样变化”。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from build_graphs import GraphConfig, _pcc_matrix, _phase_input_data, _plv_matrix, extract_node_features, read_eeg


def parse_seed_list(seed_text: str) -> list[int]:
    """把命令行里的种子字符串转换成整数列表。

    例如 "42,43,44" 会变成 [42, 43, 44]。
    多个种子用于重复实验，检查结果是否稳定。
    """

    seeds = [int(x.strip()) for x in str(seed_text).split(",") if x.strip()]
    if not seeds:
        raise ValueError("Seed list must not be empty.")
    return seeds


def calc_metrics(y_true: np.ndarray, prob_pos: np.ndarray, threshold: float = 0.5) -> dict[str, object]:
    """根据真实标签和模型输出概率计算分类指标。

    参数解释：
    - y_true：真实标签，0/1。
    - prob_pos：模型预测为正类的概率。这里正类通常是 MDD。
    - threshold：分类阈值。prob_pos >= threshold 判为 1。

    指标解释：
    - accuracy：总体预测对的比例。
    - balanced_accuracy：先分别算每一类的准确率，再平均；适合类别不完全均衡的数据。
    - f1：综合 precision 和 recall 的指标。
    - roc_auc：看模型排序能力，不依赖固定阈值。
    - confusion_matrix：混淆矩阵，展示 TP/FP/TN/FN 的分布。
    """

    # 概率转成 0/1 分类结果。
    pred = (prob_pos >= threshold).astype(np.int64)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, prob_pos)) if len(np.unique(y_true)) == 2 else None,
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def aggregate_fold_results(fold_results: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    """汇总一个 seed 下的 10 折结果。

    fold_results 中每个元素是一折测试集的指标。
    这里对每个指标求均值和标准差。
    """

    metric_names = ["accuracy", "balanced_accuracy", "f1", "roc_auc"]
    agg: dict[str, dict[str, float]] = {}
    for metric_name in metric_names:
        values = np.array([fr["test_metrics"][metric_name] for fr in fold_results], dtype=np.float64)
        agg[metric_name] = {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values))}
    return agg


def summarize_run_payloads(run_payloads: list[dict[str, object]]) -> dict[str, object]:
    """汇总多个 seed 的实验结果。

    例如 5 seeds x 10 folds：
    - 每个 seed 先得到 10 折均值。
    - 再对 5 个 seed 的均值求平均和标准差。

    最终论文/报告里的 0.9143 +/- 0.0060 就来自这里的跨 seed 汇总。
    """

    runs: list[dict[str, object]] = []
    for payload in run_payloads:
        agg = payload["aggregate"]
        run_path = payload.get("_out_path", "")
        runs.append(
            {
                "seed": int(payload["params"]["seed"]),
                "path": str(run_path),
                "accuracy_mean": float(agg["accuracy"]["mean"]),
                "accuracy_std": float(agg["accuracy"]["std"]),
                "balanced_accuracy_mean": float(agg["balanced_accuracy"]["mean"]),
                "balanced_accuracy_std": float(agg["balanced_accuracy"]["std"]),
                "f1_mean": float(agg["f1"]["mean"]),
                "f1_std": float(agg["f1"]["std"]),
                "roc_auc_mean": float(agg["roc_auc"]["mean"]),
                "roc_auc_std": float(agg["roc_auc"]["std"]),
            }
        )

    metric_pairs = [
        ("accuracy_mean", "accuracy_mean_of_means", "accuracy_std_of_means"),
        ("balanced_accuracy_mean", "balanced_accuracy_mean_of_means", "balanced_accuracy_std_of_means"),
        ("f1_mean", "f1_mean_of_means", "f1_std_of_means"),
        ("roc_auc_mean", "roc_auc_mean_of_means", "roc_auc_std_of_means"),
    ]
    summary_over_runs: dict[str, float] = {}
    for run_key, mean_key, std_key in metric_pairs:
        values = np.array([run[run_key] for run in runs], dtype=np.float64)
        summary_over_runs[mean_key] = float(np.nanmean(values))
        summary_over_runs[std_key] = float(np.nanstd(values))

    return {
        "n_runs": int(len(runs)),
        "runs": runs,
        "summary_over_runs": summary_over_runs,
    }


def _load_graph_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取一个 .npz 图文件，并转换成节点特征和邻接矩阵。

    .npz 中保存的是：
    - x：节点特征，形状通常是 (通道数, 频带数)。
    - edge_index：边的起点和终点。
    - edge_weight：每条边的权重。

    这里把 edge_index + edge_weight 还原成一个方阵 adj：
    adj[i, j] 表示通道 i 到通道 j 的连接强度。
    """

    d = np.load(npz_path, allow_pickle=True)
    x = np.asarray(d["x"], dtype=np.float32)
    edge_index = np.asarray(d["edge_index"], dtype=np.int64)
    edge_weight = np.asarray(d["edge_weight"], dtype=np.float32)
    n_nodes = int(x.shape[0])
    adj = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for (src, dst), weight in zip(edge_index.T, edge_weight):
        # edge_index.T 的每一行是 [src, dst]。
        # 把边权填回邻接矩阵对应位置。
        adj[int(src), int(dst)] = float(weight)
    return x, adj


def build_static_graph_feature_table(
    pcc_manifest: Path,
    theta_manifest: Path,
    alpha_manifest: Path,
    beta_manifest: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    """构建静态图特征表。

    输入是四套 manifest：
    - PCC 图：通道间相关系数连接。
    - theta PLV 图：theta 频段相位同步。
    - alpha PLV 图：alpha 频段相位同步。
    - beta PLV 图：beta 频段相位同步。

    输出：
    - master：包含 subject_id 和 label 的表。
    - features：每行一个受试者的静态特征向量。

    静态特征维度计算：
    - node_x.reshape(-1)：22 通道 x 5 频带 = 110。
    - 每个连接矩阵取上三角：22*21/2 = 231。
    - PCC + theta + alpha + beta 共 4 个连接矩阵：231*4 = 924。
    - 总维度：110 + 924 = 1034。
    """

    # 读取每套图的 manifest，只保留后续需要的列。
    pcc = pd.read_csv(pcc_manifest)[["subject_id", "label", "graph_path"]].rename(columns={"graph_path": "pcc_path"})
    theta = pd.read_csv(theta_manifest)[["subject_id", "graph_path"]].rename(columns={"graph_path": "theta_path"})
    alpha = pd.read_csv(alpha_manifest)[["subject_id", "graph_path"]].rename(columns={"graph_path": "alpha_path"})
    beta = pd.read_csv(beta_manifest)[["subject_id", "graph_path"]].rename(columns={"graph_path": "beta_path"})
    master = pcc.merge(theta, on="subject_id").merge(alpha, on="subject_id").merge(beta, on="subject_id")

    # 按 subject_id 排序，确保每次实验的样本顺序一致。
    master = master.sort_values("subject_id").reset_index(drop=True)

    features: list[np.ndarray] = []
    for _, row in master.iterrows():
        # 读取同一个受试者的四种图。
        node_x, adj_pcc = _load_graph_npz(Path(str(row["pcc_path"])))
        _, adj_theta = _load_graph_npz(Path(str(row["theta_path"])))
        _, adj_alpha = _load_graph_npz(Path(str(row["alpha_path"])))
        _, adj_beta = _load_graph_npz(Path(str(row["beta_path"])))
        tri = np.triu_indices(node_x.shape[0], k=1)

        # 把图转换成一行向量：
        # 先放节点特征，再放四种连接矩阵的上三角边权。
        feat = np.concatenate(
            [
                node_x.reshape(-1),
                adj_pcc[tri],
                adj_theta[tri],
                adj_alpha[tri],
                adj_beta[tri],
            ]
        ).astype(np.float32)
        features.append(feat)
    return master[["subject_id", "label"]].copy(), np.stack(features)


def _window_starts(n_samples: int, win: int, step: int) -> list[int]:
    """生成时间窗口起点。

    n_samples：总采样点数。
    win：一个窗口包含多少采样点。
    step：相邻窗口之间隔多少采样点。

    返回值例如 [0, 2000, 4000]，表示从这些位置开始切窗口。
    """

    if n_samples <= win:
        # 信号比窗口还短时，至少返回一个窗口，从 0 开始。
        return [0]
    starts = list(range(0, n_samples - win + 1, step))
    if starts[-1] != n_samples - win:
        # 如果最后一个窗口没有覆盖到信号末尾，则补一个贴着末尾的窗口。
        starts.append(n_samples - win)
    return starts


def _summarize_window_feature_stack(stacked: np.ndarray) -> np.ndarray:
    """把多个时间窗口的特征压缩成一个摘要向量。

    stacked 形状：
    - 行：时间窗口。
    - 列：每个窗口的特征。

    输出由三部分拼接：
    1. mean：所有窗口的平均水平。
    2. std：窗口之间的波动程度。
    3. delta：最后一个窗口 - 第一个窗口，表示趋势变化。
    """

    return np.concatenate(
        [
            stacked.mean(axis=0),
            stacked.std(axis=0),
            stacked[-1] - stacked[0],
        ]
    ).astype(np.float32)


def _upper_tri_features(mat: np.ndarray) -> np.ndarray:
    """取矩阵上三角作为特征。

    因为连接矩阵通常是对称的，mat[i, j] 和 mat[j, i] 表示同一条无向边。
    只取上三角可以避免重复。
    """

    tri = np.triu_indices(mat.shape[0], k=1)
    return mat[tri].astype(np.float32)


def _build_static_window_feature(segment: np.ndarray, sfreq: float) -> np.ndarray:
    """为一个短时间窗口构建静态图特征。

    这个函数用于“干净窗口替换”版本：
    - 对每个窗口单独提取节点频带功率。
    - 计算 PCC 和 theta/alpha/beta PLV。
    - 拼成和静态 1034 维相同格式的向量。
    """

    node_x = extract_node_features(segment, sfreq).astype(np.float32)
    theta_data = _phase_input_data(segment, sfreq, "theta")
    alpha_data = _phase_input_data(segment, sfreq, "alpha")
    beta_data = _phase_input_data(segment, sfreq, "beta")
    return np.concatenate(
        [
            node_x.reshape(-1),
            _upper_tri_features(_pcc_matrix(segment)),
            _upper_tri_features(_plv_matrix(theta_data)),
            _upper_tri_features(_plv_matrix(alpha_data)),
            _upper_tri_features(_plv_matrix(beta_data)),
        ]
    ).astype(np.float32)


def build_temporal_summary_table(
    split_csv: Path,
    window_seconds: float,
    step_seconds: float,
    max_windows: int,
    max_seconds: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """构建时间摘要特征表。

    对每个受试者：
    1. 读取前 max_seconds 秒 EEG。
    2. 按 window_seconds 切成多个窗口。
    3. 每个窗口提取 22*5=110 维节点频带功率。
    4. 对窗口序列做 mean/std/delta 摘要。

    输出 temporal_x 的维度：
    - mean 110 维。
    - std 110 维。
    - delta 110 维。
    - 总计 330 维。
    """

    df = pd.read_csv(split_csv)
    required_cols = {"file_path", "subject_id", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"split csv missing columns: {sorted(missing)}")

    df = df.sort_values("subject_id").drop_duplicates(subset=["subject_id"], keep="first").reset_index(drop=True)

    # 复用 GraphConfig 里的读取和滤波设置，保证与静态图构建一致。
    cfg = GraphConfig(max_seconds=max_seconds)

    features: list[np.ndarray] = []
    n_windows_all: list[int] = []
    for idx, row in df.iterrows():
        file_path = Path(str(row["file_path"]))
        subject_id = str(row["subject_id"])
        data, sfreq, _ = read_eeg(file_path, cfg)

        # 秒数转换成采样点数。
        # 例如采样率 250Hz，8 秒窗口就是 2000 个采样点。
        win = max(1, int(round(float(window_seconds) * sfreq)))
        step = max(1, int(round(float(step_seconds) * sfreq)))
        starts = _window_starts(data.shape[1], win, step)[: int(max_windows)]

        window_feats: list[np.ndarray] = []
        for start in starts:
            stop = min(start + win, data.shape[1])
            segment = data[:, start:stop]
            if segment.shape[1] < 8:
                # 太短的片段没有稳定频谱意义，跳过。
                continue

            # 每个窗口只提取节点频带功率，不计算连接矩阵。
            # 这样时间分支维度较小，更适合小样本。
            window_feats.append(extract_node_features(segment, sfreq).reshape(-1))

        if not window_feats:
            # 如果所有窗口都被跳过，就退回到整段信号提取一次。
            window_feats.append(extract_node_features(data, sfreq).reshape(-1))

        stacked = np.stack(window_feats).astype(np.float32)
        summary = _summarize_window_feature_stack(stacked)
        print(
            f"[{idx + 1}/{len(df)}] temporal summary: {subject_id} "
            f"windows={stacked.shape[0]} summary_dim={summary.shape[0]}"
        )
        features.append(summary)
        n_windows_all.append(int(stacked.shape[0]))

    out_df = df[["subject_id", "label"]].copy()
    out_df["n_windows"] = n_windows_all
    return out_df, np.stack(features)


def build_targeted_clean_temporal_summary_table(
    split_csv: Path,
    window_seconds: float,
    step_seconds: float,
    max_windows: int,
    max_seconds: int,
    bad_window_abs_threshold: float,
    replace_bad_window_ratio: float,
    clean_window_abs_threshold: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """带伪迹感知的时间摘要表。

    这是 benchmark-tuned / targeted repair 版本使用的函数。
    clean 0.9143 主结果默认不启用。

    思路：
    - 每个窗口计算最大绝对振幅。
    - 如果坏窗口比例超过阈值，就只用干净窗口生成摘要。
    - 同时记录 bad_window_ratio 等诊断信息。
    """

    df = pd.read_csv(split_csv)
    required_cols = {"file_path", "subject_id", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"split csv missing columns: {sorted(missing)}")

    df = df.sort_values("subject_id").drop_duplicates(subset=["subject_id"], keep="first").reset_index(drop=True)
    cfg = GraphConfig(max_seconds=max_seconds)
    clean_thr = float(bad_window_abs_threshold if clean_window_abs_threshold is None else clean_window_abs_threshold)

    features: list[np.ndarray] = []
    n_windows_all: list[int] = []
    n_clean_windows_all: list[int] = []
    bad_ratio_all: list[float] = []
    replaced_all: list[int] = []
    for idx, row in df.iterrows():
        file_path = Path(str(row["file_path"]))
        subject_id = str(row["subject_id"])
        data, sfreq, _ = read_eeg(file_path, cfg)
        win = max(1, int(round(float(window_seconds) * sfreq)))
        step = max(1, int(round(float(step_seconds) * sfreq)))
        starts = _window_starts(data.shape[1], win, step)[: int(max_windows)]

        window_feats: list[np.ndarray] = []
        window_abs_max: list[float] = []
        for start in starts:
            stop = min(start + win, data.shape[1])
            segment = data[:, start:stop]
            if segment.shape[1] < 8:
                continue
            window_feats.append(extract_node_features(segment, sfreq).reshape(-1))
            window_abs_max.append(float(np.max(np.abs(segment))))

        if not window_feats:
            window_feats.append(extract_node_features(data, sfreq).reshape(-1))
            window_abs_max.append(float(np.max(np.abs(data))))

        stacked = np.stack(window_feats).astype(np.float32)
        abs_max = np.asarray(window_abs_max, dtype=np.float32)

        # clean_mask=True 表示该窗口振幅没有超过干净阈值。
        clean_mask = abs_max <= float(clean_thr)

        # bad_ratio 表示坏窗口占比。
        bad_ratio = float(np.mean(abs_max > float(bad_window_abs_threshold)))

        # 坏窗口比例足够高时，触发替换逻辑。
        should_replace = bool(bad_ratio >= float(replace_bad_window_ratio))

        # 如果需要替换且存在干净窗口，则只用干净窗口；否则使用全部窗口。
        chosen = stacked[clean_mask] if should_replace and np.any(clean_mask) else stacked
        summary = _summarize_window_feature_stack(chosen)
        print(
            f"[{idx + 1}/{len(df)}] targeted temporal summary: {subject_id} "
            f"windows={stacked.shape[0]} clean_windows={int(np.sum(clean_mask))} "
            f"bad_ratio={bad_ratio:.3f} replaced={int(should_replace)}"
        )
        features.append(summary)
        n_windows_all.append(int(stacked.shape[0]))
        n_clean_windows_all.append(int(np.sum(clean_mask)))
        bad_ratio_all.append(bad_ratio)
        replaced_all.append(int(should_replace))

    out_df = df[["subject_id", "label"]].copy()
    out_df["n_windows"] = n_windows_all
    out_df["n_clean_windows"] = n_clean_windows_all
    out_df["bad_window_ratio"] = bad_ratio_all
    out_df["temporal_replaced"] = replaced_all
    return out_df, np.stack(features)


def build_clean_window_static_graph_feature_table(
    split_csv: Path,
    window_seconds: float,
    step_seconds: float,
    max_windows: int,
    max_seconds: int,
    clean_window_abs_threshold: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """用干净窗口均值替换静态图特征。

    这是 targeted repair 相关辅助函数。
    clean 0.9143 主结果默认不使用。

    与 build_temporal_summary_table 不同：
    - 这里每个窗口会构建完整静态 1034 维图特征。
    - 然后只对干净窗口求平均，作为该受试者的静态特征。
    """

    df = pd.read_csv(split_csv)
    required_cols = {"file_path", "subject_id", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"split csv missing columns: {sorted(missing)}")

    df = df.sort_values("subject_id").drop_duplicates(subset=["subject_id"], keep="first").reset_index(drop=True)
    cfg = GraphConfig(max_seconds=max_seconds)

    features: list[np.ndarray] = []
    n_windows_all: list[int] = []
    n_clean_windows_all: list[int] = []
    bad_ratio_all: list[float] = []
    max_abs_all: list[float] = []
    for idx, row in df.iterrows():
        file_path = Path(str(row["file_path"]))
        subject_id = str(row["subject_id"])
        data, sfreq, _ = read_eeg(file_path, cfg)
        win = max(1, int(round(float(window_seconds) * sfreq)))
        step = max(1, int(round(float(step_seconds) * sfreq)))
        starts = _window_starts(data.shape[1], win, step)[: int(max_windows)]

        window_feats: list[np.ndarray] = []
        window_abs_max: list[float] = []
        for start in starts:
            stop = min(start + win, data.shape[1])
            segment = data[:, start:stop]
            if segment.shape[1] < 8:
                continue
            window_feats.append(_build_static_window_feature(segment, sfreq))
            window_abs_max.append(float(np.max(np.abs(segment))))

        if not window_feats:
            window_feats.append(_build_static_window_feature(data, sfreq))
            window_abs_max.append(float(np.max(np.abs(data))))

        stacked = np.stack(window_feats).astype(np.float32)
        abs_max = np.asarray(window_abs_max, dtype=np.float32)
        clean_mask = abs_max <= float(clean_window_abs_threshold)

        # 优先用干净窗口；如果一个干净窗口都没有，就退回使用全部窗口。
        chosen = stacked[clean_mask] if np.any(clean_mask) else stacked

        # 多个窗口的静态图特征取平均，得到一个 1034 维向量。
        summary = chosen.mean(axis=0).astype(np.float32)
        bad_ratio = float(np.mean(abs_max > float(clean_window_abs_threshold)))
        print(
            f"[{idx + 1}/{len(df)}] clean static graph: {subject_id} "
            f"windows={stacked.shape[0]} clean_windows={int(np.sum(clean_mask))} "
            f"bad_ratio={bad_ratio:.3f}"
        )
        features.append(summary)
        n_windows_all.append(int(stacked.shape[0]))
        n_clean_windows_all.append(int(np.sum(clean_mask)))
        bad_ratio_all.append(bad_ratio)
        max_abs_all.append(float(np.max(np.abs(data))))

    out_df = df[["subject_id", "label"]].copy()
    out_df["n_windows"] = n_windows_all
    out_df["n_clean_windows"] = n_clean_windows_all
    out_df["bad_window_ratio"] = bad_ratio_all
    out_df["max_abs"] = max_abs_all
    return out_df, np.stack(features)


def write_json(path: Path, payload: dict[str, object]) -> None:
    """把实验结果写成 JSON 文件。

    ensure_ascii=False 允许中文正常保存。
    indent=2 方便人工阅读和检查。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
