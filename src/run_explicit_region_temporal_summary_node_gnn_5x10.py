from __future__ import annotations

"""
0.9143 clean 主模型的入口脚本。

模型名称：
    ExplicitRegionTemporalWeightedStarGNN

这个脚本做的事情：
1. 读取静态图特征：
   - 节点频带功率。
   - PCC / theta PLV / alpha PLV / beta PLV 连接特征。
2. 读取时间摘要特征：
   - 把 EEG 切成多个 8 秒窗口。
   - 每个窗口提取节点频带功率。
   - 对窗口序列做 mean / std / delta 摘要。
3. 显式构建脑区节点：
   - frontal / central / temporal / parietal / occipital。
   - left / right / midline / cross_region / global。
4. 显式构建时间组节点：
   - temporal_mean / temporal_std / temporal_delta。
5. 构建图结构：
   - feature node 连接到 region node。
   - temporal feature 连接到 temporal group node。
   - region node 和 temporal group node 连接到 global node。
6. 进行 5 seeds x 10 folds 交叉验证。


- 每个受试者最终会变成一张“特征图”。
- 图里的节点不是原始像素，而是 EEG 特征、脑区聚合值和时间摘要值。
- 模型输出两个分数：正常类分数和抑郁类分数。
- softmax 后取抑郁类概率，再计算分类指标。

重要说明：
这个 clean 主模型使用 weighted-star 图读出。
它保留了显式图节点和图结构，但读出层是轻量、可解释的节点加权形式，
适合 61 人小样本实验。
"""

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

# 默认 EEG 通道顺序。
# 后续所有特征维度和脑区映射都依赖这个顺序。
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
    """标准化通道名。

    原始 EDF 中通道名可能写成：
    - "EEG F3-LE"
    - "F3-REF"
    - "F3"

    这个函数把它们统一成类似 "F3" 的形式，方便判断属于哪个脑区。
    """

    s = str(name).upper().strip()
    if s.startswith("EEG "):
        s = s[4:]
    for suffix in ["-LE", "-REF"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s.strip()


def channel_regions(name: str) -> list[int]:
    """根据 EEG 通道名判断它属于哪些脑区节点。

    一个通道通常同时属于多个区域概念：
    - 按前缀判断主脑区，例如 F3 属于 frontal。
    - 按尾号判断左右半球，例如 F3 属于 left。
    - 所有通道都属于 global。

    返回的是 REGION_KEYS 中的下标，而不是字符串。
    """

    ch = _norm_ch_name(name)
    regions: list[str] = []

    # 第一步：根据通道前缀判断前后位置。
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

    # 第二步：根据尾号判断左右半球或中线。
    if ch.endswith(("1", "3", "5", "7")):
        regions.append("left")
    elif ch.endswith(("2", "4", "6", "8")):
        regions.append("right")
    else:
        regions.append("midline")

    # 所有通道都归入 global，表示全局脑网络贡献。
    regions.append("global")

    # dict.fromkeys 用来去重，同时保留顺序。
    return [REGION_KEYS.index(r) for r in dict.fromkeys(regions)]


def base_feature_region_map(ch_names: list[str], n_bands: int = 5) -> list[list[int]]:
    """为 1034 个静态特征建立“特征 -> 脑区”映射。

    静态特征由两部分组成：
    1. 节点频带功率：
       - 22 个通道 x 5 个频段 = 110 个特征。
       - 每个特征直接继承所属通道的脑区。
    2. 连接特征：
       - PCC / theta PLV / alpha PLV / beta PLV 四个连接矩阵。
       - 每个连接特征继承两端通道的脑区。
       - 如果两端通道属于不同主脑区，就额外标记 cross_region。
    """

    n_channels = len(ch_names)
    feature_regions: list[list[int]] = []
    channel_map = [channel_regions(ch) for ch in ch_names]

    for channel_idx in range(n_channels):
        for _ in range(n_bands):
            # 前 110 个特征：每个通道的 5 个频段功率。
            feature_regions.append(channel_map[channel_idx])

    tri_i, tri_j = np.triu_indices(n_channels, k=1)
    for _band_block in range(4):
        for i, j in zip(tri_i, tri_j):
            # 后面的连接特征：每条边由两个通道共同决定脑区归属。
            regs = list(channel_map[int(i)]) + list(channel_map[int(j)])
            primary_i = {REGION_KEYS[r] for r in channel_map[int(i)] if REGION_KEYS[r] in {"frontal", "central", "temporal", "parietal", "occipital"}}
            primary_j = {REGION_KEYS[r] for r in channel_map[int(j)] if REGION_KEYS[r] in {"frontal", "central", "temporal", "parietal", "occipital"}}
            if primary_i != primary_j:
                # 两个端点位于不同主脑区时，认为这条边是跨脑区连接。
                regs.append(REGION_KEYS.index("cross_region"))
            feature_regions.append(sorted(set(regs)))
    return feature_regions


def node_feature_region_map(ch_names: list[str], n_bands: int = 5) -> list[list[int]]:
    """为节点频带功率类特征建立脑区映射。

    时间摘要特征只来自每个窗口的节点频带功率，
    所以它使用和前 110 个静态节点特征相同的映射方式。
    """

    feature_regions: list[list[int]] = []
    channel_map = [channel_regions(ch) for ch in ch_names]
    for channel_idx in range(len(ch_names)):
        for _ in range(n_bands):
            feature_regions.append(channel_map[channel_idx])
    return feature_regions


def combined_feature_region_map(base_map: list[list[int]], temporal_map: list[list[int]] | None) -> list[list[int]]:
    """拼接静态特征映射和时间特征映射。"""

    out = list(base_map)
    if temporal_map is not None:
        out.extend(temporal_map)
    return out


def build_region_aggregates(x: np.ndarray, feature_region_map: list[list[int]]) -> np.ndarray:
    """根据特征到脑区的映射，计算 10 个脑区节点值。

    x 的形状：
    - 行：受试者。
    - 列：feature nodes。

    输出形状：
    - 行：受试者。
    - 列：10 个 region nodes。

    举例：
    如果某个特征同时属于 frontal、left、global，
    那它的值会平均分到这三个脑区节点中。
    """

    out = np.zeros((x.shape[0], len(REGION_KEYS)), dtype=np.float32)
    for feat_idx, regions in enumerate(feature_region_map):
        if not regions:
            continue

        # 平均分摊，避免一个特征因为映射到多个脑区而被重复放大。
        scale = 1.0 / float(len(regions))
        for region_idx in regions:
            out[:, int(region_idx)] += x[:, feat_idx] * scale
    return out


def build_temporal_group_aggregates(temporal_x: np.ndarray) -> np.ndarray:
    """把 330 维时间摘要压缩成 3 个时间组节点。

    temporal_x 的 330 维顺序固定为：
    - 0:110     temporal_mean
    - 110:220   temporal_std
    - 220:330   temporal_delta
    """

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
    """构建最终送入模型的节点值表。

    最终节点值由三部分组成：
    1. base_x：
       - static_x 1034 维。
       - temporal_x 330 维。
       - 合计 1364 个 feature nodes。
    2. region_x：
       - 10 个脑区节点。
    3. temporal_group_x：
       - 3 个时间组节点。

    expanded_x 总维度：
    1364 + 10 + 3 = 1377。

    注意：global node 不在 expanded_x 里，而是在 Dataset 中额外补一个 0。
    """

    base_x = np.concatenate([static_x, temporal_x], axis=1).astype(np.float32)
    region_x = build_region_aggregates(base_x, feature_region_map)
    temporal_group_x = build_temporal_group_aggregates(temporal_x)
    expanded_x = np.concatenate([base_x, region_x, temporal_group_x], axis=1).astype(np.float32)
    slices = {
        # 记录每一段特征在 expanded_x 中的位置，后续用于切片。
        "feature": (0, int(base_x.shape[1])),
        "region": (int(base_x.shape[1]), int(base_x.shape[1] + region_x.shape[1])),
        "temporal_group": (int(base_x.shape[1] + region_x.shape[1]), int(expanded_x.shape[1])),
    }
    return expanded_x, slices


def build_graph_edges(feature_region_map: list[list[int]], n_feature_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """构建模型使用的显式图结构。

    节点顺序：
    1. feature nodes：0 到 n_feature_nodes-1。
    2. region nodes：紧跟在 feature nodes 后面。
    3. temporal group nodes：紧跟在 region nodes 后面。
    4. global node：最后一个节点。

    边的含义：
    - feature <-> region：特征属于哪些脑区。
    - temporal feature <-> temporal group：时间特征属于 mean/std/delta 哪一组。
    - region <-> global：脑区信息汇总到全局。
    - temporal group <-> global：时间组信息汇总到全局。
    """

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
            # 理论上不会出现；作为兜底，连到 global 区域。
            unique_regions = [REGION_KEYS.index("global")]
        scale = 1.0 / float(len(unique_regions))
        for region_idx in unique_regions:
            region_node = region_offset + region_idx

            # 双向边：feature -> region 和 region -> feature。
            src.extend([feature_idx, region_node])
            dst.extend([region_node, feature_idx])
            weights.extend([scale, scale])

    temporal_start = 1034
    temporal_block = 110
    for group_idx in range(n_temporal_groups):
        temporal_node = temporal_offset + group_idx

        # temporal_mean / temporal_std / temporal_delta 各占 110 维。
        start = temporal_start + (group_idx * temporal_block)
        stop = start + temporal_block
        for feature_idx in range(start, stop):
            # 时间特征节点连接到对应的时间组节点。
            src.extend([feature_idx, temporal_node])
            dst.extend([temporal_node, feature_idx])
            weights.extend([1.0, 1.0])

    for region_idx in range(n_regions):
        region_node = region_offset + region_idx

        # 脑区节点连到 global node，表示全局汇总。
        src.extend([region_node, global_node])
        dst.extend([global_node, region_node])
        weights.extend([1.0, 1.0])

    for group_idx in range(n_temporal_groups):
        temporal_node = temporal_offset + group_idx

        # 时间组节点也连到 global node。
        src.extend([temporal_node, global_node])
        dst.extend([global_node, temporal_node])
        weights.extend([1.0, 1.0])

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float32)
    return edge_index, edge_weight


class ExplicitRegionTemporalNodeDataset(Dataset):
    """把每个受试者包装成 PyTorch Geometric 的 Data 图。

    Dataset 的每一个样本就是一个受试者：
    - x：节点值。
    - edge_index：边。
    - edge_attr：边权重。
    - y：标签。
    - subject_id：受试者编号。

    所有受试者共享同一套图结构，因为节点含义完全一致；
    不同受试者之间变化的是每个节点的数值。
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, subject_ids: np.ndarray, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> None:
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.subject_ids = np.asarray(subject_ids)
        self.edge_index = edge_index
        self.edge_weight = edge_weight

        # expanded_x 有 1377 列；这里额外加 1 个 global node。
        self.n_nodes = int(self.x.shape[1]) + 1

        # node_id 用来标识每个节点的位置/身份。
        self.node_id = torch.arange(self.n_nodes, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Data:
        # 取出第 idx 个受试者的 1377 个节点值。
        # 末尾追加 0.0 作为 global node 的初始值。
        values = np.concatenate([self.x[idx], np.array([0.0], dtype=np.float32)], axis=0).reshape(-1, 1)

        # Data 是 PyG 的单图数据结构。
        return Data(
            x=torch.tensor(values, dtype=torch.float32),
            node_id=self.node_id.clone(),
            edge_index=self.edge_index.clone(),
            edge_attr=self.edge_weight.clone(),
            y=torch.tensor([int(self.y[idx])], dtype=torch.long),
            subject_id=str(self.subject_ids[idx]),
        )


class ExplicitRegionTemporalWeightedStarGNN(nn.Module):
    """显式脑区 + 时间摘要 weighted-star GNN。

    模型核心思想：
    - 每个节点都有一个二分类权重。
    - feature nodes、region nodes、temporal group nodes 分别贡献一部分 logit。
    - 三部分相加后得到 [正常, 抑郁] 两个类别分数。

    它不是多层 GCNConv/GATConv 的深层消息传递模型，
    而是图结构化节点读出模型：
    - 图节点和脑区/时间结构是显式的。
    - 分类读出是轻量线性形式，适合小样本并且容易解释。
    """

    def __init__(self, n_feature_nodes: int, n_region_nodes: int, n_temporal_group_nodes: int) -> None:
        super().__init__()
        self.n_feature_nodes = int(n_feature_nodes)
        self.n_region_nodes = int(n_region_nodes)
        self.n_temporal_group_nodes = int(n_temporal_group_nodes)
        self.n_nodes = self.n_feature_nodes + self.n_region_nodes + self.n_temporal_group_nodes + 1

        # 每个 feature node 对两个类别各有一个权重。
        # 形状为 [n_feature_nodes, 2]。
        self.feature_logits = nn.Embedding(self.n_feature_nodes, 2)

        # 每个 region node 对两个类别各有一个权重。
        self.region_logits = nn.Embedding(self.n_region_nodes, 2)

        # 每个 temporal group node 对两个类别各有一个权重。
        self.temporal_group_logits = nn.Embedding(self.n_temporal_group_nodes, 2)

        # 两个类别的全局偏置。
        self.bias = nn.Parameter(torch.zeros(2))

        # 先置零，后续从 LogisticRegression 加载参数。
        nn.init.zeros_(self.feature_logits.weight)
        nn.init.zeros_(self.region_logits.weight)
        nn.init.zeros_(self.temporal_group_logits.weight)

    def load_logistic_parameters(self, coef: np.ndarray, intercept: np.ndarray) -> None:
        """把 LogisticRegression 的参数加载成 GNN 节点权重。

        LogisticRegression 学到的是一条线性决策边界：
            logit = x @ coef + intercept

        这里把 coef 拆成三段：
        1. feature nodes 的权重。
        2. region nodes 的权重。
        3. temporal group nodes 的权重。

        然后写入模型的 Embedding 参数中。
        """

        coef = np.asarray(coef, dtype=np.float32).reshape(-1)
        intercept = np.asarray(intercept, dtype=np.float32).reshape(-1)
        if int(coef.shape[0]) != int(self.n_feature_nodes + self.n_region_nodes + self.n_temporal_group_nodes):
            raise ValueError("Coefficient dimension mismatch.")

        # 三类节点在 coef 中的切片边界。
        feat_end = self.n_feature_nodes
        reg_end = feat_end + self.n_region_nodes
        tmp_end = reg_end + self.n_temporal_group_nodes

        feat_w = torch.zeros((self.n_feature_nodes, 2), dtype=torch.float32)
        reg_w = torch.zeros((self.n_region_nodes, 2), dtype=torch.float32)
        tmp_w = torch.zeros((self.n_temporal_group_nodes, 2), dtype=torch.float32)
        # 二分类 LogisticRegression 只需要一个正类方向的系数。
        # 这里把它放到第 1 类，也就是 MDD 类 logit 上。
        # 第 0 类保持 0。
        feat_w[:, 1] = torch.tensor(coef[:feat_end], dtype=torch.float32)
        reg_w[:, 1] = torch.tensor(coef[feat_end:reg_end], dtype=torch.float32)
        tmp_w[:, 1] = torch.tensor(coef[reg_end:tmp_end], dtype=torch.float32)

        self.feature_logits.weight.data.copy_(feat_w)
        self.region_logits.weight.data.copy_(reg_w)
        self.temporal_group_logits.weight.data.copy_(tmp_w)
        self.bias.data.zero_()

        # intercept 同样写入第 1 类。
        self.bias.data[1] = float(intercept[0])

    def forward(self, data: Batch) -> torch.Tensor:
        """前向推理：从一批图得到二分类 logits。

        输入 data 是 PyG Batch：
        - 多个受试者的节点被拼接在一起。
        - data.batch 记录每个节点属于哪个受试者。

        输出形状：
        - [batch_size, 2]
        - 第 0 列：正常类 logit。
        - 第 1 列：抑郁类 logit。
        """

        # 计算 batch 中有多少个受试者图。
        batch_size = int(data.batch.max().item()) + 1 if data.batch.numel() else 0

        # 把展平的节点值恢复成 [batch_size, n_nodes]。
        values = data.x.view(batch_size, self.n_nodes)

        # 前 n_feature_nodes 个是普通特征节点。
        feature_values = values[:, : self.n_feature_nodes]
        region_start = self.n_feature_nodes
        region_stop = region_start + self.n_region_nodes
        temporal_start = region_stop
        temporal_stop = temporal_start + self.n_temporal_group_nodes
        # 中间一段是 10 个脑区节点。
        region_values = values[:, region_start:region_stop]

        # 再后一段是 3 个时间组节点。
        temporal_group_values = values[:, temporal_start:temporal_stop]

        # 三类节点分别矩阵乘自己的权重，得到二分类贡献。
        # 最后加上全局 bias。
        return (
            (feature_values @ self.feature_logits.weight)
            + (region_values @ self.region_logits.weight)
            + (temporal_group_values @ self.temporal_group_logits.weight)
            + self.bias
        )


def collate_graphs(batch: list[Data]) -> Batch:
    """DataLoader 的合并函数。

    PyG 需要用 Batch.from_data_list 把多个 Data 图拼成一个 Batch。
    """

    return Batch.from_data_list(batch)


@torch.no_grad()
def predict(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """用模型对 DataLoader 中的样本进行预测。

    返回：
    - y_all：真实标签。
    - p_all：模型预测为 MDD 的概率。
    """

    model.eval()
    y_all: list[np.ndarray] = []
    p_all: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)

        # softmax 把两个类别 logit 转换成概率。
        # [:, 1] 表示取第 1 类，也就是 MDD 的概率。
        prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        y_all.append(batch.y.view(-1).detach().cpu().numpy())
        p_all.append(prob)
    return np.concatenate(y_all), np.concatenate(p_all)


def parse_args() -> argparse.Namespace:
    """命令行参数。

    默认参数对应 clean 0.9143 主结果：
    - 61 人 TASK。
    - 5 个 seed。
    - 每个 seed 10 折。
    - 不启用 targeted repair。
    """

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
    """主流程入口。

    执行顺序：
    1. 解析参数和设备。
    2. 构建静态特征和时间摘要特征。
    3. 对齐受试者顺序。
    4. 构建显式脑区/时间组节点和图边。
    5. 执行 5 seeds x 10 folds 评估。
    6. 保存每个 seed 的结果和总 summary。
    """

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
        # targeted repair 版本：会根据坏窗口比例替换时间摘要。
        # clean 0.9143 默认不走这里。
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
        # clean 主结果使用普通时间摘要。
        temporal_meta, temporal_x = build_temporal_summary_table(
            split_csv=args.split_csv,
            window_seconds=args.window_seconds,
            step_seconds=args.step_seconds,
            max_windows=args.max_windows,
            max_seconds=args.max_seconds,
        )
    merged = static_meta.merge(temporal_meta, on=["subject_id", "label"], how="inner").sort_values("subject_id").reset_index(drop=True)
    if int(len(merged)) != int(len(static_meta)) or int(len(merged)) != int(len(temporal_meta)):
        # 如果静态特征和时间特征不是同一批受试者，直接报错，避免错位训练。
        raise ValueError("Static and temporal tables did not align on identical subject sets.")

    static_index = {sid: idx for idx, sid in enumerate(static_meta["subject_id"].tolist())}
    temporal_index = {sid: idx for idx, sid in enumerate(temporal_meta["subject_id"].tolist())}
    subject_order = merged["subject_id"].tolist()

    # 按 merged 的 subject_id 顺序重新排列两个特征矩阵，保证行对齐。
    static_x = np.stack([static_x[static_index[sid]] for sid in subject_order]).astype(np.float32)
    temporal_x = np.stack([temporal_x[temporal_index[sid]] for sid in subject_order]).astype(np.float32)
    bad_window_ratio = merged["bad_window_ratio"].to_numpy(dtype=np.float32) if "bad_window_ratio" in merged.columns else None

    static_clean_enabled = args.targeted_static_clean_bad_window_ratio is not None
    if static_clean_enabled:
        # targeted repair：坏窗口比例高的受试者，用干净窗口静态特征替换。
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
        # targeted repair：坏窗口比例高的受试者，将 temporal 分支置零。
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

    # 时间摘要有 mean/std/delta 三段，每段都对应同一套通道频带映射。
    temporal_map = node_map + node_map + node_map
    feature_region_map = combined_feature_region_map(base_map, temporal_map)

    expanded_x, slices = build_expanded_feature_table(static_x, temporal_x, feature_region_map)

    # 根据 slices 计算三类节点数量。
    n_feature_nodes = int(slices["feature"][1] - slices["feature"][0])
    n_region_nodes = int(slices["region"][1] - slices["region"][0])
    n_temporal_group_nodes = int(slices["temporal_group"][1] - slices["temporal_group"][0])
    edge_index, edge_weight = build_graph_edges(feature_region_map, n_feature_nodes=n_feature_nodes)

    run_payloads: list[dict[str, object]] = []
    for seed in seeds:
        # 每个 seed 都重新生成 10 折划分，用来评估稳定性。
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=int(seed))
        fold_results: list[dict[str, object]] = []

        for fold_no, (train_idx, test_idx) in enumerate(skf.split(expanded_x, y), start=1):
            # 当前折的训练集和测试集。
            train_x = expanded_x[train_idx]
            test_x = expanded_x[test_idx]
            train_y = y[train_idx]
            test_y = y[test_idx]

            solver = Pipeline(
                [
                    # 标准化只能在训练集上 fit，再应用到测试集，避免数据泄漏。
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=5000,

                            # 类别权重平衡，缓解 HC/MDD 数量不完全一致的问题。
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

            # 用训练集学到的 scaler 转换训练和测试特征。
            train_z = scaler.transform(train_x).astype(np.float32)
            test_z = scaler.transform(test_x).astype(np.float32)

            model = ExplicitRegionTemporalWeightedStarGNN(
                n_feature_nodes=n_feature_nodes,
                n_region_nodes=n_region_nodes,
                n_temporal_group_nodes=n_temporal_group_nodes,
            ).to(device)

            # 把 LogisticRegression 参数装进 weighted-star GNN。
            model.load_logistic_parameters(clf.coef_[0], clf.intercept_)

            test_ds = ExplicitRegionTemporalNodeDataset(
                x=test_z,
                y=test_y,
                subject_ids=subject_ids[test_idx],
                edge_index=edge_index,
                edge_weight=edge_weight,
            )
            test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_graphs)

            # 预测测试折，并用固定阈值 0.5 计算指标。
            y_pred, prob_pred = predict(model, test_loader, device=device)
            metrics = calc_metrics(y_pred, prob_pred, threshold=0.5)

            # 保存脑区和时间组系数，方便后续解释模型关注点。
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
                # 记录模型实际节点数量，便于检查输入维度。
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

    # summary_5x10.json 是最终汇总文件，README 和 benchmark 表格中的 0.9143 来自这里。
    write_json(summary_path, summary_payload)
    print(f"\nsummary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
