# Dual-Graph 路线问题与解决方案记录

## 文档目的
本文件专门记录 Dual-Graph（PCC + PLV）路线中遇到的问题、定位过程、解决方案与结果变化，供论文“方法改进与消融分析”章节直接引用。

## 一、问题背景
在单图融合（`alpha * PCC + (1-alpha) * PLV`）方案中，模型效果提升有限，且解释性较弱。为此引入 Dual-Graph：
- 一路使用 signed PCC 图
- 一路使用 PLV 图
- 在模型内部学习融合权重（而不是手工压缩边权）

## 二、发现的问题
### 问题1：手工边权压缩会损失信息
- 单图融合把两种连接信息压成一条边权，容易掩盖 PCC 的正负相关结构。
- PLV 与 PCC 在生理意义上互补，强行合成一个权重会降低模型可学习空间。

### 问题2：宽频 PLV 的信息混杂
- broad PLV 把多个频段的同步信息混在一起，可能把有效差异“平均掉”。
- 出现 AUC 有改善但 Acc/BalAcc/F1 提升有限的现象。

### 问题3：图结构稳定性不足（后续发现）
- 使用全局分位数阈值选边时，不同被试图拓扑差异较大，fold 间波动明显。

## 三、根因分析
1. 融合层级不合理：
- 手工边权融合属于“输入级强约束”，模型无法按样本自适应选择 PCC 与 PLV 的贡献。

2. 频段粒度不足：
- broad PLV 没有显式区分 theta/alpha/beta 的异质性贡献。

3. 选边策略过于全局：
- 全局阈值只控制“总边数”，不保证每个节点邻域质量。

## 四、解决方案
### 方案A：引入 Dual-Graph 模型内融合
- 新增 `train_gnn_dualgraph_cv10.py`
- 两个编码器分别提取 `PCC` 和 `PLV` 表征
- 使用 gate（softmax）学习每个样本的融合权重

### 方案B：从 broad 扩展到多频段动态融合
- 新增 `train_gnn_dualgraph_multiband_cv10.py`
- 四路输入：`PCC + PLV(theta/alpha/beta)`
- gate 在样本级动态分配四路权重

### 方案C：优化选边策略
- 构图从 `global_quantile` 改为 `per_node_topk`
- 对 signed 图保留正/负边（`signed_topk_split`）
- 目标：提高图结构稳定性与信息保真度

## 五、结果变化（10-fold）
### 1) Dual-Graph broad vs 单图 broad
- Dual-Graph broad：Acc `0.8167`，BalAcc `0.8000`，F1 `0.8183`，AUC `0.8111`
- 相比单图 broad，AUC 有提升，说明排序能力更强。

### 2) Dual-Graph multiband vs Dual-Graph broad
- Acc：`0.8167 -> 0.8357`
- BalAcc：`0.8000 -> 0.8250`
- F1：`0.8183 -> 0.8373`
- AUC：`0.8111 -> 0.8556`

解释：分频段后，模型能按样本动态选择更有判别力的频段信息。

### 3) Dual-Graph multiband + top-k vs Dual-Graph multiband
- Acc：`0.8357 -> 0.8524`
- BalAcc：`0.8250 -> 0.8333`
- F1：`0.8373 -> 0.8695`
- AUC：`0.8556 -> 0.8708`
- Acc 标准差：`0.1057 -> 0.0497`

解释：top-k + signed split 提高了图拓扑稳定性，降低了 fold 方差。

## 六、当前结论
1. Dual-Graph 相比手工边权压缩更合理，信息保留和可解释性更好。
2. 多频段动态融合是 Dual-Graph 路线的关键增益来源。
3. 选边策略（per-node top-k + signed split）是稳定性与性能提升的重要因素。
4. 当前最佳实践：
- 模型：Dual-Graph Multiband（PCC + theta/alpha/beta）
- 构图：per-node top-k + signed split

## 七、仍待优化的问题
1. 在新增数据集后需要验证跨数据源泛化能力。
2. 需在该最佳结构上继续验证动态阈值策略（固定0.5 vs val_balacc）。
3. 可进一步将 PLV 替换/补充为 wPLI/dwPLI，降低体积传导影响。

## 八、关联文件
- `src/train_gnn_dualgraph_cv10.py`
- `src/train_gnn_dualgraph_multiband_cv10.py`
- `src/build_graphs.py`
- `docs/benchmark_summary.md`
- `docs/topk_edge_strategy_issue_resolution.md`

