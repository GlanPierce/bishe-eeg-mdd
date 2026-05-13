# Top-k 选边优化问题记录（独立文档）

## 文档目的
这份文档用于记录本轮实验中“发现的问题、解决的问题、解决方法和结果变化”，供论文写作直接引用。

## 背景
在 `Dual-Graph Multiband（PCC + PLV theta/alpha/beta）` 路线中，初始构图采用“全局分位数阈值（global quantile）”选边。

## 发现的问题
1. 图结构稳定性不足。
- 全局分位数是全图统一阈值，不同被试的连接分布不同，会导致每个节点实际保留边数差异很大。
- 结果是某些节点过稀疏、某些节点过密，图拓扑不均衡。

2. signed 信息利用不完整。
- 在 PCC 场景下，负相关边可能有判别价值。
- 若只按统一强度筛边，容易让正边主导，负边保留不稳定。

3. 指标方差偏大。
- 在 10-fold 评估中，Acc/BalAcc/F1 跨 fold 波动较明显，说明图构建对数据划分较敏感。

## 根因分析
- “全局阈值”本质上是样本级分布敏感策略：只保证全图边总量，不保证每个节点邻域质量。
- 对小样本 EEG 数据，节点局部结构稳定性往往比全局边密度更关键。

## 解决方案
将选边策略从 `global_quantile` 改为 `per_node_topk`，并在 signed 图上分正负边分别保留：

1. 每个节点 top-k（per-node top-k）
- 每个节点单独保留最强的 k 条连接，保证局部邻域覆盖。

2. 正负边分开保留（signed_topk_split）
- 对 signed 权重（如 PCC/fused signed）分别保留 top-k 正边和 top-k 负边。
- 避免负相关信息被正边“挤掉”。

3. 对非负连接（PLV）保持 top-k
- PLV 本身非负，不做正负拆分，直接 per-node top-k。

## 实施细节
### 代码改动
- 构图脚本新增参数与逻辑：
  - `--edge-selection {global_quantile, per_node_topk}`
  - `--top-k-per-node`
  - `--signed-topk-split / --no-signed-topk-split`
- 新增 `per_node_topk` 的无向去重选边实现。
- 在 `pcc / plv / pcc_plv` 三种 edge_mode 下统一接入。

### 使用命令（本轮采用）
```powershell
# PCC: per-node top-k + signed split
.\.venv\Scripts\python.exe .\src\build_graphs.py --edge-mode pcc --edge-selection per_node_topk --top-k-per-node 4 --signed-topk-split --out-dir data/processed/graphs_pcc_topk_task

# PLV bands: per-node top-k
.\.venv\Scripts\python.exe .\src\build_graphs.py --edge-mode plv --plv-band theta --edge-selection per_node_topk --top-k-per-node 4 --out-dir data/processed/graphs_plv_theta_topk_task
.\.venv\Scripts\python.exe .\src\build_graphs.py --edge-mode plv --plv-band alpha --edge-selection per_node_topk --top-k-per-node 4 --out-dir data/processed/graphs_plv_alpha_topk_task
.\.venv\Scripts\python.exe .\src\build_graphs.py --edge-mode plv --plv-band beta  --edge-selection per_node_topk --top-k-per-node 4 --out-dir data/processed/graphs_plv_beta_topk_task

# 训练多频段动态融合模型
.\.venv\Scripts\python.exe .\src\train_gnn_dualgraph_multiband_cv10.py --pcc-manifest data/processed/graphs_pcc_topk_task/manifest.csv --theta-manifest data/processed/graphs_plv_theta_topk_task/manifest.csv --alpha-manifest data/processed/graphs_plv_alpha_topk_task/manifest.csv --beta-manifest data/processed/graphs_plv_beta_topk_task/manifest.csv --out-path outputs/metrics/runs/dualgraph_multiband_topk_full_cv10/gnn_dualgraph_multiband_topk_cv10.json --experiment-name gnn_dualgraph_multiband_topk_cv10
```

## 结果对比（10-fold）
对比对象：同一模型结构下，`global quantile` vs `per-node top-k + signed split`

- Accuracy: `0.8357 -> 0.8524`（+0.0167）
- Balanced Accuracy: `0.8250 -> 0.8333`（+0.0083）
- F1: `0.8373 -> 0.8695`（+0.0322）
- ROC-AUC: `0.8556 -> 0.8708`（+0.0152）
- Acc 标准差: `0.1057 -> 0.0497`（显著下降）

结论：
- 该优化不仅提升均值，还明显降低波动，说明图结构更稳定。
- 对小样本 EEG 分类任务，这种“节点级约束 + signed 信息保留”是有效的结构性改进。

## 当前仍存在的问题
1. 与部分非 GNN 基线相比，某些指标仍有差距。
2. 当前阈值仍固定 0.5，尚未在“扩充数据后”系统验证动态阈值策略。
3. 连接指标仍可升级（如 PLV -> wPLI/dwPLI），进一步提高抗体积传导能力。

## 下一步建议
1. 在新增数据后，优先验证 `dynamic threshold` 是否能把 AUC 优势转成更高 Acc/BalAcc。
2. 保持 `per-node top-k + signed split` 不变，替换 PLV 为 wPLI/dwPLI 做对照。
3. 增加重复实验（不同随机种子）评估稳定性。

## 关联文档
- `docs/benchmarks/benchmark_summary.md`：总表与横向对比
- `docs/notes/plv_band_experiment_notes.md`：PLV 相关问题记录

