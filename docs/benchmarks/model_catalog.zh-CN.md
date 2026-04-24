# 模型目录

[English](model_catalog.md) | 简体中文

这个文件是 `src/` 下模型入口的清晰登记表，用来区分稳定基准、对照脚本、探索路线和底层模块。

## 1. 稳定基准入口
这些脚本构成当前仓库里最干净、最适合论文叙事的基准主线。

| 层级 | 家族 | 脚本 | 范围 | 当前最佳结果 | 说明 |
|---|---|---|---|---|---|
| Stable | 静态纯 GNN | `src/run_static_feature_node_gnn_5x10.py` | `61` 人，`TASK` | `0.9010` acc | 完整主基准上的静态纯 GNN 基线 |
| Stable | 显式脑区 + 时间摘要纯 GNN | `src/run_explicit_region_temporal_summary_node_gnn_5x10.py` | `61` 人，`TASK` | `0.9143` acc | 当前 clean 的完整数据集主力纯 GNN 结果 |
| Stable | 多状态显式脑区时空纯 GNN | `src/run_multistate_explicit_region_temporal_node_gnn_5x10.py` | 完整状态子集 | `0.9260` acc on `TASK+EC+EO` | 完整状态子集上的干净多状态基线 |
| Stable | 多状态 QC 显式脑区时空纯 GNN | `src/run_multistate_explicit_region_temporal_node_gnn_5x10.py` 加 QC 参数 | QC 过滤完整状态子集 | `0.9750` acc on `38` 人 | 当前最强多状态路线；使用客观伪迹过滤 (`max_abs` + impulsive-window rule) |

## 2. Benchmark-tuned 的完整数据集变体
这些结果仍然可复现，也有工程价值，但因为用了针对当前数据集调过的修复规则，benchmark-overfitting 风险更高，所以要和 clean 基准分开汇报。

| 层级 | 家族 | 脚本 | 范围 | 当前最佳结果 | 说明 |
|---|---|---|---|---|---|
| Benchmark-tuned | 显式脑区 + 时间摘要纯 GNN + 定向修复 | `src/run_explicit_region_temporal_summary_node_gnn_5x10.py`，配合 targeted repair 参数 | `61` 人，`TASK` | `0.9343` acc | 对最差受试者做 targeted temporal cleaning、static clean replacement 和 temporal zero repair |

## 3. 对照与历史参考脚本
这些脚本适合做受控比较或保留历史记录，但不是当前论文主叙事里的纯 GNN 主线。

| 层级 | 家族 | 脚本 | 状态 | 说明 |
|---|---|---|---|---|
| Comparator | 早期非 GNN 基线 | `src/train_baseline.py` | Historical | 早期统计特征基线和受试者级 `TASK` 划分 |
| Comparator | 静态 top-k GNN | `src/run_static_topk_5x10.py` | Historical | 更早的静态 top-k 路线 |
| Comparator | 静态 graph-feature residual | `src/run_static_gnn_graphfeature_residual_5x10.py` | Reference only | residual 图特征变体 |
| Comparator | 静态 graph-vector hybrid | `src/run_static_gnn_graphvector_hybrid_5x10.py` | Reference only | 包含非纯 GNN readout 的混合对照 |
| Comparator | 静态 flat-readout hybrid | `src/run_static_flatreadout_graphvector_hybrid_5x10.py` | Reference only | 适合做消融，不是主纯 GNN 口径 |
| Comparator | 历史 benchmark wrapper | `src/run_benchmarks.py` | Reference only | 汇总更早期的 benchmark 路线 |

## 4. 探索性纯 GNN 路线
这些是当前仍在推进的研究脚本，应该保留，但不应默认作为主结果。

| 层级 | 家族 | 脚本 | 当前状态 | 说明 |
|---|---|---|---|---|
| Exploratory | Region-temporal node GNN | `src/run_region_temporal_feature_node_gnn_5x10.py` | Below promoted line | 更早期的显式脑区路线 |
| Exploratory | Dynamics weighted-star GNN | `src/run_explicit_region_temporal_dynamics_node_gnn_5x10.py` | Below promoted line | 增加动态摘要与验证搜索 |
| Exploratory | Dynamics graph-conv GNN | `src/run_explicit_region_temporal_dynamics_graphconv_5x10.py` | Probe stage | 更像真图卷积的版本，目前更弱 |
| Exploratory | Prior-adaptive region-temporal GNN | `src/run_prior_adaptive_regiontemporal_gnn_5x10.py` | Probe stage | 带脑区先验的路线 |
| Exploratory | Multistate region-graph GNN | `src/run_multistate_regiongraph_gnn_5x10.py` | Probe stage | 多状态小图区域 GNN |
| Exploratory | Temporal variant pack | `src/run_temporal_variants_5x10.py` | Probe stage | 时间摘要相关消融合集 |
| Exploratory | Spatiotemporal PYC route | `src/run_spatiotemporal_pyc_5x10.py` | Probe stage | 单次时空路线 |
| Exploratory | Spatiotemporal PYC multi-init | `src/run_spatiotemporal_pyc_multiinit_5x10.py` | Probe stage | 多初始化时空版本 |
| Exploratory | Spatiotemporal region-GAT | `src/run_spatiotemporal_regiongat_multiinit_5x10.py` | Probe stage | 脑区感知注意力路线 |

## 5. 集成和选择类脚本
这些脚本适合搜索、集成和探测，但不能被当成单一模型 benchmark。

| 层级 | 家族 | 脚本 | 说明 |
|---|---|---|---|
| Research utility | Pure-GNN ensemble | `src/run_pure_gnn_ensemble_cv10.py` | 集成路线，不是单模型主结果 |
| Research utility | Validation selector | `src/run_region_temporal_validation_selector_5x10.py` | 折级选择器，不是单模型 |
| Research utility | Flatreadout multi-init ensemble | `src/run_flatreadout_multiinit_ensemble_cv10.py` | 搜索/集成辅助 |
| Research utility | Flatwin multi-init ensemble | `src/run_flatwin_multiinit_ensemble_cv10.py` | 搜索/集成辅助 |
| Research utility | Spatiotemporal snapshot ensemble | `src/run_spatiotemporal_snapshot_ensemble_cv10.py` | 搜索/集成辅助 |
| Research utility | Spatiotemporal multi-init ensemble | `src/run_spatiotemporal_multiinit_ensemble_cv10.py` | 搜索/集成辅助 |

## 6. 底层训练模块
这些更像构件，而不是建议直接汇报的顶层 benchmark 脚本。

| 层级 | 家族 | 脚本 |
|---|---|---|
| Core modules | 图构建与工具 | `src/build_graphs.py`, `src/feature_model_utils.py` |
| Legacy training entry points | 静态 / multiband / dual-graph / spatiotemporal trainers | `src/train_gnn*.py`, `src/train_nongnn_cv10.py` |

## 7. 汇报约定
- 当报告必须保持在完整 `61` 人主基准上时，默认使用不带 targeted repair 参数的 `run_explicit_region_temporal_summary_node_gnn_5x10.py`。
- 如果使用 `clean250 + static_clean>=0.33 + temporal_zero>=0.33` 这一组 targeted repair 设置，必须单独标注为 benchmark-tuned 的伪迹感知修复版；结果目录在 `outputs/metrics/runs/explicit_region_temporal_summary_targeted_clean250_staticclean033_tempzero033_c025_5x10/`。
- 只有在报告明确说明是完整状态多状态子集实验时，才使用 `run_multistate_explicit_region_temporal_node_gnn_5x10.py`。
- 汇报最强多状态结果时，要同时给出 QC 阈值和过滤后的受试者数量，因为 `0.9750` 是质量控制后的子集结果，不是原始 `53` 人完整状态池。
- 不要把 selector 或 ensemble 脚本包装成单一纯 GNN benchmark 模型。
