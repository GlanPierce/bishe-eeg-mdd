# 源码索引

[English](README.md) | 简体中文

`src/` 里的代码可以分成三层：

1. 稳定基准入口
2. 探索性 benchmark 脚本
3. 底层训练与图构建模块

如果你只关心主模型，从第一部分开始即可，后面的可以先忽略。

## 稳定基准入口
- `build_graphs.py`
  - 构建完整 `TASK` 主基准所需的标准图输入
- `run_static_feature_node_gnn_5x10.py`
  - `61` 人主基准上的稳定 clean 静态纯 GNN 基线（`0.9010`）
- `run_explicit_region_temporal_summary_node_gnn_5x10.py`
  - 不带 targeted repair 参数时，它是完整 `61` 人主基准上的 clean 主力纯 GNN 入口（`0.9143`）
  - 同一个脚本也支持 `--targeted-clean-*`、`--targeted-static-clean-*` 和 `--targeted-temporal-zero-*` 这组受试者级伪迹修复参数；对应的是单独汇报的 benchmark-tuned 版本（`0.9343`）
- `run_multistate_explicit_region_temporal_node_gnn_5x10.py`
  - 面向完整状态子集（如 `TASK+EC+EO`）的多状态纯 GNN 入口
  - 同时支持 `--qc-*` 客观伪迹过滤参数；当前最强多状态结果就是这条路线

## 对照或历史参考脚本
- `train_baseline.py`
  - 早期非 GNN 基线和数据划分生成
- `run_static_topk_5x10.py`
  - 更早期的静态 top-k 路线
- `run_static_gnn_graphfeature_residual_5x10.py`
  - graph-feature residual 对照
- `run_static_gnn_graphvector_hybrid_5x10.py`
  - 静态 hybrid 对照
- `run_static_flatreadout_graphvector_hybrid_5x10.py`
  - 静态 flat-readout 对照
- `run_benchmarks.py`
  - 历史 benchmark wrapper

## 探索性纯 GNN 脚本
- `run_region_temporal_feature_node_gnn_5x10.py`
- `run_explicit_region_temporal_dynamics_node_gnn_5x10.py`
- `run_explicit_region_temporal_dynamics_graphconv_5x10.py`
- `run_prior_adaptive_regiontemporal_gnn_5x10.py`
- `run_multistate_regiongraph_gnn_5x10.py`
- `run_temporal_variants_5x10.py`
- `run_spatiotemporal_pyc_5x10.py`
- `run_spatiotemporal_pyc_multiinit_5x10.py`
- `run_spatiotemporal_regiongat_multiinit_5x10.py`

这些脚本值得保留，但仍属于研究推进路线，不是当前最干净的 benchmark 主线。

## 集成和选择脚本
- `run_pure_gnn_ensemble_cv10.py`
- `run_region_temporal_validation_selector_5x10.py`
- `run_flatreadout_multiinit_ensemble_cv10.py`
- `run_flatwin_multiinit_ensemble_cv10.py`
- `run_spatiotemporal_snapshot_ensemble_cv10.py`
- `run_spatiotemporal_multiinit_ensemble_cv10.py`

把这些当成搜索、探测或集成工具即可，不要默认当成单模型汇报线。

## 底层训练模块
- `train_gnn.py`
- `train_gnn_cv10.py`
- `train_gnn_multiband_cv10.py`
- `train_gnn_dualgraph_cv10.py`
- `train_gnn_dualgraph_multiband_cv10.py`
- `train_gnn_dualgraph_multiband_flatreadout_cv10.py`
- `train_gnn_dualgraph_multiband_regionprior_flatreadout_cv10.py`
- `train_gnn_dualgraph_multiband_regiontemporal_cv10.py`
- `train_gnn_dualgraph_multiband_spatiotemporal_cv10.py`
- `train_gnn_dualgraph_multiband_spatiotemporal_flatwin_cv10.py`
- `train_gnn_dualgraph_multiband_spatiotemporal_hybridwin_cv10.py`
- `train_gnn_dualgraph_multiband_stagewise_cv10.py`
- `train_gnn_spatiotemporal_cv10.py`
- `train_nongnn_cv10.py`

这些是实现模块或更早期的训练入口。只有在你明确要复现旧路线或在其上搭新 benchmark 脚本时才需要直接使用。

## 汇报边界
- 当汇报必须保持在完整 `61` 人主基准上时，默认使用不带 targeted repair 参数的 `run_explicit_region_temporal_summary_node_gnn_5x10.py`。
- 如果使用 `clean250 + static_clean>=0.33 + temporal_zero>=0.33` 这一组 targeted repair 设置，必须单独标注为 benchmark-tuned 的伪迹感知修复版，而不是 clean 主基准。
- 只有在报告明确说明是完整状态多状态子集实验时，才使用 `run_multistate_explicit_region_temporal_node_gnn_5x10.py`。
- 不要把 selector 或 ensemble 脚本包装成单一纯 GNN benchmark 模型。
