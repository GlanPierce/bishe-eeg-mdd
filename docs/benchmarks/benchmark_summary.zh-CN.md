# 基准摘要

[English](benchmark_summary.md) | 简体中文

## 范围说明
这个仓库现在有两类基准，不能混在一起汇报：

1. 完整数据集 `TASK` 主基准：`61` 人，是论文主对比线
2. 完整状态多状态基准：只保留同时具备 `TASK / EC / EO` 的受试者

多状态模型分数更高，但样本子集更小，因此不能直接与 `61` 人主基准比较。

## A. 61 人主基准

### 数据与协议
- 受试者：`61`（`HC=28`，`MDD=33`）
- 输入范围：仅 `TASK`
- 划分方式：受试者级分层交叉验证
- 稳定性报告：`5 seeds x 10 folds`

### 推荐模型
| 方法 | 脚本 | Accuracy | Balanced Accuracy | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|
| 静态纯 GNN (`FeatureNodeGNN`) | `src/run_static_feature_node_gnn_5x10.py` | `0.9010 +/- 0.0106` | `0.8950 +/- 0.0081` | `0.9066 +/- 0.0138` | `0.9456 +/- 0.0227` |
| 显式脑区 + 时间摘要纯 GNN (`ExplicitRegionTemporalWeightedStarGNN`) + 定向伪迹修复 | `src/run_explicit_region_temporal_summary_node_gnn_5x10.py` | **`0.9343 +/- 0.0012`** | **`0.9283 +/- 0.0041`** | **`0.9412 +/- 0.0021`** | **`0.9581 +/- 0.0190`** |

### 当前推荐的 61 人参数
- `--c 0.25`
- `--targeted-clean-bad-abs-threshold 0.00025`
- `--targeted-clean-bad-window-ratio 0.16`
- `--targeted-clean-summary-abs-threshold 0.00025`
- `--targeted-static-clean-bad-window-ratio 0.33`
- `--targeted-temporal-zero-bad-window-ratio 0.33`
- 这条线保留全部 `61` 名受试者，只对最差的 `6` 人做修复：静态特征替换成 clean-window 平均值，同时把 temporal 分支置零。

### 历史参考
| 方法 | 验证设置 | Accuracy | Balanced Accuracy | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|
| 静态 dual-graph multiband + top-k + signed split | `10-fold` | `0.8524 +/- 0.0497` | `0.8333 +/- 0.0645` | `0.8695 +/- 0.0533` | `0.8708 +/- 0.1491` |
| 时空 dual-graph (`16/8/12`) | `10-fold` | `0.8690 +/- 0.1243` | `0.8667 +/- 0.1247` | `0.8794 +/- 0.1226` | `0.8667 +/- 0.1296` |

## B. 完整状态多状态基准

### 协议说明
- 这类实验要求受试者同时具备多个状态，因此样本量更小。
- 它们适合做研究方向和多状态消融，但应与 `61` 人主基准分开汇报。

| 方法 | 脚本 / 设置 | 受试者数 | Accuracy | Balanced Accuracy | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| `TASK + EC + EO` + 客观 QC | `run_multistate_explicit_region_temporal_node_gnn_5x10.py --include-pairwise-contrasts --c-grid 0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0,5.0,10.0 --qc-max-abs-threshold 0.0018 --qc-kurtosis-threshold 550 --qc-bad-window-abs-threshold 0.0003 --qc-bad-window-min-ratio 0.16` | `38` | **`0.9750`** | **`0.9750`** | **`0.9800`** | **`0.9900`** |
| 完整状态子集上的 `TASK` only | `run_multistate_explicit_region_temporal_node_gnn_5x10.py --states TASK` | `53` | `0.9253` | `0.9233` | `0.9312` | `0.9767` |
| `TASK + EC` | `run_multistate_explicit_region_temporal_node_gnn_5x10.py --states TASK,EC` | `55` | `0.9233` | `0.9183` | `0.9222` | `0.9567` |
| `TASK + EO` | `run_multistate_explicit_region_temporal_node_gnn_5x10.py --states TASK,EO` | `58` | `0.9247` | `0.9183` | `0.9291` | `0.9581` |
| `TASK + EC + EO` | `run_multistate_explicit_region_temporal_node_gnn_5x10.py` | `53` | `0.9260` | `0.9233` | `0.9231` | `0.9667` |

## 主要结果文件
- 61 人主基准：
  - `outputs/metrics/runs/static_feature_weightedstar_5x10_thracc_widethr/summary_5x10.json`
  - `outputs/metrics/runs/explicit_region_temporal_summary_targeted_clean250_staticclean033_tempzero033_c025_5x10/summary_5x10.json`
- 多状态完整子集：
  - `outputs/metrics/runs/multistate_explicit_region_temporal_node_gnn_5x10_v2/summary_5x10.json`
  - `outputs/metrics/runs/multistate_qc_impulsive_explicit_region_temporal_node_gnn_5x10/summary_5x10.json`
  - `outputs/metrics/runs/multistate_taskonly_on_tristate53_5x10/summary_5x10.json`
  - `outputs/metrics/runs/multistate_task_ec_5x10/summary_5x10.json`
  - `outputs/metrics/runs/multistate_task_eo_5x10/summary_5x10.json`

## 实际建议
- 如果目标是论文主结果，使用带上述 targeted cleaning 参数的 `run_explicit_region_temporal_summary_node_gnn_5x10.py`。
- 如果目标是多状态纯 GNN 研究，使用 `run_multistate_explicit_region_temporal_node_gnn_5x10.py`。
- 如果目标是最强多状态结果，使用同一脚本并加上客观 QC 参数；这条线最终保留 `38/53` 人。
- 如果需要区分稳定模型、探索脚本和探路脚本，参见 `docs/benchmarks/model_catalog.zh-CN.md`。
