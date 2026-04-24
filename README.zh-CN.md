# EEG-MDD 毕业设计项目

[English](README.md) | 简体中文

这是一个面向 EEG 抑郁症识别的毕业设计仓库，核心围绕图构建、纯 GNN 建模、重复 `5 x 10` 验证，以及论文写作所需的结果整理。

## 从这里开始
- `docs/benchmarks/benchmark_summary.zh-CN.md`：精简基准摘要和推荐结果文件
- `docs/benchmarks/model_catalog.zh-CN.md`：模型清单与状态分层
- `src/README.zh-CN.md`：脚本入口索引
- `docs/README.zh-CN.md`：文档总索引

## 当前推荐结果

### 61 人主基准
这部分是论文主结果，因为它使用了完整受试者集合，并保持 `TASK` 单状态的统一口径。

| 模型 | 脚本 | 范围 | 5x10 准确率 | BalAcc | F1 | AUC |
|---|---|---|---:|---:|---:|---:|
| 静态纯 GNN (`FeatureNodeGNN`) | `src/run_static_feature_node_gnn_5x10.py` | `61` 人，`TASK` | `0.9010` | `0.8950` | `0.9066` | `0.9456` |
| 显式脑区 + 时间摘要纯 GNN (`ExplicitRegionTemporalWeightedStarGNN`) | `src/run_explicit_region_temporal_summary_node_gnn_5x10.py`，配合 targeted cleaning 参数 | `61` 人，`TASK` | **`0.9343`** | **`0.9283`** | **`0.9412`** | **`0.9581`** |

### 多状态研究基准
这条线数值更高，但只使用同时具备 `TASK + EC + EO` 的完整状态子集，不能直接和上面的 `61` 人主基准混报。

| 模型 | 脚本 | 范围 | 5x10 准确率 | BalAcc | F1 | AUC |
|---|---|---|---:|---:|---:|---:|
| 多状态显式脑区时空纯 GNN | `src/run_multistate_explicit_region_temporal_node_gnn_5x10.py` | `53` 人完整状态子集，`TASK+EC+EO` | `0.9260` | `0.9233` | `0.9231` | `0.9667` |
| 多状态 QC 显式脑区时空纯 GNN | `src/run_multistate_explicit_region_temporal_node_gnn_5x10.py` 加 QC 参数 | `38` 人 QC 过滤完整状态子集，`TASK+EC+EO` | **`0.9750`** | **`0.9750`** | **`0.9800`** | **`0.9900`** |

## 仓库结构
- `src/`：图构建、基准入口脚本和底层训练模块
- `tools/`：环境检查和本地辅助脚本
- `notebooks/`：探索性 notebook
- `docs/benchmarks/`：基准摘要、模型目录和结果说明
- `docs/notes/`：实验笔记和工程记录
- `docs/reports/`：开题、中期和报告材料
- `docs/literature/`：论文检索导出和文献摘要
- `docs/slides/`：答辩或汇报幻灯片
- `outputs/metrics/runs/`：基准与探测实验的 summary JSON

## 环境检查
```powershell
.\tools\run_env_check.ps1
```

## 快速复现

### 1) 构建 TASK 图数据
```powershell
# PCC
.\.venv\Scripts\python.exe .\src\build_graphs.py --edge-mode pcc --edge-selection per_node_topk --top-k-per-node 4 --signed-topk-split --out-dir data/processed/graphs_pcc_topk_task

# PLV theta/alpha/beta
.\.venv\Scripts\python.exe .\src\build_graphs.py --edge-mode plv --plv-band theta --edge-selection per_node_topk --top-k-per-node 4 --out-dir data/processed/graphs_plv_theta_topk_task
.\.venv\Scripts\python.exe .\src\build_graphs.py --edge-mode plv --plv-band alpha --edge-selection per_node_topk --top-k-per-node 4 --out-dir data/processed/graphs_plv_alpha_topk_task
.\.venv\Scripts\python.exe .\src\build_graphs.py --edge-mode plv --plv-band beta --edge-selection per_node_topk --top-k-per-node 4 --out-dir data/processed/graphs_plv_beta_topk_task
```

### 2) 复现 61 人主基准推荐纯 GNN
```powershell
.\.venv\Scripts\python.exe .\src\run_explicit_region_temporal_summary_node_gnn_5x10.py --c 0.25 --targeted-clean-bad-abs-threshold 0.00025 --targeted-clean-bad-window-ratio 0.16 --targeted-clean-summary-abs-threshold 0.00025 --targeted-static-clean-bad-window-ratio 0.33 --targeted-temporal-zero-bad-window-ratio 0.33 --out-dir outputs/metrics/runs/explicit_region_temporal_summary_targeted_clean250_staticclean033_tempzero033_c025_5x10 --experiment-name explicit_region_temporal_summary_targeted_clean250_staticclean033_tempzero033_c025_5x10
```

### 3) 复现多状态研究纯 GNN
```powershell
.\.venv\Scripts\python.exe .\src\run_multistate_explicit_region_temporal_node_gnn_5x10.py
```

### 4) 复现最强 QC 多状态纯 GNN
```powershell
.\.venv\Scripts\python.exe .\src\run_multistate_explicit_region_temporal_node_gnn_5x10.py --include-pairwise-contrasts --c-grid 0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0,5.0,10.0 --qc-max-abs-threshold 0.0018 --qc-kurtosis-threshold 550 --qc-bad-window-abs-threshold 0.0003 --qc-bad-window-min-ratio 0.16 --out-dir outputs/metrics/runs/multistate_qc_impulsive_explicit_region_temporal_node_gnn_5x10 --experiment-name multistate_qc_impulsive_explicit_region_temporal_node_gnn_5x10
```

## 关键结果文件
- `outputs/metrics/runs/static_feature_weightedstar_5x10_thracc_widethr/summary_5x10.json`
- `outputs/metrics/runs/explicit_region_temporal_summary_targeted_clean250_staticclean033_tempzero033_c025_5x10/summary_5x10.json`
- `outputs/metrics/runs/multistate_explicit_region_temporal_node_gnn_5x10_v2/summary_5x10.json`
- `outputs/metrics/runs/multistate_qc_impulsive_explicit_region_temporal_node_gnn_5x10/summary_5x10.json`

## 说明
- `61` 人主基准和多状态完整子集基准必须分开汇报。
- 当前最强 `61` 人结果没有删人，而是做了受试者级伪迹修复：对噪声窗做 temporal clean summary，再对最差的 `6` 个受试者做 static-clean 替换和 temporal 分支置零。
- 多状态 QC 结果也要和原始 `53` 人完整状态结果分开汇报，因为它在交叉验证前先剔除了客观伪迹离群样本。
- 大多数原始数据、处理后图和生成结果都在 `.gitignore` 中。
- 本地草稿和临时文件统一放在 `local/`。
