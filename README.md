# EEG MDD Baseline (毕业设计阶段代码)

## 当前进度
- 已完成 Kaggle 数据集下载（本地）与 `TASK` 样本解析。
- 已完成受试者级 `train/test` 切分。
- 已完成初步基线模型训练（Logistic Regression / Random Forest）。
- 已输出初步结果报告与十天检查计划文档。

## 项目结构
- `src/train_baseline.py`：基线训练主脚本（特征提取 + 切分 + 训练 + 指标输出）
- `src/build_graphs.py`：PCC 脑网络构图脚本（输出图数据供 GNN 使用）
- `src/train_gnn.py`：单次划分 GNN 训练
- `src/train_gnn_cv10.py`：GNN 10-fold 评估
- `src/train_gnn_multiband_cv10.py`：多频段融合 GNN 10-fold（动态权重/动态阈值实验）
- `src/train_nongnn_cv10.py`：非 GNN（LR/RF）10-fold 评估
- `docs/initial_model_report.md`：初步模型结果报告
- `docs/10_day_check_plan.md`：十天代码检查执行路径
- `docs/graph_principle.md`：构图原理说明
- `docs/benchmark_summary.md`：完整 benchmark 对比汇总（含非GNN、单PCC、broad、分频段、动态权重、动态阈值）
- `docs/plv_band_experiment_notes.md`：问题发现与修复记录（论文可直接引用）
- `requirements.txt`：依赖清单
- `env_check.py`：环境健康检查
- `run_env_check.ps1`：一键环境检查入口

> 说明：原始数据和虚拟环境体积较大，默认不纳入版本库（见 `.gitignore`）。

## 运行方式
```powershell
.\.venv\Scripts\python.exe .\env_check.py
.\.venv\Scripts\python.exe .\src\train_baseline.py
.\.venv\Scripts\python.exe .\src\build_graphs.py
.\.venv\Scripts\python.exe .\src\train_gnn_cv10.py --manifest data/processed/graphs_pcc_task/manifest.csv
.\.venv\Scripts\python.exe .\src\train_gnn_multiband_cv10.py
.\.venv\Scripts\python.exe .\src\train_gnn_multiband_cv10.py --threshold-strategy val_balacc --out-path outputs/metrics/gnn_multiband_fusion_cv10_metrics_valthr_latest.json
.\.venv\Scripts\python.exe .\src\train_nongnn_cv10.py
```

`build_graphs.py` 现在支持两种选边策略：
- `global_quantile`（原策略）：全局分位数阈值
- `per_node_topk`（新策略）：每个节点 top-k 选边；对 signed 图可分开保留正/负边

示例（推荐用于 signed PCC 图）：
```powershell
.\.venv\Scripts\python.exe .\src\build_graphs.py `
  --edge-mode pcc `
  --edge-selection per_node_topk `
  --top-k-per-node 4 `
  --signed-topk-split `
  --out-dir data/processed/graphs_pcc_topk_task
```

## 统一模板入口（推荐）
```powershell
.\.venv\Scripts\python.exe .\src\run_benchmarks.py
```

- 统一入口会自动：
  - 固定随机种子（默认 `seed=42`，可通过参数修改）
  - 给每个实验生成唯一输出文件（避免覆盖）
  - 产出统一字段的标准化汇总（便于横向对比）
- 输出目录示例：
  - `outputs/metrics/runs/benchmark_YYYYMMDDTHHMMSSZ_seed42/benchmark_summary.json`
  - `outputs/metrics/runs/benchmark_YYYYMMDDTHHMMSSZ_seed42/benchmark_summary.md`

## 当前基线结果（测试集）
- Accuracy: `0.9231`
- Balanced Accuracy: `0.9167`
- F1: `0.9333`

详细指标见 `outputs/metrics/baseline_task_binary_metrics.json`（本地生成，不纳入 git）。

## 最新 10-Fold 对比（总览）
| 方法 | Accuracy | Balanced Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Non-GNN (LogisticRegression) | 0.8690 ± 0.0995 | 0.8583 ± 0.1057 | 0.8816 ± 0.0984 | 0.8500 ± 0.1262 |
| Non-GNN (RandomForest) | 0.8690 ± 0.0659 | 0.8500 ± 0.0816 | 0.8781 ± 0.0702 | 0.9222 ± 0.0911 |
| GNN (Signed PCC, 单PCC) | 0.8405 ± 0.1339 | 0.8292 ± 0.1425 | 0.8651 ± 0.1158 | 0.7972 ± 0.1757 |
| GNN (Signed PCC+PLV, broad) | 0.8238 ± 0.1437 | 0.8125 ± 0.1505 | 0.8544 ± 0.1209 | 0.7986 ± 0.1707 |
| GNN (Signed PCC+PLV, theta) | 0.8405 ± 0.1339 | 0.8292 ± 0.1425 | 0.8651 ± 0.1158 | 0.7986 ± 0.1707 |
| GNN (Signed PCC+PLV, alpha) | 0.8238 ± 0.1437 | 0.8125 ± 0.1505 | 0.8544 ± 0.1209 | 0.8194 ± 0.1788 |
| GNN (Signed PCC+PLV, beta) | 0.8238 ± 0.1437 | 0.8125 ± 0.1505 | 0.8487 ± 0.1220 | 0.8083 ± 0.1776 |
| GNN (多频段动态权重融合, 固定阈值0.5) | 0.8214 ± 0.1113 | 0.8125 ± 0.1137 | 0.8401 ± 0.1109 | 0.8889 ± 0.1511 |
| GNN (多频段动态权重融合 + 动态阈值) | 0.7738 ± 0.1414 | 0.7750 ± 0.1397 | 0.7781 ± 0.1293 | 0.8556 ± 0.1432 |

完整对比、问题解释与实验记录见：
- `docs/benchmark_summary.md`
- `docs/plv_band_experiment_notes.md`

## 下一步
1. 引入连接矩阵特征（PCC/PLV）并构图。
2. 引入 GCN/GAT（对齐开题报告路线）。
3. 做时空融合（GCN + 时序模块）与可解释性分析。

## Dual-Graph PCC+PLV（不压缩边权）
- 新增训练入口：`src/train_gnn_dualgraph_cv10.py`
- 思路：PCC 图和 PLV 图分别编码，模型内部学习融合权重，不再使用 `alpha*PCC + (1-alpha)*PLV` 的手工边权压缩。

示例流程：
```powershell
# 1) 构建 PCC 图
.\.venv\Scripts\python.exe .\src\build_graphs.py --edge-mode pcc --out-dir data/processed/graphs_pcc_task

# 2) 构建 PLV 图（broad）
.\.venv\Scripts\python.exe .\src\build_graphs.py --edge-mode plv --plv-band broad --out-dir data/processed/graphs_plv_broad_task

# 3) 训练双路融合模型（10-fold）
.\.venv\Scripts\python.exe .\src\train_gnn_dualgraph_cv10.py `
  --pcc-manifest data/processed/graphs_pcc_task/manifest.csv `
  --plv-manifest data/processed/graphs_plv_broad_task/manifest.csv
```

统一 benchmark 入口也已支持双路模型（可用 `--skip-dualgraph` 跳过）。
