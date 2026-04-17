# 10-Fold 基准对比

## 数据说明
- 共 61 名受试者（TASK 子集，HC=28，MDD=33）。
- 所有指标为 10-fold 的 mean ± std。
- GNN 使用有符号边建模（区分正相关边与负相关边）。

## 结果
| 模型 | Accuracy | Balanced Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| GNN (Signed PCC) | 0.8405 ± 0.1339 | 0.8292 ± 0.1425 | 0.8651 ± 0.1158 | 0.7972 ± 0.1757 |
| GNN (Signed PCC+PLV) | 0.8405 ± 0.1339 | 0.8292 ± 0.1425 | 0.8651 ± 0.1158 | 0.7972 ± 0.1757 |
| Non-GNN (LogisticRegression) | 0.8690 ± 0.0995 | 0.8583 ± 0.1057 | 0.8816 ± 0.0984 | 0.8500 ± 0.1262 |
| Non-GNN (RandomForest) | 0.8690 ± 0.0659 | 0.8500 ± 0.0816 | 0.8781 ± 0.0702 | 0.9222 ± 0.0911 |

## 初步结论
1. `GNN(Signed PCC)` 与 `GNN(Signed PCC+PLV)` 当前结果一致，说明在现有构图与训练设置下，PLV 分支尚未带来可观增益。
2. 当前最优为 `Non-GNN(RandomForest)`，在 Accuracy 与 ROC-AUC 上表现最好。
3. GNN 仍有优化空间，后续可从图构建策略、融合方式与超参数搜索继续推进。

## 结果文件
- GNN Signed PCC: `outputs/metrics/gnn_gcn_signedpcc_cv10_metrics.json`
- GNN Signed PCC+PLV: `outputs/metrics/gnn_gcn_signed_pccplv_cv10_metrics.json`
- Non-GNN CV10: `outputs/metrics/nongnn_cv10_metrics.json`
