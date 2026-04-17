# 10-Fold Benchmark 汇总

## 数据与评估设置
- 数据：TASK 子集，共 61 名受试者（HC=28，MDD=33）。
- 指标：均为 10-fold `mean ± std`。
- 图模型：默认使用 signed edge 方式（保留连接正负信息）。

## 全量对比结果
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

## 简要结论（便于后续写论文）
1. 非 GNN 目前仍是 Accuracy 最优（LR/RF 均为 0.8690），其中 RF 的 AUC 最高（0.9222）。
2. 单 PCC 的图模型是当前 GNN 稳定基线；宽频 PLV 融合未带来稳定增益。
3. 分频段 PLV 中，`theta` 在分类指标上与单 PCC 基本持平，`alpha/beta` 更偏向提升 AUC。
4. 多频段动态权重融合显著提升了 AUC（0.8889），但 Acc/BalAcc/F1 有所下降，说明排序能力增强但分类边界仍需优化。
5. 动态阈值（按验证集 BalAcc 搜索）在当前小样本下不稳定，整体指标低于固定阈值 0.5。

## 结果文件对应
- Non-GNN: `outputs/metrics/nongnn_cv10_metrics.json`
- Signed PCC: `outputs/metrics/gnn_gcn_signedpcc_cv10_metrics_latest.json`
- Signed PCC+PLV broad: `outputs/metrics/gnn_gcn_signed_pccplv_cv10_metrics_latest.json`
- Signed PCC+PLV theta: `outputs/metrics/gnn_gcn_signed_pccplv_theta_cv10_metrics_latest.json`
- Signed PCC+PLV alpha: `outputs/metrics/gnn_gcn_signed_pccplv_alpha_cv10_metrics_latest.json`
- Signed PCC+PLV beta: `outputs/metrics/gnn_gcn_signed_pccplv_beta_cv10_metrics_latest.json`
- 多频段动态权重（固定阈值）: `outputs/metrics/gnn_multiband_fusion_cv10_metrics_latest.json`
- 多频段动态权重（动态阈值）: `outputs/metrics/gnn_multiband_fusion_cv10_metrics_valthr_latest.json`
