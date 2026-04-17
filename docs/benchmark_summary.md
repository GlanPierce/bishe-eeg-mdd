# 10-Fold Benchmark 汇总

## 数据与评估设置
- 数据：TASK 子集，共 61 名受试者（HC=28，MDD=33）。
- 指标：10-fold `mean +/- std`。
- 图模型：默认保留 signed 边信息（不丢失 PCC 的正负相关）。
- 本次新增：`Dual-Graph PCC+PLV`（两路图分别编码，模型内学习融合权重，不做手工边权压缩）；并扩展了 `PCC + PLV(theta/alpha/beta)` 多频段动态融合版本。

## 全量对比结果
| 方法 | Accuracy | Balanced Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Non-GNN (LogisticRegression) | 0.8690 +/- 0.0995 | 0.8583 +/- 0.1057 | 0.8816 +/- 0.0984 | 0.8500 +/- 0.1262 |
| Non-GNN (RandomForest) | 0.8690 +/- 0.0659 | 0.8500 +/- 0.0816 | 0.8781 +/- 0.0702 | 0.9222 +/- 0.0911 |
| GNN (Signed PCC, 单PCC) | 0.8405 +/- 0.1339 | 0.8292 +/- 0.1425 | 0.8651 +/- 0.1158 | 0.7972 +/- 0.1757 |
| GNN (Signed PCC+PLV, broad, 单图融合) | 0.8238 +/- 0.1437 | 0.8125 +/- 0.1505 | 0.8544 +/- 0.1209 | 0.7986 +/- 0.1707 |
| GNN (Dual-Graph PCC+PLV, broad, 模型融合) | 0.8167 +/- 0.1167 | 0.8000 +/- 0.1190 | 0.8183 +/- 0.1509 | 0.8111 +/- 0.1975 |
| GNN (Dual-Graph Multiband: PCC + PLV(theta/alpha/beta), 动态融合) | 0.8357 +/- 0.1057 | 0.8250 +/- 0.1083 | 0.8373 +/- 0.1424 | 0.8556 +/- 0.1432 |
| GNN (Signed PCC+PLV, theta) | 0.8405 +/- 0.1339 | 0.8292 +/- 0.1425 | 0.8651 +/- 0.1158 | 0.7986 +/- 0.1707 |
| GNN (Signed PCC+PLV, alpha) | 0.8238 +/- 0.1437 | 0.8125 +/- 0.1505 | 0.8544 +/- 0.1209 | 0.8194 +/- 0.1788 |
| GNN (Signed PCC+PLV, beta) | 0.8238 +/- 0.1437 | 0.8125 +/- 0.1505 | 0.8487 +/- 0.1220 | 0.8083 +/- 0.1776 |
| GNN (多频段动态权重融合 + 固定阈值0.5) | 0.8214 +/- 0.1113 | 0.8125 +/- 0.1137 | 0.8401 +/- 0.1109 | 0.8889 +/- 0.1511 |
| GNN (多频段动态权重融合 + 动态阈值) | 0.7738 +/- 0.1414 | 0.7750 +/- 0.1397 | 0.7781 +/- 0.1293 | 0.8556 +/- 0.1432 |

## 本次新增模型（Dual-Graph / Multiband）的关键观察
1. 相比“Dual-Graph broad”，多频段动态融合在四项指标均提升：  
   - Accuracy: `0.8167 -> 0.8357`（+0.0190）  
   - Balanced Accuracy: `0.8000 -> 0.8250`（+0.0250）  
   - F1: `0.8183 -> 0.8373`（+0.0190）  
   - ROC-AUC: `0.8111 -> 0.8556`（+0.0444）
2. 这说明“把 PLV 分成 theta/alpha/beta 并让模型动态选权重”是有效的，能减少宽频 PLV 把有用频段信息平均掉的问题。
3. 多频段动态融合的平均权重：`PCC=0.3103, theta=0.2438, alpha=0.2274, beta=0.2185`，说明模型不是单押某一频段，而是在样本层面做自适应组合。
4. 与“单 PCC GNN”相比，多频段版本在 AUC 上明显更高（`0.8556` vs `0.7972`），但 Acc/BalAcc/F1 仍略低于单 PCC，提示下一步应优化决策阈值/校准以把排序优势转成分类优势。

## 结果文件对应
- Non-GNN: `outputs/metrics/nongnn_cv10_metrics.json`
- Signed PCC: `outputs/metrics/gnn_gcn_signedpcc_cv10_metrics_latest.json`
- Signed PCC+PLV broad: `outputs/metrics/gnn_gcn_signed_pccplv_cv10_metrics_latest.json`
- Signed PCC+PLV theta: `outputs/metrics/gnn_gcn_signed_pccplv_theta_cv10_metrics_latest.json`
- Signed PCC+PLV alpha: `outputs/metrics/gnn_gcn_signed_pccplv_alpha_cv10_metrics_latest.json`
- Signed PCC+PLV beta: `outputs/metrics/gnn_gcn_signed_pccplv_beta_cv10_metrics_latest.json`
- 多频段动态权重（固定阈值）: `outputs/metrics/gnn_multiband_fusion_cv10_metrics_latest.json`
- 多频段动态权重（动态阈值）: `outputs/metrics/gnn_multiband_fusion_cv10_metrics_valthr_latest.json`
- Dual-Graph PCC+PLV broad（本次新增）: `outputs/metrics/runs/dualgraph_full_cv10/gnn_dualgraph_pcc_plv_broad.json`
- Dual-Graph Multiband PCC+PLV(theta/alpha/beta)（本次新增）: `outputs/metrics/runs/dualgraph_multiband_full_cv10/gnn_dualgraph_multiband_pcc_plv_tab_cv10.json`

## 运行记录
- 统一入口运行ID：`dualgraph_full_cv10`
- 汇总输出：
  - `outputs/metrics/runs/dualgraph_full_cv10/benchmark_summary.json`
  - `outputs/metrics/runs/dualgraph_full_cv10/benchmark_summary.md`
- 多频段动态融合运行：
  - `outputs/metrics/runs/dualgraph_multiband_full_cv10/gnn_dualgraph_multiband_pcc_plv_tab_cv10.json`
