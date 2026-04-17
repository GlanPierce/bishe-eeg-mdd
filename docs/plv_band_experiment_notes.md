# Signed PCC + PLV 分频带实验记录（可用于论文）

## 1. 版本目标
- 在 `signed PCC` 路线上继续引入 `PLV` 信息，验证是否能提升 MDD/HC 二分类性能。
- 避免连接信息被错误“正值化”，并测试分频带 PLV（`theta/alpha/beta`）相对宽频 PLV 的效果。

## 2. 发现的问题与修复

### 问题 A：`PCC+PLV` 曾把边权变成全正，丢失负相关信息
- 现象：`pcc_plv` 早期融合写法是 `fused = alpha*|PCC| + (1-alpha)*PLV`，边权只保留非负值。
- 风险：`PCC` 的负相关连接被抹平，和 `signed PCC` 思路不一致。

修复：
- 将 PLV 从 `[0,1]` 映射到 `[-1,1]`：
  - `plv_signed = 2*PLV - 1`
- 改为有符号融合：
  - `fused_signed = alpha*PCC + (1-alpha)*plv_signed`
- 选边评分使用 `fused_signed^2`，但最终边权保留 `fused_signed` 正负号。

代码位置：
- `src/build_graphs.py` 中 `build_edges(...)` 的 `pcc_plv` 分支。

### 问题 B：PLV 宽频计算可能掩盖频段差异
- 现象：直接在宽频（0.5-45 Hz）上做 Hilbert 相位同步，`PLV` 对判别的增益不稳定。
- 风险：不同频段的相位同步差异被平均，难以体现抑郁相关频段特征。

修复：
- 新增参数 `--plv-band`，支持：
  - `broad`（宽频，默认）
  - `theta`（4-8 Hz）
  - `alpha`（8-13 Hz）
  - `beta`（13-30 Hz）
- 在计算 PLV 前先按指定频段带通，再做 Hilbert 相位同步。

代码位置：
- `src/build_graphs.py`：`PLV_BANDS`、`_plv_input_data(...)`、`parse_args(...)`。

### 问题 C：10-fold 训练结果文件会被覆盖
- 现象：`train_gnn_cv10.py` 默认写到同一路径 `outputs/metrics/gnn_gcn_task_cv10_metrics.json`。
- 风险：多组实验连续运行时，后一次会覆盖前一次，影响复现实验对比。

解决方式（本轮实验流程层面）：
- 每次运行后立即复制为唯一文件名保存（例如 `..._theta_...json`）。
- 后续可考虑在训练脚本中增加 `--out-path` 参数，从根源避免覆盖。

## 3. 实验设置
- 数据：TASK 子集，61 例（HC=28，MDD=33）。
- 评估：10-fold CV，指标为 mean ± std。
- 模型：当前 `GCNClassifier`（signed edge 版本）。
- 共同参数：epochs=200，batch=16，lr=1e-3，seed=42。

## 4. 最新结果（本轮重跑）

| 模型 | Accuracy | Balanced Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| GNN (Signed PCC) | 0.8405 ± 0.1339 | 0.8292 ± 0.1425 | 0.8651 ± 0.1158 | 0.7972 ± 0.1757 |
| GNN (Signed PCC+PLV, broad) | 0.8238 ± 0.1437 | 0.8125 ± 0.1505 | 0.8544 ± 0.1209 | 0.7986 ± 0.1707 |
| GNN (Signed PCC+PLV, theta) | 0.8405 ± 0.1339 | 0.8292 ± 0.1425 | 0.8651 ± 0.1158 | 0.7986 ± 0.1707 |
| GNN (Signed PCC+PLV, alpha) | 0.8238 ± 0.1437 | 0.8125 ± 0.1505 | 0.8544 ± 0.1209 | 0.8194 ± 0.1788 |
| GNN (Signed PCC+PLV, beta) | 0.8238 ± 0.1437 | 0.8125 ± 0.1505 | 0.8487 ± 0.1220 | 0.8083 ± 0.1776 |

结果文件：
- `outputs/metrics/gnn_gcn_signedpcc_cv10_metrics_latest.json`
- `outputs/metrics/gnn_gcn_signed_pccplv_cv10_metrics_latest.json`
- `outputs/metrics/gnn_gcn_signed_pccplv_theta_cv10_metrics_latest.json`
- `outputs/metrics/gnn_gcn_signed_pccplv_alpha_cv10_metrics_latest.json`
- `outputs/metrics/gnn_gcn_signed_pccplv_beta_cv10_metrics_latest.json`

## 5. 可写入论文的结论要点
- 本文首先修复了 `PCC+PLV` 融合中“边权全正化”的实现问题，使融合过程保留连接符号信息。
- 在此基础上进一步引入分频带 PLV，发现不同频段对指标影响存在差异：
  - `theta` 频段在 Accuracy/BalAcc/F1 上与 Signed PCC 基本持平，且 AUC 略有提升。
  - `alpha`、`beta` 在 AUC 上有潜在收益，但分类阈值相关指标（Accuracy/BalAcc/F1）下降。
- 说明 PLV 的贡献具有频段依赖性，且“排序能力提升（AUC）”与“固定阈值分类提升（Accuracy）”不一定同步。

## 6. 后续建议（论文下一步实验）
- 引入验证集阈值选择（按 fold 自适应阈值）后再比较各频段。
- 对 `fusion_alpha` 做网格搜索（如 0.2/0.5/0.8）评估 PLV 融合强度。
- 增加重复 CV（如 5×10-fold）并报告置信区间，提高统计稳健性。
