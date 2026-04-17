# 毕设初步模型结果报告

- 报告日期：2026-04-17
- 项目：面向多脑区划分的抑郁症估计系统（初步实验）
- 数据来源：Kaggle `alphajr7/mdd-patients-eeg-dataset`

## 1. 实验目标
本次实验目标是构建一个可复现的二分类基线流程，验证数据读取、训练/测试拆分、特征提取与模型评估链路可以稳定跑通。

## 2. 数据与划分
- 使用样本：仅使用 `TASK` 条件 EDF 文件
- 标签定义：`H=0`，`MDD=1`
- 拆分方式：受试者级分层拆分（避免同一受试者泄漏）

数据统计：
- 总样本：61
- 训练集：48（MDD=26，H=22）
- 测试集：13（MDD=7，H=6）

## 3. 特征与模型
- 预处理：EEG 0.5-45Hz 滤波
- 特征：delta/theta/alpha/beta/gamma 相对能量的均值与标准差
- 模型：
  - Logistic Regression（含标准化）
  - Random Forest

## 4. 初步结果
### 4.1 Logistic Regression
- Accuracy：0.9231
- Balanced Accuracy：0.9167
- F1：0.9333
- ROC-AUC：0.8810
- Confusion Matrix：`[[5,1],[0,7]]`

### 4.2 Random Forest
- Accuracy：0.9231
- Balanced Accuracy：0.9167
- F1：0.9333
- ROC-AUC：0.9286
- Confusion Matrix：`[[5,1],[0,7]]`

## 5. 初步结论
1. 两个模型在当前测试集上都达到约 0.9231 的准确率，说明基线流程可用。
2. 当前主要错误类型是将少量 `H` 预测为 `MDD`。
3. 由于测试集仅 13 个样本，此结果用于“流程可行性验证”，不能直接代表最终泛化能力。

## 6. 局限与风险
1. 当前仅使用 `TASK` 条件，尚未纳入 `EC/EO` 做对比。
2. 当前是统计特征基线，尚未进入图结构特征与 GNN 主路线。
3. 还需交叉验证和多次重复实验，降低单次随机拆分波动。

## 7. 下一步计划
1. 加入连接矩阵特征（PCC/PLV）。
2. 训练 GCN/GAT，与本报告基线做同口径比较。
3. 输出展示图：混淆矩阵、ROC、关键特征贡献。

## 8. 复现实验命令
```powershell
.\.venv\Scripts\python.exe .\src\train_baseline.py
```

---

附：
- 指标文件：`outputs/metrics/baseline_task_binary_metrics.json`
- 划分文件：`data/splits/task_split_subject_level.csv`
- 特征文件：`data/splits/task_feature_table.csv`
