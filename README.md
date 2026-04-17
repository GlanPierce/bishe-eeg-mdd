# EEG MDD Baseline (毕业设计阶段代码)

## 当前进度
- 已完成 Kaggle 数据集下载（本地）与 `TASK` 样本解析。
- 已完成受试者级 `train/test` 切分。
- 已完成初步基线模型训练（Logistic Regression / Random Forest）。
- 已输出初步结果报告与十天检查计划文档。

## 项目结构
- `src/train_baseline.py`：基线训练主脚本（特征提取 + 切分 + 训练 + 指标输出）
- `src/build_graphs.py`：PCC 脑网络构图脚本（输出图数据供 GNN 使用）
- `docs/initial_model_report.md`：初步模型结果报告
- `docs/10_day_check_plan.md`：十天代码检查执行路径
- `requirements.txt`：依赖清单
- `env_check.py`：环境健康检查
- `run_env_check.ps1`：一键环境检查入口

> 说明：原始数据和虚拟环境体积较大，默认不纳入版本库（见 `.gitignore`）。

## 运行方式
```powershell
.\.venv\Scripts\python.exe .\env_check.py
.\.venv\Scripts\python.exe .\src\train_baseline.py
.\.venv\Scripts\python.exe .\src\build_graphs.py
```

## 当前基线结果（测试集）
- Accuracy: `0.9231`
- Balanced Accuracy: `0.9167`
- F1: `0.9333`

详细指标见 `outputs/metrics/baseline_task_binary_metrics.json`（本地生成，不纳入 git）。

## 下一步
1. 引入连接矩阵特征（PCC/PLV）并构图。
2. 引入 GCN/GAT（对齐开题报告路线）。
3. 做时空融合（GCN + 时序模块）与可解释性分析。
