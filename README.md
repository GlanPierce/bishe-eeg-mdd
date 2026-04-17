# EEG-MDD 毕设实验仓库

## 当前最佳结果（2026-04）
- 模型：Dual-Graph Multiband（PCC + PLV theta/alpha/beta）
- 构图：`per_node_topk` + `signed_topk_split`
- 10-fold：Acc `0.8524 +/- 0.0497`，BalAcc `0.8333 +/- 0.0645`，F1 `0.8695 +/- 0.0533`，AUC `0.8708 +/- 0.1491`
- 结果文件：`outputs/metrics/runs/dualgraph_multiband_topk_full_cv10/gnn_dualgraph_multiband_topk_cv10.json`

## 项目结构
- `src/build_graphs.py`：EEG 构图（支持 `global_quantile` 与 `per_node_topk`）
- `src/train_nongnn_cv10.py`：非 GNN（LR/RF）10-fold
- `src/train_gnn_cv10.py`：单图 GNN 10-fold
- `src/train_gnn_dualgraph_cv10.py`：双路图（PCC + PLV broad）10-fold
- `src/train_gnn_dualgraph_multiband_cv10.py`：多频段双路（PCC + theta/alpha/beta）10-fold
- `src/train_gnn_multiband_cv10.py`：历史多频段融合基线
- `src/run_benchmarks.py`：统一 benchmark 入口
- `docs/benchmark_summary.md`：完整对比与结论

## 环境检查
```powershell
.\.venv\Scripts\python.exe .\env_check.py
```

## 快速复现（当前最佳配置）
```powershell
# 1) PCC top-k 图
.\.venv\Scripts\python.exe .\src\build_graphs.py --edge-mode pcc --edge-selection per_node_topk --top-k-per-node 4 --signed-topk-split --out-dir data/processed/graphs_pcc_topk_task

# 2) PLV top-k 图（theta / alpha / beta）
.\.venv\Scripts\python.exe .\src\build_graphs.py --edge-mode plv --plv-band theta --edge-selection per_node_topk --top-k-per-node 4 --out-dir data/processed/graphs_plv_theta_topk_task
.\.venv\Scripts\python.exe .\src\build_graphs.py --edge-mode plv --plv-band alpha --edge-selection per_node_topk --top-k-per-node 4 --out-dir data/processed/graphs_plv_alpha_topk_task
.\.venv\Scripts\python.exe .\src\build_graphs.py --edge-mode plv --plv-band beta  --edge-selection per_node_topk --top-k-per-node 4 --out-dir data/processed/graphs_plv_beta_topk_task

# 3) 训练多频段动态融合
.\.venv\Scripts\python.exe .\src\train_gnn_dualgraph_multiband_cv10.py --pcc-manifest data/processed/graphs_pcc_topk_task/manifest.csv --theta-manifest data/processed/graphs_plv_theta_topk_task/manifest.csv --alpha-manifest data/processed/graphs_plv_alpha_topk_task/manifest.csv --beta-manifest data/processed/graphs_plv_beta_topk_task/manifest.csv --out-path outputs/metrics/runs/dualgraph_multiband_topk_full_cv10/gnn_dualgraph_multiband_topk_cv10.json --experiment-name gnn_dualgraph_multiband_topk_cv10
```

## 统一入口（批量实验）
```powershell
.\.venv\Scripts\python.exe .\src\run_benchmarks.py
```

输出目录示例：
- `outputs/metrics/runs/<run_id>/benchmark_summary.json`
- `outputs/metrics/runs/<run_id>/benchmark_summary.md`

## 说明
- 数据和大体积产物默认不进 git（见 `.gitignore`）。
- 如需论文表格/结论，优先参考 `docs/benchmark_summary.md`。
