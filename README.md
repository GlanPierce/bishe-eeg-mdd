# EEG-MDD Graduation Project

## Highlights (2026-04)
- Best static model: `Dual-Graph Multiband (PCC + PLV theta/alpha/beta) + per_node_topk + signed_split`
- Best static 10-fold: Acc `0.8524 +/- 0.0497`, BalAcc `0.8333 +/- 0.0645`, F1 `0.8695 +/- 0.0533`, AUC `0.8708 +/- 0.1491`
- Best spatiotemporal 10-fold: `Static-best + BiGRU + window/step/max=16/8/12`
  - Acc `0.8690 +/- 0.1243`, BalAcc `0.8667 +/- 0.1247`, F1 `0.8794 +/- 0.1226`, AUC `0.8667 +/- 0.1296`
- New stable static 5x10: `Graph-vector LogisticRegression on PCC + PLV(theta/alpha/beta) top-k graphs`
  - Acc `0.9043 +/- 0.0063`, BalAcc `0.8983 +/- 0.0057`, F1 `0.9095 +/- 0.0094`, AUC `0.9594 +/- 0.0098`
- New stable spatiotemporal 5x10: `Static graph vector + temporal window summary (8/8/12) + LogisticRegression`
  - Acc `0.9143 +/- 0.0087`, BalAcc `0.9100 +/- 0.0082`, F1 `0.9173 +/- 0.0130`, AUC `0.9686 +/- 0.0088`

## Repository Layout
- `src/build_graphs.py`: graph construction (`pcc`, `plv`, `wpli`, `dwpli`, `pcc_plv`)
- `src/train_nongnn_cv10.py`: LR/RF 10-fold baseline
- `src/train_gnn_cv10.py`: single-graph signed GCN 10-fold
- `src/train_gnn_dualgraph_multiband_cv10.py`: dual-graph multiband static model
- `src/train_gnn_dualgraph_multiband_stagewise_cv10.py`: stage-wise ablation (`baseline -> reweight -> prior -> mask`)
- `src/train_gnn_dualgraph_multiband_spatiotemporal_cv10.py`: multiband spatiotemporal model
- `src/train_gnn_spatiotemporal_cv10.py`: PCC-only spatiotemporal baseline
- `src/run_benchmarks.py`: unified benchmark entry
- `src/run_static_topk_5x10.py`: repeated 5x10 static graph-vector benchmark
- `src/run_temporal_variants_5x10.py`: repeated 5x10 temporal-summary / hybrid benchmark
- `docs/benchmark_summary.md`: concise benchmark report

## Environment Check
```powershell
.\.venv\Scripts\python.exe .\env_check.py
```

## Quick Reproduction

### 1) Build static graphs (top-k)
```powershell
# PCC
.\.venv\Scripts\python.exe .\src\build_graphs.py --edge-mode pcc --edge-selection per_node_topk --top-k-per-node 4 --signed-topk-split --out-dir data/processed/graphs_pcc_topk_task

# PLV theta/alpha/beta
.\.venv\Scripts\python.exe .\src\build_graphs.py --edge-mode plv --plv-band theta --edge-selection per_node_topk --top-k-per-node 4 --out-dir data/processed/graphs_plv_theta_topk_task
.\.venv\Scripts\python.exe .\src\build_graphs.py --edge-mode plv --plv-band alpha --edge-selection per_node_topk --top-k-per-node 4 --out-dir data/processed/graphs_plv_alpha_topk_task
.\.venv\Scripts\python.exe .\src\build_graphs.py --edge-mode plv --plv-band beta  --edge-selection per_node_topk --top-k-per-node 4 --out-dir data/processed/graphs_plv_beta_topk_task
```

### 2) Train static best model
```powershell
.\.venv\Scripts\python.exe .\src\train_gnn_dualgraph_multiband_cv10.py --pcc-manifest data/processed/graphs_pcc_topk_task/manifest.csv --theta-manifest data/processed/graphs_plv_theta_topk_task/manifest.csv --alpha-manifest data/processed/graphs_plv_alpha_topk_task/manifest.csv --beta-manifest data/processed/graphs_plv_beta_topk_task/manifest.csv --out-path outputs/metrics/runs/dualgraph_multiband_topk_full_cv10/gnn_dualgraph_multiband_topk_cv10.json --experiment-name gnn_dualgraph_multiband_topk_cv10
```

### 3) Train current best spatiotemporal model
```powershell
.\.venv\Scripts\python.exe .\src\train_gnn_dualgraph_multiband_spatiotemporal_cv10.py --window-seconds 16 --step-seconds 8 --max-windows 12 --epochs 200 --batch-size 16 --lr 1e-3 --weight-decay 1e-4 --dropout 0.2 --seed 42 --out-path outputs/metrics/runs/spatiotemporal_cv10/gnn_dualgraph_multiband_spatiotemporal_cv10_e200_w16_s8_m12.json --experiment-name gnn_dualgraph_multiband_spatiotemporal_cv10_e200_w16_s8_m12
```

### 4) Reproduce the stable static 5x10 result
```powershell
.\.venv\Scripts\python.exe .\src\run_static_topk_5x10.py
```

### 5) Reproduce the stable spatiotemporal 5x10 result
```powershell
.\.venv\Scripts\python.exe .\src\run_temporal_variants_5x10.py --variant hybrid
```

## Notes
- Large datasets and generated outputs are git-ignored.
- For full metric details, see `docs/benchmark_summary.md`.
