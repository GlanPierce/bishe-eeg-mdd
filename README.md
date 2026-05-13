# EEG-MDD Graduation Project

English | [简体中文](README.zh-CN.md)

EEG-based MDD recognition project built around graph construction, pure GNN experimentation, repeated `5 x 10` validation, and thesis-facing documentation.

## Start Here
- `docs/benchmarks/benchmark_summary.md`: compact benchmark view and recommended result files
- `docs/benchmarks/model_catalog.md`: clean model inventory with status and scope
- `src/README.md`: script-level entry-point guide
- `docs/README.md`: documentation index

## Results By Generalization Risk

### Full 61-subject primary benchmark
These are the thesis-facing full-dataset results. Keep the clean benchmark and the benchmark-tuned repair variant separate.

#### Clean benchmark results
These do not use dataset-tuned artifact-repair routing.

| Model | Script | Scope | 5x10 Accuracy | BalAcc | F1 | AUC |
|---|---|---|---:|---:|---:|---:|
| Static pure GNN (`FeatureNodeGNN`) | `src/run_static_feature_node_gnn_5x10.py` | `61` subjects, `TASK` only | `0.9010` | `0.8950` | `0.9066` | `0.9456` |
| Explicit region + temporal pure GNN (`ExplicitRegionTemporalWeightedStarGNN`) | `src/run_explicit_region_temporal_summary_node_gnn_5x10.py` | `61` subjects, `TASK` only | **`0.9143`** | **`0.9100`** | **`0.9173`** | **`0.9708`** |

#### Benchmark-tuned artifact-aware repair variant
This keeps all `61` subjects, but adds subject-level artifact-aware repair rules tuned on this dataset. Report it separately from the clean benchmark because it has higher benchmark-overfitting risk.

| Model | Script | Scope | 5x10 Accuracy | BalAcc | F1 | AUC |
|---|---|---|---:|---:|---:|---:|
| Explicit region + temporal pure GNN (`ExplicitRegionTemporalWeightedStarGNN`) + targeted repair | `src/run_explicit_region_temporal_summary_node_gnn_5x10.py` with targeted cleaning flags | `61` subjects, `TASK` only | **`0.9343`** | **`0.9283`** | **`0.9412`** | **`0.9581`** |

### Multistate research benchmark
This line is currently strongest numerically, but it uses only subjects with complete `TASK + EC + EO` states, so it is not directly comparable with the full `61`-subject benchmark above.

| Model | Script | Scope | 5x10 Accuracy | BalAcc | F1 | AUC |
|---|---|---|---:|---:|---:|---:|
| Multistate explicit region-temporal pure GNN | `src/run_multistate_explicit_region_temporal_node_gnn_5x10.py` | `53` complete-state subjects, `TASK+EC+EO` | `0.9260` | `0.9233` | `0.9231` | `0.9667` |
| Multistate QC explicit region-temporal pure GNN | `src/run_multistate_explicit_region_temporal_node_gnn_5x10.py` with QC flags | `38` QC-filtered complete-state subjects, `TASK+EC+EO` | **`0.9750`** | **`0.9750`** | **`0.9800`** | **`0.9900`** |

## Repository Layout
- `src/`: graph builders, benchmark entry points, and lower-level training modules
- `tools/`: environment checks and local helper scripts
- `notebooks/`: exploratory notebooks
- `docs/benchmarks/`: benchmark summaries, model catalog, and result-facing guidance
- `docs/notes/`: experiment notes and engineering records
- `docs/reports/`: proposal, interim, and report-oriented materials
- `docs/literature/`: paper search exports and literature summaries
- `docs/slides/`: slide drafts
- `outputs/metrics/runs/`: summary JSONs for benchmark and probe runs

## Environment Check
```powershell
.\tools\run_env_check.ps1
```

## Quick Reproduction

### 1) Build static task graphs
```powershell
# PCC
.\.venv\Scripts\python.exe .\src\build_graphs.py --edge-mode pcc --edge-selection per_node_topk --top-k-per-node 4 --signed-topk-split --out-dir data/processed/graphs_pcc_topk_task

# PLV theta/alpha/beta
.\.venv\Scripts\python.exe .\src\build_graphs.py --edge-mode plv --plv-band theta --edge-selection per_node_topk --top-k-per-node 4 --out-dir data/processed/graphs_plv_theta_topk_task
.\.venv\Scripts\python.exe .\src\build_graphs.py --edge-mode plv --plv-band alpha --edge-selection per_node_topk --top-k-per-node 4 --out-dir data/processed/graphs_plv_alpha_topk_task
.\.venv\Scripts\python.exe .\src\build_graphs.py --edge-mode plv --plv-band beta --edge-selection per_node_topk --top-k-per-node 4 --out-dir data/processed/graphs_plv_beta_topk_task
```

### 2) Reproduce the clean full-dataset primary pure GNN
```powershell
.\.venv\Scripts\python.exe .\src\run_explicit_region_temporal_summary_node_gnn_5x10.py
```

### 3) Reproduce the benchmark-tuned artifact-aware full-dataset variant
```powershell
.\.venv\Scripts\python.exe .\src\run_explicit_region_temporal_summary_node_gnn_5x10.py --c 0.25 --targeted-clean-bad-abs-threshold 0.00025 --targeted-clean-bad-window-ratio 0.16 --targeted-clean-summary-abs-threshold 0.00025 --targeted-static-clean-bad-window-ratio 0.33 --targeted-temporal-zero-bad-window-ratio 0.33 --out-dir outputs/metrics/runs/explicit_region_temporal_summary_targeted_clean250_staticclean033_tempzero033_c025_5x10 --experiment-name explicit_region_temporal_summary_targeted_clean250_staticclean033_tempzero033_c025_5x10
```

### 4) Reproduce the multistate research pure GNN
```powershell
.\.venv\Scripts\python.exe .\src\run_multistate_explicit_region_temporal_node_gnn_5x10.py
```

### 5) Reproduce the strongest QC-filtered multistate pure GNN
```powershell
.\.venv\Scripts\python.exe .\src\run_multistate_explicit_region_temporal_node_gnn_5x10.py --include-pairwise-contrasts --c-grid 0.01,0.02,0.05,0.1,0.2,0.5,1.0,2.0,5.0,10.0 --qc-max-abs-threshold 0.0018 --qc-kurtosis-threshold 550 --qc-bad-window-abs-threshold 0.0003 --qc-bad-window-min-ratio 0.16 --out-dir outputs/metrics/runs/multistate_qc_impulsive_explicit_region_temporal_node_gnn_5x10 --experiment-name multistate_qc_impulsive_explicit_region_temporal_node_gnn_5x10
```

## Key Result Files
- `outputs/metrics/runs/static_feature_weightedstar_5x10_thracc_widethr/summary_5x10.json`
- `outputs/metrics/runs/explicit_region_temporal_summary_node_gnn_5x10/summary_5x10.json`
- `outputs/metrics/runs/explicit_region_temporal_summary_targeted_clean250_staticclean033_tempzero033_c025_5x10/summary_5x10.json`
- `outputs/metrics/runs/multistate_explicit_region_temporal_node_gnn_5x10_v2/summary_5x10.json`
- `outputs/metrics/runs/multistate_qc_impulsive_explicit_region_temporal_node_gnn_5x10/summary_5x10.json`

## Notes
- The canonical full-dataset benchmark and the multistate complete-subject benchmark should be reported separately.
- For the canonical `61`-subject benchmark, the clean no-repair line is `0.9143`; this should be the default thesis benchmark number when the goal is lower benchmark-overfitting risk.
- The `0.9343` full-dataset result should be reported separately as a benchmark-tuned artifact-aware repair variant, not as the clean benchmark number.
- The QC-filtered multistate line should also be reported separately from the raw complete-state `53`-subject benchmark because it excludes objective artifact outliers before CV.
- Most raw data, processed graphs, and generated outputs are git-ignored.
- Local scratch files are grouped under `local/`.
