# EEG-MDD Graduation Project

EEG-based MDD recognition project built around graph construction, pure GNN experimentation, repeated `5 x 10` validation, and thesis-facing documentation.

## Start Here
- `docs/benchmarks/benchmark_summary.md`: compact benchmark view and recommended result files
- `docs/benchmarks/model_catalog.md`: clean model inventory with status and scope
- `src/README.md`: script-level entry-point guide
- `docs/README.md`: documentation index

## Current Recommended Results

### Full 61-subject primary benchmark
These are the main thesis-facing results because they use the full subject set and the same `TASK` benchmark scope.

| Model | Script | Scope | 5x10 Accuracy | BalAcc | F1 | AUC |
|---|---|---|---:|---:|---:|---:|
| Static pure GNN (`FeatureNodeGNN`) | `src/run_static_feature_node_gnn_5x10.py` | `61` subjects, `TASK` only | `0.9010` | `0.8950` | `0.9066` | `0.9456` |
| Explicit region + temporal pure GNN (`ExplicitRegionTemporalWeightedStarGNN`) | `src/run_explicit_region_temporal_summary_node_gnn_5x10.py` | `61` subjects, `TASK` only | `0.9143` | `0.9100` | `0.9173` | `0.9708` |

### Multistate research benchmark
This line is currently strongest numerically, but it uses only subjects with complete `TASK + EC + EO` states, so it is not directly comparable with the full `61`-subject benchmark above.

| Model | Script | Scope | 5x10 Accuracy | BalAcc | F1 | AUC |
|---|---|---|---:|---:|---:|---:|
| Multistate explicit region-temporal pure GNN | `src/run_multistate_explicit_region_temporal_node_gnn_5x10.py` | `53` complete-state subjects, `TASK+EC+EO` | `0.9260` | `0.9233` | `0.9231` | `0.9667` |

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

### 2) Reproduce the full-dataset recommended pure GNN
```powershell
.\.venv\Scripts\python.exe .\src\run_explicit_region_temporal_summary_node_gnn_5x10.py
```

### 3) Reproduce the multistate research pure GNN
```powershell
.\.venv\Scripts\python.exe .\src\run_multistate_explicit_region_temporal_node_gnn_5x10.py
```

## Key Result Files
- `outputs/metrics/runs/static_feature_weightedstar_5x10_thracc_widethr/summary_5x10.json`
- `outputs/metrics/runs/explicit_region_temporal_summary_node_gnn_5x10/summary_5x10.json`
- `outputs/metrics/runs/multistate_explicit_region_temporal_node_gnn_5x10_v2/summary_5x10.json`

## Notes
- The canonical full-dataset benchmark and the multistate complete-subject benchmark should be reported separately.
- Most raw data, processed graphs, and generated outputs are git-ignored.
- Local scratch files are grouped under `local/`.
