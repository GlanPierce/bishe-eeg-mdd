# Benchmark Summary

## Scope Note
This repository now contains two benchmark families and they should not be mixed:

1. Full-dataset `TASK` benchmark: `61` subjects, thesis-facing primary comparison line
2. Complete-state multistate benchmark: only subjects with complete `TASK / EC / EO` recordings

The multistate models score higher, but they run on a smaller subset and are therefore not directly comparable with the primary `61`-subject benchmark.

## A. Canonical Full-Dataset Benchmark

### Dataset and protocol
- Subjects: `61` (`HC=28`, `MDD=33`)
- Input scope: `TASK` only
- Split: subject-level stratified CV
- Stability report: `5 seeds x 10 folds`

### Recommended models
| Method | Script | Accuracy | Balanced Accuracy | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Static pure GNN (`FeatureNodeGNN`) | `src/run_static_feature_node_gnn_5x10.py` | `0.9010 +/- 0.0106` | `0.8950 +/- 0.0081` | `0.9066 +/- 0.0138` | `0.9456 +/- 0.0227` |
| Explicit region + temporal pure GNN (`ExplicitRegionTemporalWeightedStarGNN`) | `src/run_explicit_region_temporal_summary_node_gnn_5x10.py` | **`0.9143 +/- 0.0060`** | **`0.9100 +/- 0.0086`** | **`0.9173 +/- 0.0054`** | **`0.9708 +/- 0.0126`** |

### Historical references
| Method | Validation setup | Accuracy | Balanced Accuracy | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Static dual-graph multiband + top-k + signed split | `10-fold` | `0.8524 +/- 0.0497` | `0.8333 +/- 0.0645` | `0.8695 +/- 0.0533` | `0.8708 +/- 0.1491` |
| Spatiotemporal dual-graph (`16/8/12`) | `10-fold` | `0.8690 +/- 0.1243` | `0.8667 +/- 0.1247` | `0.8794 +/- 0.1226` | `0.8667 +/- 0.1296` |

## B. Complete-State Multistate Benchmark

### Protocol note
- These runs require complete state availability and therefore use smaller subject subsets.
- They are useful for research direction and multistate ablation, but should be reported separately from the canonical `61`-subject line.

| Method | Script / Setting | Subjects | Accuracy | Balanced Accuracy | F1 | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| `TASK` only on complete-state subset | `run_multistate_explicit_region_temporal_node_gnn_5x10.py --states TASK` | `53` | `0.9253` | `0.9233` | `0.9312` | `0.9767` |
| `TASK + EC` | `run_multistate_explicit_region_temporal_node_gnn_5x10.py --states TASK,EC` | `55` | `0.9233` | `0.9183` | `0.9222` | `0.9567` |
| `TASK + EO` | `run_multistate_explicit_region_temporal_node_gnn_5x10.py --states TASK,EO` | `58` | `0.9247` | `0.9183` | `0.9291` | `0.9581` |
| `TASK + EC + EO` | `run_multistate_explicit_region_temporal_node_gnn_5x10.py` | `53` | **`0.9260`** | **`0.9233`** | `0.9231` | `0.9667` |

## Main Result Files
- Canonical full-dataset:
  - `outputs/metrics/runs/static_feature_weightedstar_5x10_thracc_widethr/summary_5x10.json`
  - `outputs/metrics/runs/explicit_region_temporal_summary_node_gnn_5x10/summary_5x10.json`
- Multistate complete-subject:
  - `outputs/metrics/runs/multistate_explicit_region_temporal_node_gnn_5x10_v2/summary_5x10.json`
  - `outputs/metrics/runs/multistate_taskonly_on_tristate53_5x10/summary_5x10.json`
  - `outputs/metrics/runs/multistate_task_ec_5x10/summary_5x10.json`
  - `outputs/metrics/runs/multistate_task_eo_5x10/summary_5x10.json`

## Practical Recommendation
- If the target is the main thesis benchmark on the full dataset, use `run_explicit_region_temporal_summary_node_gnn_5x10.py`.
- If the target is multistate pure-GNN research on complete subjects, use `run_multistate_explicit_region_temporal_node_gnn_5x10.py`.
- Use `docs/benchmarks/model_catalog.md` to distinguish stable models from exploratory and probe-only scripts.
