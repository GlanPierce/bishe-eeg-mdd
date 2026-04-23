# Model Catalog

This file is the clean registry for model entry points in `src/`. It separates stable benchmark scripts from hybrid comparators, exploratory routes, and lower-level training modules.

## 1. Stable benchmark entry points
These are the scripts that currently define the repository's clean benchmark story.

| Tier | Family | Script | Scope | Best tracked result | Notes |
|---|---|---|---|---|---|
| Stable | Static pure GNN | `src/run_static_feature_node_gnn_5x10.py` | `61` subjects, `TASK` only | `0.9010` acc | Best clean static pure-GNN baseline on the full benchmark |
| Stable | Explicit region + temporal pure GNN | `src/run_explicit_region_temporal_summary_node_gnn_5x10.py` | `61` subjects, `TASK` only | `0.9143` acc | Current full-dataset primary pure-GNN result |
| Stable | Multistate explicit region-temporal pure GNN | `src/run_multistate_explicit_region_temporal_node_gnn_5x10.py` | complete-state subsets | `0.9260` acc on `TASK+EC+EO` | Clean multistate baseline on the complete-state subset |
| Stable | Multistate QC explicit region-temporal pure GNN | `src/run_multistate_explicit_region_temporal_node_gnn_5x10.py` with QC flags | QC-filtered complete-state subset | `0.9750` acc on `38` subjects | Current strongest multistate line; uses objective artifact filtering (`max_abs` + impulsive-window rule) |

## 2. Comparator and reference scripts
These scripts are useful for controlled comparisons or historical references, but they are not the primary pure-GNN story.

| Tier | Family | Script | Status | Notes |
|---|---|---|---|---|
| Comparator | Early non-GNN baseline | `src/train_baseline.py` | Historical | Baseline statistics and subject-level `TASK` split |
| Comparator | Static top-k GNN | `src/run_static_topk_5x10.py` | Historical | Earlier static top-k benchmark route |
| Comparator | Static graph-feature residual route | `src/run_static_gnn_graphfeature_residual_5x10.py` | Reference only | Residual graph-feature variant |
| Comparator | Hybrid static graph-vector route | `src/run_static_gnn_graphvector_hybrid_5x10.py` | Reference only | Includes non-pure-GNN readout behavior |
| Comparator | Hybrid flat-readout static route | `src/run_static_flatreadout_graphvector_hybrid_5x10.py` | Reference only | Useful for ablation, not main pure-GNN line |
| Comparator | Historical benchmark wrapper | `src/run_benchmarks.py` | Reference only | Aggregates older benchmark paths |

## 3. Exploratory pure-GNN routes
These are active research or probe scripts. Keep them, but do not present them as the main benchmark unless they are explicitly promoted.

| Tier | Family | Script | Current status | Notes |
|---|---|---|---|---|
| Exploratory | Region-temporal node GNN | `src/run_region_temporal_feature_node_gnn_5x10.py` | Below promoted line | Earlier explicit-region route |
| Exploratory | Dynamics weighted-star GNN | `src/run_explicit_region_temporal_dynamics_node_gnn_5x10.py` | Below promoted line | Adds dynamic summaries and validation search |
| Exploratory | Dynamics graph-conv GNN | `src/run_explicit_region_temporal_dynamics_graphconv_5x10.py` | Probe stage | True graph-conv version, currently weaker |
| Exploratory | Prior-adaptive region-temporal GNN | `src/run_prior_adaptive_regiontemporal_gnn_5x10.py` | Probe stage | Region-prior route |
| Exploratory | Multistate region-graph GNN | `src/run_multistate_regiongraph_gnn_5x10.py` | Probe stage | Small true-GNN region graph over states |
| Exploratory | Temporal variant pack | `src/run_temporal_variants_5x10.py` | Probe stage | Bundled temporal ablations |
| Exploratory | Spatiotemporal PYC route | `src/run_spatiotemporal_pyc_5x10.py` | Probe stage | Single-run spatiotemporal line |
| Exploratory | Spatiotemporal PYC multi-init | `src/run_spatiotemporal_pyc_multiinit_5x10.py` | Probe stage | Multi-init spatiotemporal variant |
| Exploratory | Spatiotemporal region-GAT | `src/run_spatiotemporal_regiongat_multiinit_5x10.py` | Probe stage | Region-aware attention route |

## 4. Ensemble or selection research scripts
These scripts are useful for searching or ensembling, but they should not be confused with a single-model benchmark.

| Tier | Family | Script | Notes |
|---|---|---|---|
| Research utility | Pure-GNN ensemble | `src/run_pure_gnn_ensemble_cv10.py` | Ensemble-style route, not the main single-model report line |
| Research utility | Validation selector | `src/run_region_temporal_validation_selector_5x10.py` | Fold-level selector, not a single standalone model |
| Research utility | Flatreadout multi-init ensemble | `src/run_flatreadout_multiinit_ensemble_cv10.py` | Search/ensemble support |
| Research utility | Flatwin multi-init ensemble | `src/run_flatwin_multiinit_ensemble_cv10.py` | Search/ensemble support |
| Research utility | Spatiotemporal snapshot ensemble | `src/run_spatiotemporal_snapshot_ensemble_cv10.py` | Search/ensemble support |
| Research utility | Spatiotemporal multi-init ensemble | `src/run_spatiotemporal_multiinit_ensemble_cv10.py` | Search/ensemble support |

## 5. Lower-level training modules
These are building blocks rather than the recommended top-level benchmark scripts.

| Tier | Family | Scripts |
|---|---|---|
| Core modules | Graph building and utilities | `src/build_graphs.py`, `src/feature_model_utils.py` |
| Legacy training entry points | Static / multiband / dual-graph / spatiotemporal trainers | `src/train_gnn*.py`, `src/train_nongnn_cv10.py` |

## 6. Reporting convention
- Use `run_explicit_region_temporal_summary_node_gnn_5x10.py` when the report must stay on the canonical full `61`-subject benchmark.
- Use `run_multistate_explicit_region_temporal_node_gnn_5x10.py` only when the report explicitly says it is a complete-state multistate subset experiment.
- When reporting the strongest multistate number, include the QC thresholds and the post-filter subject count, because the `0.9750` line is a quality-controlled subset rather than the raw `53`-subject complete-state pool.
- Do not present selector or ensemble scripts as if they were a single pure-GNN benchmark model.
