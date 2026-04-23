# Source Index

`src/` contains three layers of code:

1. Stable benchmark entry points
2. Exploratory benchmark scripts
3. Lower-level training and graph-building modules

If you only need the main models, start from the first section and ignore the rest.

## Stable benchmark entry points
- `build_graphs.py`
  - Builds the canonical `TASK` graph inputs used by the main full-dataset benchmarks
- `run_static_feature_node_gnn_5x10.py`
  - Stable static pure-GNN baseline on the full `61`-subject benchmark
- `run_explicit_region_temporal_summary_node_gnn_5x10.py`
  - Current primary full-dataset pure-GNN benchmark
- `run_multistate_explicit_region_temporal_node_gnn_5x10.py`
  - Multistate pure-GNN entry point for complete-state subsets such as `TASK+EC+EO`

## Comparator or historical reference scripts
- `train_baseline.py`
  - Early non-GNN baseline and split generation
- `run_static_topk_5x10.py`
  - Earlier static top-k benchmark route
- `run_static_gnn_graphfeature_residual_5x10.py`
  - Residual graph-feature comparator
- `run_static_gnn_graphvector_hybrid_5x10.py`
  - Static hybrid comparator
- `run_static_flatreadout_graphvector_hybrid_5x10.py`
  - Static flat-readout comparator
- `run_benchmarks.py`
  - Historical benchmark wrapper

## Exploratory pure-GNN scripts
- `run_region_temporal_feature_node_gnn_5x10.py`
- `run_explicit_region_temporal_dynamics_node_gnn_5x10.py`
- `run_explicit_region_temporal_dynamics_graphconv_5x10.py`
- `run_prior_adaptive_regiontemporal_gnn_5x10.py`
- `run_multistate_regiongraph_gnn_5x10.py`
- `run_temporal_variants_5x10.py`
- `run_spatiotemporal_pyc_5x10.py`
- `run_spatiotemporal_pyc_multiinit_5x10.py`
- `run_spatiotemporal_regiongat_multiinit_5x10.py`

These are worth keeping, but they are still research routes rather than the clean benchmark headline.

## Ensemble and selector scripts
- `run_pure_gnn_ensemble_cv10.py`
- `run_region_temporal_validation_selector_5x10.py`
- `run_flatreadout_multiinit_ensemble_cv10.py`
- `run_flatwin_multiinit_ensemble_cv10.py`
- `run_spatiotemporal_snapshot_ensemble_cv10.py`
- `run_spatiotemporal_multiinit_ensemble_cv10.py`

Treat these as utilities for search, probing, or ensembling. They are not the default single-model report line.

## Lower-level training modules
- `train_gnn.py`
- `train_gnn_cv10.py`
- `train_gnn_multiband_cv10.py`
- `train_gnn_dualgraph_cv10.py`
- `train_gnn_dualgraph_multiband_cv10.py`
- `train_gnn_dualgraph_multiband_flatreadout_cv10.py`
- `train_gnn_dualgraph_multiband_regionprior_flatreadout_cv10.py`
- `train_gnn_dualgraph_multiband_regiontemporal_cv10.py`
- `train_gnn_dualgraph_multiband_spatiotemporal_cv10.py`
- `train_gnn_dualgraph_multiband_spatiotemporal_flatwin_cv10.py`
- `train_gnn_dualgraph_multiband_spatiotemporal_hybridwin_cv10.py`
- `train_gnn_dualgraph_multiband_stagewise_cv10.py`
- `train_gnn_spatiotemporal_cv10.py`
- `train_nongnn_cv10.py`

These are implementation modules or older training entry points. Use them only if you are intentionally reproducing an older line or building a new benchmark script on top of them.
