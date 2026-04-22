# Benchmark Summary (10-Fold, TASK subset)

## Dataset and protocol
- Subjects: 61 (`HC=28`, `MDD=33`)
- Split: subject-level stratified 10-fold CV
- Metrics: mean +/- std across folds

## Recommended models
- Stable repeated-CV static recommendation: `Graph-vector LogisticRegression on PCC + PLV(theta/alpha/beta) top-k graphs`
  - 5x10 Acc `0.9043 +/- 0.0063`
  - 5x10 BalAcc `0.8983 +/- 0.0057`
  - 5x10 F1 `0.9095 +/- 0.0094`
  - 5x10 AUC `0.9594 +/- 0.0098`
- Stable repeated-CV spatiotemporal recommendation: `Static graph vector + temporal window summary (16/8/12) + LogisticRegression`
  - 5x10 Acc `0.8943 +/- 0.0081`
  - 5x10 BalAcc `0.8883 +/- 0.0061`
  - 5x10 F1 `0.8957 +/- 0.0123`
  - 5x10 AUC `0.9553 +/- 0.0164`
- Static recommendation: `Dual-Graph Multiband + per-node top-k + signed split`
  - Acc `0.8524 +/- 0.0497`
  - BalAcc `0.8333 +/- 0.0645`
  - F1 `0.8695 +/- 0.0533`
  - AUC `0.8708 +/- 0.1491`
- Current best accuracy: `Static-best + BiGRU + window/step/max=16/8/12`
  - Acc `0.8690 +/- 0.1243`
  - BalAcc `0.8667 +/- 0.1247`
  - F1 `0.8794 +/- 0.1226`
  - AUC `0.8667 +/- 0.1296`

## Core comparison table
| Method | Accuracy | Balanced Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Non-GNN (LogisticRegression) | 0.8690 +/- 0.0995 | 0.8583 +/- 0.1057 | 0.8816 +/- 0.0984 | 0.8500 +/- 0.1262 |
| Non-GNN (RandomForest) | 0.8690 +/- 0.0659 | 0.8500 +/- 0.0816 | 0.8781 +/- 0.0702 | 0.9222 +/- 0.0911 |
| Static best: Dual-Graph Multiband + top-k + signed split | 0.8524 +/- 0.0497 | 0.8333 +/- 0.0645 | 0.8695 +/- 0.0533 | 0.8708 +/- 0.1491 |
| Spatiotemporal (matched baseline, 8/4/24) | 0.8357 +/- 0.1293 | 0.8417 +/- 0.1205 | 0.8294 +/- 0.1596 | 0.8764 +/- 0.1048 |
| Spatiotemporal (best config, 16/8/12) | **0.8690 +/- 0.1243** | **0.8667 +/- 0.1247** | **0.8794 +/- 0.1226** | 0.8667 +/- 0.1296 |

## Window sensitivity (spatiotemporal model)
| Config (`window/step/max`) | Accuracy | Balanced Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| 8/4/24 | 0.8357 +/- 0.1293 | 0.8417 +/- 0.1205 | 0.8294 +/- 0.1596 | 0.8764 +/- 0.1048 |
| 8/8/12 | 0.8548 +/- 0.1328 | 0.8542 +/- 0.1334 | 0.8488 +/- 0.1601 | **0.9028 +/- 0.1287** |
| 12/6/16 | 0.7905 +/- 0.1733 | 0.7958 +/- 0.1733 | 0.8143 +/- 0.1528 | 0.8556 +/- 0.1928 |
| 16/8/12 | **0.8690 +/- 0.1243** | **0.8667 +/- 0.1247** | **0.8794 +/- 0.1226** | 0.8667 +/- 0.1296 |

## Stage-wise ablation (learnable prior/mask)
| Stage | Accuracy | Balanced Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Stage-0 baseline | 0.8524 +/- 0.0497 | 0.8333 +/- 0.0645 | 0.8695 +/- 0.0533 | 0.8708 +/- 0.1491 |
| Stage-1 reweight | 0.8524 +/- 0.0497 | 0.8333 +/- 0.0645 | 0.8695 +/- 0.0533 | 0.8708 +/- 0.1491 |
| Stage-2 prior + reweight | 0.8357 +/- 0.0749 | 0.8167 +/- 0.0816 | 0.8395 +/- 0.1229 | 0.8708 +/- 0.1491 |
| Stage-3 prior + reweight + mask | 0.8190 +/- 0.0905 | 0.8000 +/- 0.0928 | 0.8262 +/- 0.1333 | 0.8694 +/- 0.1591 |

Conclusion: in the current 61-subject setting, simple top-k graph selection is more stable than heavier edge-level learnable masking.

## Result files (main)
- Stable static 5x10: `outputs/metrics/runs/static_graphvector_lr_5x10/summary_5x10.json`
- Stable spatiotemporal 5x10: `outputs/metrics/runs/hybrid_spatiotemporal_graphvector_lr_5x10/summary_5x10.json`
- Static best: `outputs/metrics/runs/dualgraph_multiband_topk_full_cv10/gnn_dualgraph_multiband_topk_cv10.json`
- Spatiotemporal 8/4/24: `outputs/metrics/runs/spatiotemporal_cv10/gnn_dualgraph_multiband_spatiotemporal_cv10_e200_matched.json`
- Spatiotemporal 16/8/12: `outputs/metrics/runs/spatiotemporal_cv10/gnn_dualgraph_multiband_spatiotemporal_cv10_e200_w16_s8_m12.json`
- Stage-wise ablation: `outputs/metrics/runs/learnable_prior_mask_cv10/stagewise_prior_learnable_topk_cv10_e200.json`
