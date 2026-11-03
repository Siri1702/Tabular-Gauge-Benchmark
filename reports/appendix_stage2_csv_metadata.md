# Appendix B — Stage-II CSV Metadata Audit

- Path: `C:\Users\siri\Desktop\Tabfm-benchmark\experiments\results\few_shot\multiseed_few_shot_results.csv`
- Rows: **1800**
- Cols: **48**
- Expected total cells: **1800** = 5 models × 8 datasets × 9 train sizes × 5 seeds
- Duplicate cells: 0 (must be 0)

## Columns

- `model` (`object`)
- `dataset` (`object`)
- `seed` (`int64`)
- `n_train` (`int64`)
- `n_test` (`int64`)
- `n_features` (`int64`)
- `fit_time_sec` (`float64`)
- `predict_time_sec` (`float64`)
- `total_time_sec` (`float64`)
- `tuning_time_sec` (`float64`)
- `total_wall_time_sec` (`float64`)
- `throughput_rows_per_sec` (`float64`)
- `ms_per_test_row` (`float64`)
- `peak_memory_mb` (`float64`)
- `memory_delta_mb` (`float64`)
- `gpu_memory_mb` (`float64`)
- `error` (`float64`)
- `roc_auc` (`float64`)
- `avg_precision` (`float64`)
- `f1_macro` (`float64`)
- `f1_weighted` (`float64`)
- `brier_score` (`float64`)
- `log_loss_val` (`float64`)
- `accuracy` (`float64`)
- `sensitivity` (`float64`)
- `specificity` (`float64`)
- `ppv` (`float64`)
- `ece_10` (`float64`)
- `ece_15` (`float64`)
- `max_calibration_error` (`float64`)
- `pr_auc` (`float64`)
- `optimal_threshold` (`float64`)
- `optimal_tpr` (`float64`)
- `optimal_fpr` (`float64`)
- `precision_class_0` (`float64`)
- `precision_class_1` (`float64`)
- `recall_class_0` (`float64`)
- `recall_class_1` (`float64`)
- `f1_class_0` (`float64`)
- `f1_class_1` (`float64`)
- `balanced_accuracy` (`float64`)
- `matthews_corrcoef` (`float64`)
- `demographic_parity` (`float64`)
- `tpr_overall` (`float64`)
- `fpr_overall` (`float64`)
- `train_size` (`object`)
- `actual_n_train` (`int64`)
- `domain` (`object`)

## AUC summary per model

| Model | n | mean AUC | median | min | max | std |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost | 360 | 0.8960 | 0.9210 | 0.4151 | 1.0000 | 0.1022 |
| LightGBM | 360 | 0.8710 | 0.9083 | 0.5000 | 1.0000 | 0.1333 |
| MLP | 360 | 0.8743 | 0.8952 | 0.3155 | 1.0000 | 0.1082 |
| TabPFN | 360 | 0.9101 | 0.9381 | 0.5493 | 1.0000 | 0.0914 |
| XGBoost | 360 | 0.8591 | 0.9055 | 0.5000 | 1.0000 | 0.1499 |

## TabPFN AUC by train size

| Train size | n | mean | std |
|---|---:|---:|---:|
| 100 | 40 | 0.8915 | 0.0904 |
| 1000 | 40 | 0.9392 | 0.0651 |
| 200 | 40 | 0.9160 | 0.0763 |
| 25 | 40 | 0.8469 | 0.1266 |
| 300 | 40 | 0.9184 | 0.0747 |
| 50 | 40 | 0.8610 | 0.1172 |
| 500 | 40 | 0.9291 | 0.0706 |
| 750 | 40 | 0.9353 | 0.0671 |
| full | 40 | 0.9539 | 0.0537 |

## TabPFN AUC by dataset (across all N + seeds)

| Dataset | n | mean | std |
|---|---:|---:|---:|
| australian | 45 | 0.8052 | 0.0559 |
| banknote-auth | 45 | 0.9999 | 0.0006 |
| kc1 | 45 | 0.7963 | 0.0593 |
| ozone-level | 45 | 0.8826 | 0.0763 |
| phoneme | 45 | 0.8729 | 0.0732 |
| sick | 45 | 0.9664 | 0.0425 |
| spambase | 45 | 0.9693 | 0.0207 |
| titanic | 45 | 0.9884 | 0.0056 |

## Notes for the paper

- Memory column present: **True** (`peak_memory_mb`).
- Inference time column present: **True** (`predict_time_sec`).
- Total raw cells = 1800.
- Zero duplicates = True.
