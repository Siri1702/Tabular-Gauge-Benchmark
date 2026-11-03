# Dataset & Model Inventory

## 🛠️ Current Toolkit Models
These models are currently implemented and supported by Tabular-Gauge.

| Model | Wrapper | Version | Tuning Strategy | Hardware Requirement |
|---|---|---|---|---|
| TabPFN-v2 | `TabPFNWrapper` | `tabpfn==8.0.8` | Zero-shot | GPU Recommended |
| XGBoost | `XGBoostWrapper` | Latest | Optuna (10 trials) | CPU/GPU |
| LightGBM | `LightGBMWrapper` | Latest | Optuna (10 trials) | CPU/GPU |
| CatBoost | `CatBoostWrapper` | Latest | Optuna (10 trials) | CPU/GPU |
| MLP | `MLPWrapper` | Latest | Optuna (8 trials) | CPU/GPU |

## 📜 Historical Research Models
These models were used in the original research paper experiments.

| Model | Role | Result Availability | Current Support |
|---|---|---|---|
| SAP-RPT-1 | Stage I TFM Comparison | ✅ Full (Stage I) | ❌ Historical Only |

**Note on SAP-RPT-1:** This model was critical for the Stage I Foundation-Model selection process. While its results are preserved in the research artifacts, it is not part of the general-purpose toolkit.

## 📋 Datasets
The following datasets were used in the Stage II scaling study. All are sourced from OpenML.

| Dataset | OpenML ID | Domain | n_train | n_features | Availability |
|---|---|---|---:|---:|---|
| australian | 40536 | finance | 6702 | 120 | Public |
| banknote-auth | 1462 | security | 1097 | 4 | Public |
| kc1 | 1067 | software | 1687 | 21 | Public |
| ozone-level | 1487 | environment | 2027 | 72 | Public |
| phoneme | 1489 | audio | 4323 | 5 | Public |
| sick | 1050 | healthcare | 3017 | 37 | Public |
| spambase | 44 | text-features | 3680 | 57 | Public |
| titanic | 40945 | historical | 1047 | 13 | Public |
