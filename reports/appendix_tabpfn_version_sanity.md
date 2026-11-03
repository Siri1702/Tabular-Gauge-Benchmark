# Appendix B.5 — Cross-version sanity check for the TabPFN wrapper

The Act-2 multi-seed benchmark was run with `tabpfn==8.0.8` (see `REPRODUCIBILITY.md`). This appendix is the audit trail for any future reproduction attempt on a different `tabpfn` release, showing that the wrapper code in `tabular_gauge/models/tabpfn_model.py` matches the contract of the installed version.

## B.5.1 Check 1 — Version contract

- Installed version: **8.0.8**
- Expected version (from REPRODUCIBILITY.md): **8.0.8**
- Version match: **YES**
- All required kwargs present in `TabPFNClassifier.__init__`.

Full `TabPFNClassifier.__init__` signature (snapshot):

```
auto_scale_n_estimators, average_before_softmax, balance_probabilities, categorical_features_indices, device, differentiable_input, eval_metric, fit_mode, ignore_pretraining_limits, inference_config, inference_precision, keep_cache_on_device, memory_saving_mode, model_path, n_estimators, n_jobs, n_preprocessing_jobs, random_state, self, show_progress_bar, softmax_temperature, tuning_config
```

## B.5.2 Check 2 — Smoke test (fit + predict + AUC)

- n_train = 80, n_test = 40
- predict_proba shape = [40]
- ROC-AUC = **1.0000**  (must be > 0.5 on linearly-separable data)
- fit time = 3.709 s, predict time = 4.150 s
- context bookkeeping: context_used=80, context_original=80, was_subsampled=False
- AUC > 0.5: **YES**

## B.5.3 Check 3 — Subsampling past the context window

- Input n_train = 12000 (above the 8 000-row context cap)
- max_context_size = 8000
- After fit: context_used = 8000, was_subsampled = True
- Subsampled to max: **YES**

## B.5.4 Check 4 — Determinism

- AUCs across 3 fresh instantiations (same seed): [1.0, 1.0, 1.0]
- max − min = 0.00e+00
- Deterministic (Δ < 1e-6): **YES**

## B.5.5 Check 5 — `ignore_pretraining_limits=True` accepted

- Accepted by `tabpfn.TabPFNClassifier`: **YES**

If `ignore_pretraining_limits` is not accepted, the wrapper would silently fall back to the cloud `tabpfn_client` and invalidate all Act-2 headline numbers. A reviewer reproducing this paper should re-run this script first.

## B.5.6 Verdict

**PASS — wrapper contract verified for tabpfn==8.0.8**
