# Tabular-Gauge-Benchmark — Phase-by-Phase Execution Plan

> **Last updated:** 2026-08-02
> **Reading order:** start with §1 (audit), skim §2 (master timeline), then
> drill into any phase with a green card.
> **Cadence:** each phase closes with a deliverable doc + git tag; nothing
> rolls forward without that.

---

## 1. Audit of what's already in the repo (before we add anything)

### 1.1 Code surface
- `src/models/` — wrappers for TabPFN, SAP-RPT-1, XGBoost, LightGBM, CatBoost, MLP.
- `src/data/loader.py` — OpenML dataset pull + cached JSON.
- `src/evaluation/` — `metrics.py`, `statistical.py` (Friedman, Wilcoxon, bootstrap CI), `analysis.py`, `profiling.py` (timing + peak memory).
- `src/analysis/` — `few_shot.py`, `learning_curves.py`, `ensembles.py`, `sap_rpt_few_shot.py`.
- `src/viz/` — heatmap, leaderboard, statistical, curves, dashboard.
- `experiments/run_benchmark.py`, `experiments/regenerate_report.py`.
- `scripts/` — 12 ad-hoc analysis scripts (few-shot sweeps, crossover analysis, robustness, SAP-RPT-1 few-shot).

### 1.2 Empirical surface
- 15 binary-classification OpenML tables, N∈[455, 48k], p∈[4, 100].
- 5 seeds × 6 models main run.
- Few-shot (25/50/100/200/300/500/750/1000) on 6 datasets.
- Crossover curve on Adult.
- Calibration ECE/MCE; ranking + CD diagram.

### 1.3 What reviewers will demand (and we don't yet have)

| Demand | Today's answer | Phase that fixes it |
|--------|----------------|---------------------|
| Multi-class + regression | Only binary | Phase 3 |
| Distribution-shift evaluation | None | Phase 2 |
| Open-source TFM refresh (TabICL v2, TabDPT, LimiX) | Only TabPFN, SAP-RPT-1 | Phase 1 |
| Equal wall-clock budget | GBDT tuned with 15 Optuna trials, FM zero-shot | Phase 5 |
| Calibration under shift | None | Phase 2 / Phase 4 |
| Fairness / subgroup | None | Phase 4 |
| Pinned env + dataset hashes | Caches exist but no `conda lock`, no `sha256sum` registry | Phase 0 |
| Cost-quality Pareto | Timing only | Phase 5 |
| Multi-seed paired tests | Bootstrap only | Phase 0 (cheap) |

---

## 2. Master timeline (weeks)

```
Phase 0  Repo hygiene + reproducibility scaffold              [W0   – W2]
Phase 1  Model refresh + dataset refresh                      [W3   – W6]
Phase 2  Distribution-shift + feature-shift track              [W7   – W12]
Phase 3  Multi-class + regression tracks                       [W13  – W17]
Phase 4  Fairness + missingness + calibration depth            [W18  – W21]
Phase 5  Budgeted evaluation + cost-quality Pareto             [W22  – W25]
Phase 6  Integration: hub paper + dashboard release            [W26  – W30]
```

Total: ~30 weeks (≈ 7 months). Compatible with NeurIPS 2026 D&B (deadline
~May 2026) if we front-load Phases 0–2; with JMLR continuous review (rolling).

---

## Phase 0 — Repo hygiene & reproducibility scaffold

**Objective:** clean the working tree and guarantee one-button reproduction of
*any* past experiment.

### 0.1 Inventory & cleanup (already in motion)

| Action | Status |
|--------|--------|
| Remove `catboost_info/tmp/`, `catboost_info/learn/`, training logs | ✅ done |
| Remove `__pycache__/` everywhere | ✅ done |
| Delete stale scratch docs (`ANALYSIS_PLAN.md`, `IMPROVEMENT_PLAN.md`, `comment.md`, `paper_i/`) | ✅ done |
| Drop `experiments/results/last_run.log` | ✅ done |
| Drop `reports/linkedin_posts.md` | ✅ done |
| Consolidate 4 overlapping planning docs into `PUBLICATION_STRATEGY.md` + `PHASE_PLAN.md` (this file) | ✅ done |
| Single source of truth for citations → `REFERENCES.md` | ✅ done |

### 0.2 Reproducibility scaffolds (new)

- `requirements.lock` — frozen pip resolver. `pip install -r requirements.lock`.
- `environment.yml` with `channels: conda-forge` + python 3.11 + tabpfn + sap_rpt stack.
- `Dockerfile` — `FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04` + every dep; entrypoint `python experiments/run_benchmark.py --all --seeds 5`.
- `scripts/hash_datasets.py` — emits `dataset_hashes.json` with SHA-256 of every cached OpenML file; CI re-verifies.
- `scripts/print_environment.py` — collects `pip freeze`, `nv-smi`, OpenML versions for paper supplementary.
- Add `.gitattributes` so LaTeX diffs cleanly in PRs.

### 0.3 Statistical-pair-testing helper
- New `src/evaluation/paired_stats.py` — implements **paired bootstrap** (same train/test splits across models) and **Bayesian signed-rank** for per-seed significance. The current code uses unpaired bootstrap.
- Rationale: pairs reduce variance ≈ 3×; reviewers familiar with AutoML stat testing will flag unpaired tests.

### 0.4 Success criteria
- `make reproduce` (new Makefile) re-runs every saved experiment and matches hashes within 1e-3 AUC.
- `pytest tests/test_pipeline_smoke.py` green.
- Repository linting: `ruff`, `black --check`, `mypy --strict src/`.

### 0.5 Deliverable
- Tag `v0.1-clean` on `main`.

---

## Phase 1 — Model refresh & dataset refresh

**Objective:** match the **state of the art** for tabular FMs as of 2026-08.

### 1.1 New model integrations

| Model | License | Why include | Effort |
|-------|---------|------------|--------|
| **TabICL v2** (INRIA/SODA) | Apache 2.0 | Open-source SOTA on TabArena + TALENT; 10.6× faster than TabPFN-2.5 | 3 d |
| **TabDPT** | Apache 2.0 | Self-supervised on 123 real datasets, strong regression | 3 d |
| **LimiX** | Research only | Joint distribution LDM; representative of scaling-law family | 2 d |
| **Mitra** | Research only | Curated synthetic-prior mixture; competitive on mid-size | 2 d |
| **Real-TabPFN v2.5** (if license permits) | Non-commercial | TabPFN-2.5 outperforms tuned GBDTs; need non-commercial code path for paper | 2 d (conditional) |
| Existing: **TabPFN v2, SAP-RPT-1, XGBoost, LightGBM, CatBoost, MLP** | — | Keep all | — |

### 1.2 New datasets
- Replace hill-valley + qsar-biodeg (single-domain) with more diverse coverage:
  - **HELOC** (Shift, finance, 10k) — already on TableShift
  - **Mushroom** (8k, biology) — for high p/N
  - **Higgs** (11M, physics) — for scaling ceiling
  - **Adult-Census** (already used for crossover) — also for OOD
  - **Credit-Card** (300k, finance) — for cost-quality story
- Promote `configs/datasets_few_shot.yaml` to cover multi-class OpenML ID 1590 (Adult has ≥2 categorical strata beyond binary target).
- Pull curated **multi-class** + **regression** sets from Jojo 2025 (e.g., `diamonds`, `nyc-taxi-green`, `higgs-small`).

### 1.3 Architectural changes
- Refactor `src/models/base.py` to enforce a `ModelInterface` with `fit(X, y, t_budget_s=None)` and `predict_proba(X)`. Required so Phase 5 budgeted runs work uniformly.
- All new model wrappers must implement a `memory_log` callback to feed Phase 5.

### 1.4 Statistical sweep (5 seeds minimum)
- Run every (model, dataset) cell with **5 seeds**, store full `y_pred` per fold in `experiments/results/raw/`.
- Pre-compute **per-seed** metrics so Phase 0's paired tests are real.

### 1.5 Success criteria
- `experiments/results/aggregated/results_v1.csv` exists, contains 7+ models × 20+ datasets × 5 seeds.
- New CD diagram shows **3 distinct ranking bands** (best, middle, worst) rather than overlapping GBDT cluster.
- Time budget: 2 weeks compute (assumes 4 × A100 80GB available).

### 1.6 Deliverable
- `models_v1` report and `v1-models` git tag.
- New `models.yaml` schema validated.

---

## Phase 2 — Distribution-shift & feature-shift track

**Objective:** ship **Paper 2** (the shift audit). This is the single most
publishable near-term increment because every related TFM paper from 2025–26
has the same gap (L1 in `PUBLICATION_STRATEGY.md`).

### 2.1 Datasets
- **TableShift** (Gardner 2023): HELOC, Childhood Lead, Voting.
- **TabFSBench** (Cheng 2025): at least 3 of the 4 shift scenarios.
- **Synthetic shift generator** (`src/data/shift.py`) for:
  - Label shift: flip prevalence 0.5 → 0.1 and 0.9.
  - Concept shift: shuffle feature-target relationship.
  - Feature shift: drop 25%/50% of features at inference time.
  - Sparsification shift: zero-inflate 30%/60% of cells.

### 2.2 Models under shift
All 7+ models from Phase 1.

### 2.3 Metrics
- ID ROC-AUC vs OOD ROC-AUC (**shift gap**).
- ID ECE vs OOD ECE.
- **Critical difference per shift family** (not just one global CD diagram).
- Bootstrap CI on shift gap.

### 2.4 Statistical claims target
- Show that no TFM is **shift-immune**; ranking changes by ≥2 ranks in at least one shift family.
- Show that real-data-pretrained models (Real-TabPFN, ConTextTab) trade absolute accuracy for slightly larger shift gap.
- Show that **stacking TFMs** on top of XGBoost reduces shift gap while preserving calibration (the synthesis claim).

### 2.5 Deliverable
- `reports/shift_audit.md` with 5 figures.
- Draft of Paper 2 (8 pages) targeting **NeurIPS 2026 D&B**.

### 2.6 Success criteria
- All 7+ models evaluated on ≥3 shift families.
- p<0.05 paired test that "shift changes model ranking" for at least one shift family.

---

## Phase 3 — Multi-class & regression tracks

**Objective:** remove the binary-classification blinders (L4 in
`PUBLICATION_STRATEGY.md`).

### 3.1 Datasets
- Multi-class (k=3..15): `car`, `satimage`, `letter`, `isolet`, `mnist-tabular` (n=10k subset), `covertype-small` (sampled).
- Regression: `california-housing`, `diamonds`, `nyc-taxi-green-dec-2024`, `Higgs-small` (re-cast as regression), `wine-quality-white`.

### 3.2 Metrics
- **Multi-class**: log-loss, top-1 accuracy, macro F1, **multi-class ECE** (Gruber & Buettner 2023), per-class reliability diagrams.
- **Regression**: RMSE, MAE, R², calibration-of-quantiles (TabICL v2 emits 999 quantiles — use them).
- Replace ROC-AUC with PR-AUC where class imbalance > 1:9.

### 3.3 Changes to metrics module
- `src/evaluation/metrics.py` — add `multiclass_ece`, `quantile_calibration`, `regression_metrics`.
- `src/evaluation/paired_stats.py` — already model-agnostic, no changes.

### 3.4 Deliverable
- `reports/multiclass_regression.md`.
- Supplementary table in Paper 1 (hub).
- Optional standalone workshop submission at AutoML 2026.

---

## Phase 4 — Fairness, missingness, calibration depth

**Objective:** cover L5 (missingness), L6 (cal under shift), L11 (fairness).

### 4.1 Missingness
- Inject MCAR/MAR/MNAR masks at fractions {0.1, 0.3, 0.5} on a 6-dataset subset.
- Report degradation curve: AUC vs missing fraction per model.
- Note: SAP-RPT-1 and ConTextTab have natural-language column names; semantically encode "missing" vs literal NaN; treat both.

### 4.2 Fairness
- Datasets with protected attributes: UCI Adult, COMPAS recast, German Credit (already in repo).
- Metrics: **demographic parity**, **equalized odds**, **calibration across groups**.
- Examine whether in-context examples bias TFM predictions more than GBDT baselines.
- Cf. "Towards Fair ICL with Tabular FMs" (OpenReview 2026).

### 4.3 Calibration depth
- Reliability diagrams per dataset per model.
- **Expected-Confidence-Errors** (AUROC of confidence vs accuracy).
- Calibration under shift (overlap with Phase 2).

### 4.4 Deliverable
- `reports/fairness_missingness.md`.
- Inputs to Papers 2 and 4 (FAccT submission).

---

## Phase 5 — Budgeted evaluation & cost-quality Pareto

**Objective:** answer the practical question "which model for which budget?"
(L2, L3, L7). This drives Paper 3 (KDD ADS).

### 5.1 Budgeted evaluation design
For every (model, dataset) cell:
- `t_budget ∈ {0s, 30s, 120s, 600s, 3600s}`.
- `0s` ⇒ zero-shot / no tuning.
- `>0s` ⇒ run that many seconds of HPO (Optuna or Hyperband), record best params so far.
- Record wall-clock, peak GPU mem, total FLOPs (via `flopth`).

### 5.2 Cost-quality Pareto
- Plot AUC vs log wall-clock per (model, dataset).
- Plot AUC vs log GPU-hour.
- Compute **Pareto frontier** per dataset; aggregate via mean rank on frontier distance.

### 5.3 Outcome expected
- Real-TabPFN/ConTextTab may dominate the *0s* budget.
- XGBoost + Optuna should dominate the *3600s* budget.
- A **mid-budget** (120s) niche might be the best ROI for practitioners — claim to publish.

### 5.4 Deliverable
- `reports/budget_sweep.md` with cost-quality heatmap.
- Interactive dashboard (Streamlit) deployed on Hugging Face Spaces — public artefact reviewers can click.

---

## Phase 6 — Integration & submission prep

**Objective:** assemble the hub paper and ship a public dashboard.

### 6.1 Hub paper (Paper 1)
- 12-page JMLR article:
  - §2: Related work (cite `REFERENCES.md`).
  - §3: Unified experimental protocol.
  - §4: Main benchmark (binary + multi-class + regression).
  - §5: Few-shot (extended to more datasets, 5 seeds).
  - §6: Shift audit (Phase 2).
  - §7: Cost-quality Pareto (Phase 5).
  - §8: Calibration + fairness (Phase 4).
  - §9: Discussion, decision framework, limitations.

### 6.2 Public artefacts
- Docker image pushed to `ghcr.io/Siri1702/tabular-gauge-benchmark`.
- Zenodo bundle of all CSVs + raw JSON.
- Streamlit dashboard (continuously updated from `results.csv`).
- arXiv preprint of Paper 1 immediately after JMLR submission.

### 6.3 Reproducibility badge
- Pass NeurIPS's reproducibility checklist (seeds, env hashes, dataset hashes, ≤2 hours CPU reproduction for the headline result).

### 6.4 Submission calendar
- Submit Paper 2 (shift) → NeurIPS D&B deadline.
- Submit Paper 3 (budget) → KDD ADS deadline.
- Submit Paper 1 (hub) → JMLR continuous review.
- Submit Paper 4 (fairness) → FAccT / AAAI 2027.

---

## 3. Datasets registry (master list)

> Use these OpenML ids when adding configs. IDs verified during the
> literature search; re-confirm at add time.

### Binary classification (carryover)
- 31 credit-g • 37 diabetes • 44 spambase • 1063 kc2 • 1067 kc1 • 1068 pc1
- 1050 sick • 1462 banknote-auth • 1479 hill-valley • 1487 ozone-level • 1489 phoneme
- 1494 qsar-biodeg • 1510 wdbc • 40536 australian • 40945 titanic

### Distribution shift (Phase 2 — TableShift)
- HELOC • Childhood Lead • Voting (Gardner 2023)

### Distribution shift (Phase 2 — TabFSBench)
- 4 scenarios from Cheng et al. 2025 (ICML)

### Multi-class (Phase 3)
- 9 letter • 21 car • 22 mfeat-factors • 188 eucalyptus
- 32 satimage • 50 tic-tac-toe • 54 vehicle • 1493 isolet
- 41169 coil2000 (small) • 1597 covertype (sampled 50k)

### Regression (Phase 3)
- 537 houses • 41066 nyc-taxi-green-dec-2024 • 42728 diamonds
- 44965 wine-quality-white • 44964 wine-quality-red
- 23512 higgs-small (10% subsample)

### Large scale (Phase 5 / cost-quality)
- 23512 higgs (11M) • 41163 airlines (539K) • 41147 creditcard (300K)
- 41159 click-prediction-small • 1218 higgs-only-2M

### Fairness (Phase 4)
- 31 credit-g (gender+age) • 1590 adult-census • 45022 compas-scores-two-years

---

## 4. Models registry

| Family | Models | Source | Phase introducing |
|--------|--------|--------|-------------------|
| Tabular FM (open-source) | TabPFN v1/v2, Real-TabPFN v2.5 | PriorLabs | existing + Phase 1 |
| Tabular FM (open-source) | TabICL v1/v2, LimiX, Mitra, TabDPT | INRIA, anonymous, LMU | Phase 1 |
| Semantics-aware ICL | SAP-RPT-1 (ConTextTab), ContextTab | SAP | existing + Phase 1 |
| GBDT (tuned) | XGBoost, LightGBM, CatBoost, HistGradientBoosting | standard | existing + Phase 5 budgeted |
| AutoML baseline | AutoGluon 1.4 (4h extreme mode) | autogluon | Phase 5 (paper 3 baseline) |
| Deep tabular | MLP (already there), FT-Transformer, ResNet, SAINT | rtdl | Phase 1 |
| Linear baseline | Logistic regression w/ preprocessing | sklearn | sanity baseline |

---

## 5. Metrics (master list)

| Group | Metric | Used in phase |
|-------|--------|---------------|
| Discrimination (binary) | ROC-AUC, PR-AUC, F1-macro | 0/1/2/5 |
| Discrimination (multi-class) | top-1 acc, macro F1, log-loss, ROC-AUC OvR | 3 |
| Discrimination (regression) | RMSE, MAE, R² | 3 |
| Calibration | ECE-10, MCE, multi-class ECE (Gruber 2023), Brier, AUROC(confidence vs accuracy) | 0/2/3/4 |
| Resource | wall-clock (fit + predict), GPU mem peak, FLOPs | 5 |
| Cost | $/100k predictions, GPU-hour | 5 |
| Robustness | shift gap, rank change, expected ID-OOD regression | 2 |
| Missingness | AUC vs missing-fraction | 4 |
| Fairness | demographic parity, equalized odds, calibration-by-group | 4 |
| Statistical | Friedman, Nemenyi, paired Wilcoxon, paired bootstrap CI | every phase |

---

## 6. Risk register

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|-----------|
| Gated models (SAP-RPT-1, possibly Real-TabPFN-2.5) unavailable | Med | High | Fallback: report gated-model results in supplementary; open-source models form the headline claims |
| Compute shortfall for 1M-row runs | Med | Med | Use sub-sampling with explicit justification; publish "wall-clock efficiency" curves |
| New TFM ships mid-phase | High | Low | Treat as refresh — log a "snapshot date" in README; results table notes version |
| Reviewer finds a forgotten limitation | High | Med | Maintain a living `LIMITATIONS.md`; update after each phase |
| Single-author bandwidth | High | High | Hub-and-spoke split lets each paper stand alone; if a phase slips, Paper 2 (shift) can ship first |

---

## 7. "What we promised" check

By the end of Phase 6 we will have:
- 20+ datasets, 7+ models, 3 task types (binary, multi-class, regression).
- 3 shift families + 4 feature-shift scenarios.
- 4–5 seeds for every cell.
- Public Docker, Zenodo bundle, dashboard.
- 4 paper drafts queued or submitted.
- One living `LIMITATIONS.md` documenting what we *didn't* cover.

The single open question is **Phase 6 hub paper venue**: JMLR vs. Nature
Communications vs. MLJ. Decide at end of Phase 3 once we know whether the
clinical/health angle has traction.
