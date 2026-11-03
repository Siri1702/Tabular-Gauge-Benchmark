# Publication Strategy — Single Paper vs Multi-Paper Split

> **Decision target:** Where to send the work.
> **Refresh rule:** Re-run this analysis once Phase 1 results are in.
> Last updated: 2026-08-02.

---

## 1. What's already in the repo (audit)

| Component | Status | Strength | Gap |
|-----------|--------|----------|-----|
| 15 OpenML binary classification tables | ✅ | Cleansed metadata, N≤10k constraint satisfied by design | No multi-class, no regression, no shift scenarios |
| 6 models (TabPFN, SAP-RPT-1, XGBoost/LightGBM/CatBoost/MLP) | ✅ | Mix of zero-shot FM + tuned GBDT + deep baseline | Missing newer open-source TFMs (TabICL v2, LimiX, TabDPT, ConTextTab) |
| Main benchmark at full N | ✅ | TabPFN wins 67%, 12× speedup over XGBoost — clean headline numbers | SAP-RPT-1 few-shot is gated (HuggingFace) |
| Few-shot (25, 50, 100, 200, 300, 500, 750, 1000) | ✅ | First **systematic** few-shot sweep across N for any TFM | Only 6 datasets, single seed for some sizes, no CatBoost/MLP at large N |
| Crossover on Adult (48K) | ✅ | Crossover N ≈ 1.5k–3k | One dataset, single curve, multi-dataset crossover needed |
| Calibration (ECE, MCE) | ✅ | Distinct ECE reported per model | Only binary classification; no reliability diagram by domain |
| Friedman + Nemenyi CD + Wilcoxon + 95% bootstrap CI | ✅ | Standard for benchmarks | Not yet applied to **paired** per-seed differences |
| Figures: heatmap, CD diagram, win/loss matrix, learning curves, timing | ✅ | All publication-grade PNG+PDF | No calibration reliability grid, no compute-vs-quality Pareto |
| HuggingFace gating → SAP-RPT-1 few-shot blocked | ⚠ | — | Single point of failure for reproducing Row 2 model |

**What the repo clearly does NOT cover** (and what every top-tier benchmark
reviewer will demand):

1. Multi-class classification (2–10 classes beyond binary).
2. Regression tasks.
3. Out-of-distribution / temporal drift / feature-shift evaluation.
4. Independent re-implementation/audit of newer open-source TFMs (TabICL v2, LimiX, TabDPT).
5. Cost-vs-quality Pareto (FLOPs, \$, wall-clock, GPU mem).
6. Statistical comparison under **per-seed** paired tests with multiple baselines.
7. Reproducibility kit: pinned environments, dataset hashes, command-line driver.
8. Calibration under distribution shift.

These gaps **drive every limit** covered in the next section.

## 2. Limitations of *existing* benchmarks — what the literature says

From our survey (see `REFERENCES.md`):

| # | Limitation | Citation evidence |
|---|------------|-------------------|
| L1 | **IID bias.** Most TFM benchmarks evaluate in-distribution. On TableShift-family tasks every TFM degrades 0.003–0.060 ROC-AUC. | Loza et al. 2026, Gardner 2023 |
| L2 | **Single-config bias.** TabPFN/TabRepo identified that single hyperparameter evaluations over-report winners. | TabRepo 2024 |
| L3 | **Multi-config/tuning inequality.** Mix of zero-shot TFMs and *tuned* GBDTs is unfair unless GBDT tuning budget is documented. | TabArena 2024, AMLB AutoML Conf 2025 |
| L4 | **Task narrowness.** Tabular benchmarks are 80%+ binary classification. Multi-class and regression are token-presence-only. | Jojo 2025 (multi-class/regression dataset paper) |
| L5 | **Missingness / dirty data not modeled.** Real tables have MCAR/MAR/MNAR gaps; no benchmark evaluates systematically. | AMLB 2025 (deployment realism) |
| L6 | **Calibration rarely reported.** Multi-class calibration is mostly ignored; reliability under shift is not. | Kull 2019, Gruber 2023 |
| L7 | **Cost/quality trade-offs absent.** Memory, \$, energy not in published metrics. | TabRepo 2024 (acknowledged) |
| L8 | **Feature shift not tested.** TabFSBench shows feature support shifts break tabular models. | TabFSBench 2025 |
| L9 | **TFM evaluation is model-specific.** Each new TFM (v2.5, v3) needs benchmark refresh within months. | TabPFN-2.5 paper 2025, TabICL v2 2026 |
| L10 | **Reproducibility complaints.** Pinned seeds, hashes, env files often missing. | Practice across all of the above |
| L11 | **No fairness / subgroup analysis.** Even when classification is binary, bias is unmeasured. | "Towards Fair ICL with Tabular FMs" 2026 |
| L12 | **Coverage of large-scale / high-dim is sparse.** TFMs have a scaling ceiling (LimiX failed cross-attention on Voting at scale). | Loza 2026 |

## 3. Where our existing repo already partially addresses these

| Limitation | Our current answer | How new work should strengthen it |
|-----------|--------------------|----------------------------------|
| L1 OOD / shift | None | Add a Phase 2 swap. Pull in **TableShift** HELOC, Childhood Lead, Voting; introduce **TabFSBench** subsets |
| L2 single-config | Limited — GBDTs tuned via Optuna 15 trials, TFMs zero-shot | Phase 5 introduces **bandit/TPE tuning** with **equal wall-clock budgets** per model |
| L3 tuning inequality | Documented (Optuna + trial counts) | Move to **budgeted evaluation**: report AUC vs 0/30s/120s/3600s wall-clock |
| L4 task narrowness | Only binary classification | Phase 3 adds **multi-class** + **regression** tracks from Jojo 2025 + OpenML-CTR23 |
| L5 missingness | Missing | Phase 4 injects **MCAR/MAR/MNAR masks** at fixed fractions |
| L6 calibration | ECE/MCE binary | Add **multi-class ECE**, Brier, **drift reliability diagrams** |
| L7 cost-quality | Timing + peak memory only | Add **GPU-hours, FLOPs, $/100k preds**, Pareto plots |
| L8 feature shift | None | Adopt TabFSBench scenarios directly; embed as **module** |
| L9 model refresh | SAP-RPT-1 in pipeline | Add **TabICL v2, TabDPT, LimiX** (open weights only) |
| L10 reproducibility | Has seeds + caches, no env lock | Pin `conda lock` + dataset SHA256 + docker image |
| L11 fairness | None | Phase 4 computes **demographic parity / equalized odds** on UCI Adult, CelebA sub |
| L12 scaling | Limited crossover on Adult | Add **1M+ row tasks** (Higgs, Airline Delay, Criteo) |

## 4. Single-paper vs multi-paper decision matrix

| Option | Pros | Cons | Verdict |
|--------|------|------|---------|
| A. One **mega-paper** covering all of the above | Single coherent narrative, one reviewer cycle, one citation | Reviewers may desk-reject as "too broad"; risks shallow coverage; harder to fit 8-page NeurIPS/ICML | ✗ if forced to scope-fit |
| B. **Hub-and-spoke**: one journal-grade main paper + 2–3 satellite papers | Each paper is crisp; reviewer pool differs (ML, applied data mining, HCI/UX); citations cross-reference cleanly | More writing overhead; need staggered timing | **RECOMMENDED** ✓ |
| C. Conference first, journal later for the hub only | Faster review on sprint tracks; journal version gives room for full budget-OOD story | Two-stage review is more work | **RECOMMENDED** as a tactical path within B |

### The proposed split (Hub-and-spoke)

#### Paper 1 — **Hub: Tabular-Gauge-Benchmark, an Open Audit of Tabular Foundation Models** (≥16 pages, journal)
- Venue target: **JMLR** (open-access ML journal), fallback **Nature Communications** if clinical/health data angle emphasised, **Machine Learning Journal** as second backup.
- Contents: all 12 limitations summarised in a long-form, reproducible-computing format. Companion **Docker + Nextflow** pipeline.
- Status: requires everything in Phase 1–6 done first.

#### Paper 2 — **Satellite: Where Tabular Foundation Models Break — A Distribution-Shift Audit** (8 pages, conference)
- Venue target: **NeurIPS 2026 D&B**, fallback **ICML 2027 Workshops on Distribution Shifts**.
- Contents: L1 (shift gap), L8 (feature shift), part of L6 (cal under shift), L12 (scaling). Mirrors TableShift but with **TPMS diversity** (3-4 shift sources) and 7+ TFMs.

#### Paper 3 — **Satellite: Compute Quality Frontiers — A Budgeted Evaluation of Tabular ML** (8 pages)
- Venue target: **KDD 2026/2027 Applied Data Science track**, fallback **ICML AutoML workshop**.
- Contents: L2/L3/L7 + reproducibility framework + cost-quality Pareto.
- Strong industrial-appeal angle.

#### Paper 4 — **Satellite (optional): Fairness & Calibration of TFMs** (6 pages)
- Venue target: **FAccT 2027**, **AAAI 2027**.
- Contents: L6 + L11.

### Stagger plan (viable for a single author with advisor)

| Week | Output | Trigger |
|------|--------|---------|
| W+0 | Phase 0 + Phase 1 done (repo clean, public launch of results) | — |
| W+6 | Paper 2 (shift) first draft complete; submit to NeurIPS D&B | Hard deadline target |
| W+12 | Paper 3 (budget) first draft; submit to KDD | Hard deadline target |
| W+20 | Paper 4 (fairness) draft; submit to FAccT or AAAI | Rolling |
| W+24 | Paper 1 (hub) draft; submit to JMLR (open review continuous) | Continuous |

## 5. Venue fit card

| Paper | Primary venue | Probability (informed guess) | Backup | Why this venue |
|-------|---------------|-----------------------------|--------|----------------|
| Hub | JMLR | 35% | Machine Learning Journal, Nature Communications | Open-access, reproducibility celebrated, no page squeeze — fits a *benchmark* paper |
| Satellite shift | NeurIPS 2026 D&B | 25% | ICML 2027 Datasets & Benchmarks | D&B track explicitly welcomes systematic studies with community artefacts |
| Satellite budget | KDD 2026/2027 ADS | 30% | AutoML conference 2026 | Industrial/applied focus matches cost story |
| Satellite fairness | FAccT 2027 | 20% | AAAI 2027 | Cross-disciplinary ML + social concerns |

Probability estimates assume a single-author paper with one advisor; raising to 50%+ requires (a) open dataset release, (b) co-authors from at least one of the original TFM teams, (c) reproducibility badge from NeurIPS.

## 6. "Minimum publishable unit" criteria (MVC for each paper)

| Paper | Minimum artefact | Required empirical evidence |
|-------|-----------------|----------------------------|
| Hub | Docker image + Zenodo bundle of all CSVs | ≥3 seeds for **every** (model, dataset, task-type) cell; CD diagram; reliability diagrams; failure-mode taxonomy |
| Shift | Public leaderboard on Hugging Face Spaces | Shift gaps with 95% CI; statistical comparison to XGBoost; ranking change from IID→OOD |
| Budget | Public dashboard with cost-quality curves | Wall-clock @ fixed $/run comparison; reproducibility badge |
| Fairness | Group-by-group plots with demographic parity, equalized odds | Calibration per subgroup; ensembling cures |

---

## 7. Decision

Proceed with **Hub-and-spoke**, three papers for sure, four if fairness tracks
open up. The current repo's strengths (15 OpenML tables, 6 models, few-shot,
crossover) make Paper 2 (shift) the easiest **first** submission — it explicitly
extends the work that's already in the repo and is directly tied to the open
TFM literature (L1, L6, L8).
