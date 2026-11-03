# Paper 1 — Next Best Step, Rigour Audit, and Concrete Plan

> **Date stamped:** 2026-08-02
> **Audience:** the first author (student) — read this and the previous
> `PAPER1_GUIDE.md` together. This file is the **what to do this week**; the
> guide is the **how to write** when the data is ready.
> **Companion docs:** `PHASE_PLAN.md`, `PUBLICATION_STRATEGY.md`,
> `REFERENCES.md`, `LIMITATIONS.md`, `PAPER1_GUIDE.md`.

---

## 1. The verdict in one paragraph

The current repo gives you **a strong skeleton for a benchmark paper, not a
ready-to-submit paper**. The headline numbers (TabPFN AUC 0.932, +11.2% over
XGBoost at N=25, crossover at N≈1.5k–3k) are real and impressive, but they
sit on a **single-seed** evaluation, a **binary-only** dataset family, no
distribution-shift audit, no cost-quality Pareto, and an **unpaired**
statistical protocol. Those four gaps are exactly the dimensions where
TabArena, TabRepo, BeyondArena, Loza 2026 and TabFSBench show the existing
literature is strongest. Top reviewers will flag every one. The plan below
ranks the additions by return on investment and gives the title, scope, and
timeline for Paper 1.

---

## 2. Rigour audit of the existing numbers

I opened the raw JSONs, the aggregated CSV, the few-shot CSV, and the
crossover summary to check what is solid and what isn't. Here's what I found.

### 2.1 What is robust already

| Finding | Evidence | Verdict |
|---------|----------|---------|
| TabPFN mean AUC 0.932 across 15 binary OpenML tables | `experiments/results/raw/*.json` (15 datasets), 5 seeds on the main run, all 6 models in `experiments/results/aggregated/results.csv` | **Robust** — 5 seeds × 6 models × 15 datasets = 450 cells, each with full per-fold metrics saved |
| Friedman + Nemenyi CD diagram | `src/evaluation/statistical.py` + `reports/figures/cd_diagram.png` | **Method sound**, but tests are unpaired (reviewers will note this) |
| Bootstrap CI on per-dataset AUC | `src/evaluation/statistical.py` | **Method sound**, unpaired |
| Calibration ECE-10, ECE-15, MCE | every JSON row has these columns | **Robust** |
| Memory + timing profiling | every JSON row has peak_memory_mb, fit_time, predict_time, throughput | **Robust** |
| Crossover point ≈ N=3k on Adult | `experiments/results/crossover/crossover_summary.txt` shows N=3000, XGBoost leads TabPFN by 0.0016 | **Direction correct**, but the run is **single-seed and 3 models only** (TabPFN, XGBoost, LightGBM); CatBoost and MLP missing |

### 2.2 What is **not** robust today

| Finding | Issue | Severity for paper |
|---------|-------|--------------------|
| Few-shot "TabPFN wins at N < 500" headline | `experiments/results/few_shot/extended_few_shot_results.csv` has **seed=0 only** for every (model, dataset, N) cell. Every claim of "advantage" is therefore a point estimate with no confidence interval. | **High** — auto-rejectable |
| Single-seed learning curves | Same — no shaded std bands in `reports/figures/few_shot_learning_curves*.png` | **High** |
| Cross-over run is single-seed | N=50 to full on Adult only | Medium |
| Crossover is 3 models only (no CatBoost, MLP) | Decision tree in the paper cannot cover all 6 models | Medium |
| No paired tests | Wilcoxon and bootstrap are **unpaired** across model seeds | Medium-high |
| No shift evaluation | Loza 2026 + TableShift + TabFSBench all in 2024–26; reviewers from the TFM community will demand it | High |
| Only 1 of 3 SAP-RPT-1 variants integrated | SAP-RPT-1 few-shot is gated by HuggingFace; only main-benchmark SAP-RPT-1 numbers exist | Medium |
| No newer TFMs (TabICL v2, LimiX, Mitra, TabDPT, Real-TabPFN v2.5) | Models that exist since 2025–26 are missing; headlining the "TFM audit" requires them | High |
| No multi-class, no regression | 12 of the 15 binary-only datasets are easy OpenML tables; the literature has moved on | High for journal, Medium for NeurIPS D&B |
| No cost-quality Pareto | Wall-clock is reported but never plotted against AUC; the practitioner takeaway is hidden | Medium |
| No reproducibility kit (Docker, dataset hashes, env lock) | Required for NeurIPS reproducibility badge and JMLR open-review | Medium |
| Per-seed predictions not all stored | Raw JSON has predictions, but the few-shot CSV is per-cell (no predictions); can't replay | Low |

**Bottom line:** the headline numbers are real and the methodology is
correct in spirit, but the experimental matrix is too narrow (binary only)
and too sparse (single-seed) for a top-venue submission today.

---

## 3. The Paper 1 I would actually write

Given the audit above, the Paper 1 I recommend is **NOT** the "submit the
existing repo" paper. It is the paper that results from a focused 6–8 week
compute + writing sprint that closes the four largest gaps.

### 3.1 Working title

> **"Tabular-Gauge-Benchmark: A Paired Audit of Tabular Foundation Models Across
> Binary, Multi-Class, and Regression Tasks at IID and Shifted Regimes"**

Alternative shorter titles:
- *"When Tabular Foundation Models Win: A Multi-Regime Audit"*
- *"An Open, Reproducible Audit of Tabular Foundation Models"*

### 3.2 Topics covered

The Paper 1 scope is a deliberate *subset* of the full hub vision
(`PUBLICATION_STRATEGY.md`) that can ship to **NeurIPS 2027 D&B** (deadline
~May 20, 2027):

| Section | What it contains | What's already in repo | What to add |
|---------|------------------|------------------------|-------------|
| 1. Introduction | Why tabular FMs matter; the gap in current benchmarks; 4 contributions | Motivation paragraph; key numbers | Sharpen the gap paragraph; add cost-quality + shift axes |
| 2. Related Work | TabPFN family, TabICL v2, ConTextTab, LimiX, Mitra, TabDPT, GBDT, AutoML benchmarks, shift, calibration | TFMs paragraph already exists | Refresh with 2025–26 papers; add TabArena, BeyondArena, TableShift |
| 3. Protocol | Datasets, models, metrics, paired-stat protocol | Methods already documented | Add SHA-256 dataset hashes; add paired bootstrap; document env lock |
| 4. Main results | 25+ datasets × 6+ models × 5 seeds (paired) | Current 15 binary × 6 models × 5 seeds | Add 5 multi-class + 5 regression datasets |
| 5. Few-shot analysis | N=25…full with **paired 95% CI** | Single-seed curves | Re-run with 5 seeds; paired bootstrap CI; Cliff's δ |
| 6. Crossover & scaling | N=50→full on 3 datasets (Adult, Higgs, Creditcard) | Adult only, 3 models | Add Higgs-small + Creditcard; add CatBoost, MLP, LimiX |
| 7. Distribution-shift audit | TableShift HELOC/Childhood Lead/Voting + TabFSBench 4 scenarios | None | Run all 7+ models on all 3+4 scenarios |
| 8. Cost-quality Pareto | AUC vs wall-clock + AUC vs $/100k preds | Wall-clock only in JSONs | Plot Pareto frontier; integrate with §4 |
| 9. Discussion | Decision tree; when to use which model | Decision tree in README text | Formalise as Figure 12 |
| 10. Limitations | Mapped to `LIMITATIONS.md` | None formal | Write |

### 3.3 Topics to *not* include in Paper 1 (defer to Papers 2–4)

- **Fairness / subgroup analysis** → Paper 4 (FAccT).
- **Missingness stress test** → Paper 4.
- **Multi-class imbalance-specific protocols** → Paper 3 (KDD).
- **TabPFN-Wide, Orion-MSP, TabFlex** → too bleeding-edge; mention only in limitations.
- **Time series** → out of scope.

### 3.4 Contributions (paper-worthy claims)

After the compute sprint, these will be the 4–5 citable claims:

1. **First paired-seed audit of TFMs.** Across 25+ datasets × 7+ models × 5 seeds, we use paired bootstrap CI and Wilcoxon with Holm-Bonferroni correction. Most prior benchmarks report unpaired statistics.
2. **First multi-regime audit.** Single matrix crossing task type × data regime × distribution shift; existing benchmarks are single-axis.
3. **Crossover quantified across multiple datasets, not one.** Crossover point shifts with p, missingness, and class imbalance; we measure the distribution, not the point estimate.
4. **Cost-quality Pareto.** XGBoost + Optuna 50 trials is Pareto-optimal at large N and long budget; TabPFN/Real-TabPFN-v2.5 is Pareto-optimal at small N and zero budget; AutoGluon dominates mid-budget. A practitioner decision rule falls out of the curve.
5. **Public, refreshable leaderboard.** New TFMs can be added in one line of YAML; CI runs every PR. This is an artefact, not just a paper.

### 3.5 Headline numbers after the compute sprint (target ranges)

These are the numbers you should aim for once the additions are run:

| Claim | Today | Target |
|-------|-------|--------|
| Datasets covered | 15 binary | 25 (15 binary + 5 multi-class + 5 regression) |
| Models covered | 6 | 8–10 (add TabICL v2, TabDPT, LimiX, Real-TabPFN v2.5 if license permits) |
| Seeds per cell | 1–5 (mixed) | 5 throughout |
| Few-shot N values | 8 | 8 (same) |
| Crossover datasets | 1 (Adult) | 3 (Adult + Higgs-small + Creditcard) |
| Shift datasets | 0 | 7 (3 TableShift + 4 TabFSBench) |
| Multi-class datasets | 0 | 5 |
| Regression datasets | 0 | 5 |
| Reproducibility artefact | Partial | Docker + Zenodo + dataset hashes |
| Paired statistical tests | 0 | 3 (paired Wilcoxon, paired bootstrap, Cliff's δ) |
| Cost-quality curves | 0 | 2 (vs wall-clock, vs $/100k) |
| Calibration under shift | 0 | 3 shift families × 4 datasets |
| **Acceptance probability** | **5–15%** | **45–60%** |

---

## 4. Publication timeline (realistic, July 2026 baseline)

| Phase | Calendar weeks | Activities | Cumulative outcome |
|-------|----------------|------------|---------------------|
| **W0–2** (Aug 2026) | Compute | Phase 0: reproducibility kit + paired statistical helper + multi-seed orchestration | Repo is locked; one-command reproduce |
| **W3–6** (Sep–Oct 2026) | Compute | Phase 1: add TabICL v2, TabDPT, LimiX, Mitra; refresh TabPFN to v2 if needed; full 5-seed run on existing 15 binary tables | Headline benchmark reproduced on 5 seeds |
| **W7–10** (Oct–Nov 2026) | Compute | Phase 3 mini-track: 5 multi-class + 5 regression datasets × 7 models × 5 seeds | Multi-class/regression tables |
| **W11–14** (Nov–Dec 2026) | Compute | Phase 2: TableShift × 7 models × 5 seeds + TabFSBench × 7 models × 5 seeds | Shift audit complete |
| **W15–18** (Dec 2026–Jan 2027) | Compute + analyse | Phase 5 mini: cost-quality Pareto on 5 representative datasets × 4 budget tiers; reliability diagrams | Figures 8, 9, 10 |
| **W19–22** (Jan–Feb 2027) | Compute | Crossover on 3 datasets with 7 models × 5 seeds | Crossover plot Figure 5 |
| **W23–28** (Feb–Mar 2027) | Write | First complete draft using `PAPER1_GUIDE.md` outline | v1 draft |
| **W29–30** (Apr 2027) | Polish | Internal review by supervisor; figures; reproducibility kit | v2 draft |
| **W31** (early May 2027) | Submit | NeurIPS 2027 D&B submission | **Submission** |
| **W32–48** | Wait + rebuttal | Author response if reviewer response comes by Aug 2027 | Rebuttal by Sep 2027 |
| **W49–52** (Oct–Dec 2027) | Revise | Camera-ready | Publication |

**Total end-to-end: 11 months from August 2026 to publication.**

If you skip Phase 2 (shift) and Phase 5 (cost-quality) — i.e. submit a
"binary-only audit" — you can compress the timeline to **6 months** and
target **NeurIPS 2026 D&B** (deadline already passed at May 22, 2026), or
shift to **KDD 2027 ADS** (~Feb 2027) or **ICML 2027 D&B** (~Jan 22, 2027).

**Strong recommendation:** do **not** compress. The shift audit and the
cost-quality Pareto are the two findings that distinguish this paper from
every other tabular benchmark published since 2025. Without them, you are
submitting a paper that TabArena has already beaten on coverage.

---

## 5. What to start working on **this week**

The single highest-ROI next step is to **make the few-shot numbers defensible**. Specifically:

### Step 1 — Re-run few-shot with 5 seeds (3 days)

Edit `scripts/run_extended_few_shot.py` to set `--n_seeds 5` and re-run on the 6 few-shot datasets (diabetes, wdbc, australian, credit-g, titanic, banknote-auth) at all 8 N values. Store per-seed predictions, not just per-cell aggregates. Total compute: ~1 hour per (model, dataset, N) cell × 5 models × 6 datasets × 8 N values × 5 seeds ≈ 60 GPU-hours on a single A100.

### Step 2 — Implement paired statistical tests (2 days)

New file `src/evaluation/paired_stats.py` with:
- `paired_bootstrap_diff(a, b, n_resamples=10000)` — returns 95% CI of `a-b`.
- `paired_cliffs_delta(a, b)` — effect size.
- `holm_bonferroni(pvalues)` — family-wise correction.

Use it on the new 5-seed runs to produce a per-N paired p-value heatmap (Figure 5 in `PAPER1_GUIDE.md`).

### Step 3 — Add dataset SHA-256 hashes + Docker (1 day)

`scripts/hash_datasets.py` walks `experiments/results/raw/*.cache.json` and emits `dataset_hashes.json`. `Dockerfile` with python 3.11 + pinned `requirements.lock`. Both ship in v0.1 of the repo.

### Step 4 — Lock the writing setup (1 day)

- Create Overleaf project with NeurIPS D&B LaTeX template.
- Import BibTeX from `REFERENCES.md`.
- Pin figure list and table list from `PAPER1_GUIDE.md` §5–6.

**After W1 of this plan you will have:**
- 5-seed few-shot curves with shaded 95% CI.
- Paired statistical tests that survive reviewer scrutiny.
- A reproducible Docker image.
- An empty Overleaf draft with the right template and bibliography.

These four deliverables are the **prerequisites for anything else to work**. Do them first.

---

## 6. How to organise the next 6–8 weeks

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1 | Steps 1–4 above | Reproducible 5-seed few-shot; paired tests; Docker; Overleaf draft |
| 2 | Refresh models: add TabICL v2 and TabDPT | Two new wrappers in `src/models/`; results on 15 datasets × 5 seeds |
| 3 | Run Phase 2 mini-eval: TableShift HELOC + Voting on 6 models × 5 seeds | Shift-gap box plot Figure 6 |
| 4 | Run cost-quality mini: 5 datasets × 4 budgets × 5 seeds | Figure 8, 9 |
| 5 | Multi-class mini-track: 5 datasets × 6 models × 5 seeds | Multi-class results table |
| 6 | Crossover on 2 more datasets (Higgs-small, Creditcard) | Crossover plot Figure 5 |
| 7 | Calibration under shift + reliability diagrams | Figure 10 |
| 8 | First complete paper draft | v1 draft ready for supervisor review |

After W8, follow the 14-week writing schedule in `PAPER1_GUIDE.md` §11.

---

## 7. Suggested title — final recommendation

> **"Tabular-Gauge-Benchmark: A Paired, Multi-Regime Audit of Tabular Foundation Models"**

Sub-title candidates:
- *"Across Binary, Multi-Class, and Regression Tasks at Few-Shot, Full-Size, and Shifted Regimes"*
- *"With a Public, Reproducible Leaderboard and Cost-Quality Pareto Frontier"*

Short-form for talks:
> *"When Tabular Foundation Models Win — and When They Don't."*

---

## 8. Risks and contingencies

| Risk | Mitigation |
|------|------------|
| Compute shortfall for Higgs-small (11M rows) | Sub-sample with explicit "wall-clock ceiling" sub-experiment; report at 10K, 100K, 1M, 11M |
| Real-TabPFN-2.5 license blocks non-commercial paper | TabPFN v2 (open) + TabICL v2 + LimiX is the open-source trio; explain in limitations |
| Paper gets stuck at NeurIPS desk rejection | Submit as a JMLR rolling submission instead; convert to NeurIPS 2027 D&B if accepted first |
| Reviewer wants an unfair-comparison defence | Phase 5 cost-quality Pareto is the formal defence — "all models shown at their respective budget tiers, not in a single column" |
| Shift data hard to access | TableShift is publicly hosted on GitHub; TabFSBench scenarios can be reproduced from public datasets with documented preprocessing |

---

## 9. The single sentence that summarises this whole document

> *The numbers you have are real, the methodology is right in spirit, and the
> paper is one focused 6–8 week compute-and-writing sprint away from being
> a strong NeurIPS 2027 D&B submission — but only if you close the
> paired-seed, shift, cost-quality, and multi-class/regression gaps first.*

Pin this sentence. Re-read it whenever you are tempted to submit the current
repo as-is.
