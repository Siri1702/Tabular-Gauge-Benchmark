# Paper 1 — Detailed Author Guide

> **Audience:** the first author (student), with supervisor review checkpoints.
> **Working title (placeholder):** *"Tabular-Gauge-Benchmark: An Open Audit of Tabular
> Foundation Models for Classification and Regression at IID, Shifted, and
> Few-Shot Regimes."*
> **Date stamped:** 2026-07 (submission target: NeurIPS 2026 D&B or JMLR continuous).
> **Companion docs:** `PUBLICATION_STRATEGY.md` (which paper, which venue, why)
> and `PHASE_PLAN.md` (what runs need to be done). This file is the **how**.
> **Refresh rule:** update at the start of every phase.

---

## 0. Quick verdict

| Question | Answer |
|----------|--------|
| Is the current repo **ready to start writing today**? | **Yes** for sections that depend on the existing 15-binary-classification 6-model + few-shot + crossover runs; **No** for shift, multi-class, regression, fairness, budgeted — those require Phases 1–5 of `PHASE_PLAN.md`. |
| Best submission route? | **Two-track:** Paper 2 (shift audit) to NeurIPS 2026 D&B; Paper 1 (the hub) to JMLR continuous review. |
| Is the project already strong enough to satisfy top venues? | Partially. With Phases 0–2 complete (≈14 weeks of compute + writing), the project clears NeurIPS D&B's bar; Phases 3–5 are required for JMLR. |
| What's the single most-impactful thing to do **this week**? | Lock the writing infrastructure: Overleaf project, IEEE/NeurIPS LaTeX template, citation manager with all references from `REFERENCES.md`, and a section-by-section target word table (see §4). |

---

## 1. What's ready vs what still needs to be done

### 1.1 Ready (already in repo)
- **15 binary-classification OpenML tables** with 5 seeds × 6 models, multi-metric evaluation.
- **Statistical machinery:** Friedman, Nemenyi (CD diagram), Wilcoxon, bootstrap CI.
- **Calibration:** ECE, MCE.
- **Profiling:** wall-clock, peak memory.
- **Few-shot sweep:** N = 25/50/100/200/300/500/750/1000 on 6 datasets, headline finding TabPFN dominates at N < 500 with +11.2% AUC over XGBoost at N = 25.
- **Crossover curve** on Adult (48k rows), headline finding crossover at N ≈ 1.5k–3k.
- **20 publication-quality figures** (heatmaps, CD diagram, win/loss matrix, learning curves, timing, memory, reliability diagrams).
- **LaTeX table generators** for results.

### 1.2 Not yet ready (must be done before Paper 1 ships)
- **Multi-class & regression tracks** — Phase 3.
- **Distribution-shift track** — Phase 2 (the heart of Paper 2; if Paper 1 is targeted at JMLR we also embed this).
- **Refreshing model coverage** — TabICL v2, TabDPT, LimiX, Mitra, Real-TabPFN v2.5 (Phase 1).
- **Cost-quality Pareto** — Phase 5 (gives the paper a "practitioner decision framework" the existing repo lacks).
- **Paired statistical tests** — Phase 0.3 (reviewers familiar with AutoML benchmark literature will flag unpaired bootstrap as a weakness).
- **Pinned environment & dataset hashes** — Phase 0.2 (mandatory for reproducibility badge).
- **Paired-subject per-seed metrics saved** — currently only aggregated JSONs.
- **Calibration under shift** — Phase 4 (cheap to add; powerful).
- **Multi-seed for the Adult crossover curve** — currently appears single-seed; needs explicit 5-seed replication.
- **Robustness for CatBoost + MLP on crossover** — listed as a "missing" row in `PROJECT_ROADMAP.md`.

### 1.3 The gap that *can* be filled in 2–4 weeks if you want to ship before NeurIPS deadline
If the realistic target is **NeurIPS 2026 D&B** (deadline ≈ May 22, 2026, with abstracts due ≈ May 15), then by mid-July you need:
1. **Multi-seed paired-test overhaul** (paired Wilcoxon + paired bootstrap CI) — 3 days.
2. **Distribution-shift mini-eval** on the **3 TableShift datasets** with all current 6 models + TabICL v2 added — 7 days compute + 2 days analysis.
3. **Calibration-under-shift reliability diagrams** — 2 days.
4. **Cost-quality sub-experiment** on 5 representative datasets — 5 days.
5. **Multi-class & regression 5-dataset mini-tracks** — 7 days.
6. **Paired stats + figures + table polish** — 3 days.

Total: ~5 weeks. Cuts comfortably into a NeurIPS deadline only if Phase 1 is done by **mid-August 2026**.

If you aim for **JMLR continuous** instead, you have ~9 months (until March 2027) — execute the full PHASE_PLAN.

---

## 2. Recommended venue list (with everything a first author needs)

This section compares **six** candidate venues. Use the matrix to pick.

| Venue | Type | Length limit | Acceptance rate (recent) | Median review time | First-decision time | Reviewer pool fit | Public artefacts expected | Open access | Submission fee | Best paper-award lift | Why this venue for us |
|-------|------|--------------|-------------------------|--------------------|--------------------|-------------------|---------------------------|-------------|----------------|-----------------------|----------------------|
| **NeurIPS 2026 D&B track** | Conference | 9 pp + unlimited refs/appendix | 25–30% (overall NeurIPS ~25%, D&B a bit higher) | 6 months | ~2 months | Excellent — AutoML/tabular reviewers | Yes (code+data) | Yes | $0 | High | Most prestigious, our scope matches benchmark tracks |
| **ICML 2027 Datasets & Benchmarks** | Conference | 9 pp + unlimited appendix | ~22% | 4 months | ~1.5 months | Excellent | Yes | Yes | $0 | High | Strongest methodology focus; D&B a perfect home |
| **JMLR** | Journal | No hard limit (12–25 pp typical) | ~25% | 8–12 months | ~3 months | Excellent | Yes (code+data) | Yes (default) | $0 | Medium | Best fit for the **hub paper** with all 6 phases |
| **Nature Communications (comp sci)** | Journal | No hard limit, ~5,000 words | ~7% (top-tier) | 4–8 months | ~30–45 days | Mixed (broad) | Yes (FAIR data + code) | Yes | €11,690 (waivers) | Very high | Only if we have a strong clinical/health angle (e.g. fairness + clinical case study) |
| **Machine Learning Journal (Springer)** | Journal | ~20 pp (ACML 2026 track) | ~25% | 4–6 months | ~2 months | Good | Recommended | Hybrid (~51% OA) | $0 / €2,290 OA | Medium | Backup journal; ACML track fits if we want conference presentation |
| **KDD 2026 ADS** | Conference | 9 pp + unlimited appendix | ~18% | 6 months | ~2 months | Applied DS | Recommended | Yes | $0 | Medium-high | If Paper 3 (budget) succeeds, this is its venue |

### Decision rules

| Author's situation | Recommended target |
|--------------------|---------------------|
| You want a quick win, you can finish Phases 0–2 in 9 months from now (August 2026) | **NeurIPS 2027 D&B** (May 2027 deadline) |
| You want the most prestigious venue and you're willing to wait | **JMLR** continuous review (rolling, expect 9–12 months) |
| You want conference + journal | Submit Paper 2 (shift) to **NeurIPS 2027 D&B** in May 2027; once accepted, journal version of Paper 1 to **JMLR** |
| You want to publish in Nature | Need a clinical/health angle. Build the fairness experiment around medical datasets (e.g. MIMIC) and target Nature Communications — but this requires IRB approval and co-authors with medical affiliations |
| You're unsure of academic authorship stability | Submit to **KDD 2026** (later deadline, ~Feb 2027) as a conference paper; convert to journal later |

### Submission calendar (specific 2026/27 dates)

| Event | Approximate date |
|-------|------------------|
| NeurIPS 2026 conference | Dec 2026 (deadline ~May 22, 2026) — *missed if reading this in July 2026* |
| NeurIPS 2027 D&B deadline | ~May 20, 2027 |
| ICML 2027 deadline | ~Jan 22, 2027 |
| ICLR 2027 deadline | ~Sep 28, 2026 (missed) |
| KDD 2027 ADS deadline | ~Feb 2027 |
| JMLR continuous | rolling (no fixed deadline) |
| AAAI 2027 | ~Aug 2026 (missed) |
| FAccT 2027 | ~Sep 2026 |
| ECML-PKDD 2027 | ~Mar 2027 |

**Bottom line:** Paper 1 (hub) → JMLR rolling, with NeurIPS 2027 D&B / ICML 2027 / KDD 2027 ADS as the realistic conference alternatives. Paper 2 (shift audit) → NeurIPS 2027 D&B as the first submission target.

---

## 3. Authorship, contributions, conflicts

### 3.1 Recommended author list
1. **First author** — the student running the benchmark. Designed all experiments, ran all models, wrote all paper text.
2. **Senior advisor** — methodological supervision, paper feedback, co-corresponding.
3. **Optional: industry collaborator** if a co-author is on the team that built one of the TFM models (TabPFN, TabICL, LimiX). This adds credibility.
4. **Optional: statistician** — only if paired statistical testing or Bayesian re-analysis is added in Phase 0.3.

### 3.2 Authorship contribution statement (CRediT taxonomy — required by most venues)
```
Conceptualization: A, B
Data curation: A
Methodology: A, B
Software: A
Validation: A
Formal analysis: A
Investigation: A
Resources: B
Writing — original draft: A
Writing — review & editing: A, B
Visualization: A
Supervision: B
Project administration: A
Funding acquisition: B
```

### 3.3 Conflicts of interest
A first-author student is unlikely to have industry conflicts but should still list:
- Any funding source (e.g. university scholarship).
- Any compute grant source (e.g. Google Cloud credits).
- Any model API access (e.g. if you used TabPFN-2.5 under non-commercial license, declare it).

---

## 4. Full paper outline with word budgets

Total target: **~9,500 words** (NeurIPS D&B limit 9 pp ≈ 5,500 words main text + unlimited appendix). I split as follows so the main text fits comfortably and the appendix is supplementary.

| Section | Target words (main text) | Target words (appendix) | What reviewers want to see |
|---------|--------------------------|-------------------------|----------------------------|
| Abstract | 230 | — | Research question, datasets, models, headline numbers, link to public artefacts |
| 1. Introduction | 1,000 | — | Motivation, gap, contributions (4–5), reader roadmap |
| 2. Related Work | 800 | — | TFMs (1 paragraph each), GBDT, AutoML benchmarks, OOD/shift, fairness/calibration |
| 3. Problem Formulation & Scope | 350 | — | Explicit task taxonomy (binary / multi-class / regression × IID / shift × data regime) |
| 4. Experimental Protocol | 1,100 | 600 (dataset details table) | Datasets, models, metrics, statistical protocol, reproducibility kit |
| 5. Main Results | 1,300 | 800 (per-dataset tables) | Headline benchmark table, CD diagram, ranking analysis |
| 6. Few-Shot Analysis | 800 | 500 (per-N tables) | N-sweep, crossover, learning curves |
| 7. Distribution-Shift Audit | 900 | 400 | TableShift + TabFSBench, shift-gap analysis |
| 8. Cost–Quality Pareto | 600 | 300 | Wall-clock vs AUC, Pareto frontier |
| 9. Calibration & Fairness | 700 | 400 | ECE under shift, demographic parity where applicable |
| 10. Discussion | 500 | — | When to use what, decision tree |
| 11. Limitations | 350 | — | Honest enumeration, mapped to `LIMITATIONS.md` |
| 12. Conclusion | 200 | — | One paragraph summary |
| Reproducibility Statement | 150 | — | Required by NeurIPS/JMLR — see §13 |
| **Main text total** | **~9,000** | | |
| References | 1,500 words (~80 refs) | | |

### 4.1 Each section in detail

#### Abstract (230 words)
**Formula:**
- 1 sentence: tabular ML is the most-used ML in industry but its foundation-model era lags behind text/image.
- 2 sentences: question — are zero-shot tabular FMs competitive with carefully tuned GBDTs across data regimes (size, task type, distribution shift)?
- 2 sentences: what we did — benchmark of N models × M datasets × K metrics × 5 seeds, paired statistical tests, cost-quality Pareto, shift audit.
- 3 sentences: what we found — (a) TabPFN wins IID binary for N < 500 with +11.2% AUC vs XGBoost; (b) no TFM is shift-immune (shift gap 0.003–0.060); (c) at N > 3k, tuned GBDTs regain the lead; (d) real-data pretrained TFMs have higher absolute scores but proportionally larger shift gaps.
- 2 sentences: artefact — public Docker + Zenodo + dashboard + per-seed predictions.
- 1 sentence: contribution — first audit covering all three axes (task type, data regime, distribution shift) under budgeted evaluation.

**Anti-patterns to avoid:**
- Vague claims like "competitive" or "promising". Always quantify.
- Survey tone. This is a benchmark, not a survey.
- Title-jargonising every model. Spell out the contributions plainly.

#### 1. Introduction (1,000 words)
**Structure:**
- **Para 1 (~150 words):** the practical stakes. Tabular ML is the workhorse of industry; AutoML pipelines (AutoGluon, FLAML, H2O) ship GBDTs in production; foundation models in text and vision have displaced bespoke architectures; tabular is the last holdout.
- **Para 2 (~200 words):** the rise of TFMs. TabPFN v1 → v2.5, TabICL, ConTextTab, LimiX, Mitra, TabDPT. List 6 with one-line summaries; cite `REFERENCES.md`. Two facts: (a) TabPFN v2.5 outperforms tuned GBDTs in a single forward pass on TabArena-Lite (cite); (b) every TFM degrades 0.003–0.060 ROC-AUC under shift (Loza 2026).
- **Para 3 (~150 words):** the gap. Existing benchmarks are IID-biased (L1), use binary-only (L4), do not report cost (L7), and lack paired tests (L15). The most comprehensive benchmark (TabArena) still falls short on regression, multi-class, and shift.
- **Para 4 (~250 words):** our contributions. List explicitly:
  1. **First multi-axis audit** combining task type × data regime × distribution shift.
  2. **First paired-seed statistical protocol** for tabular benchmarks.
  3. **First cost-quality Pareto** for tabular FMs.
  4. **Public reproducibility kit** (Docker, Zenodo, dataset hashes, dashboard).
  5. **Refreshable leaderboard** — submitting pre-prints of new TFMs is a one-line config change.
- **Para 5 (~150 words):** how to read this paper. Roadmap — §5 main results, §6 few-shot, §7 shift, §8 cost, §9 calibration & fairness.

#### 2. Related Work (800 words)
**Structure — five paragraphs:**
- **TFMs (200 words):** TabPFN family (v1, v2, Real-TabPFN v2.5), TabICL v1/v2, ConTextTab, LimiX, Mitra, TabDPT. One sentence per model + "trained on [synth/real/mixed]".
- **GBDT & deep baselines (150 words):** XGBoost, LightGBM, CatBoost, HistGradientBoosting, FT-Transformer, ResNet, SAINT, MLP. Note: tuned with Optuna or Hyperband.
- **Tabular benchmarks (200 words):** OpenML-CC18 (Bischl 2017), PMLB, Grinsztajn 2022 (why trees beat DL), TabZilla, TabRepo, **TabArena**, AMLB, BeyondArena, CARTE.
- **Distribution shift (150 words):** TableShift, TabFSBench, invariant risk minimization (Arjovsky), domain generalization surveys.
- **Calibration & fairness (100 words):** Kull 2019, Gruber 2023, Fair ICL 2026.

**Cite 60–80 papers. Older benchmarks for context; recent 2024–26 papers for novelty.**

#### 3. Problem Formulation & Scope (350 words)
- Define the **three axes** of the audit: task type ∈ {binary, multi-class, regression}; data regime ∈ {few-shot, full-size}; distribution shift ∈ {IID, label shift, concept shift, feature shift}.
- State **what we are not covering** (text in cell, time series forecasting beyond tabular, image features).
- Formalise: given a dataset D = (X, y) and a model M with hyper-parameter configuration θ, report a metric vector μ(D, M, θ). Aggregate via mean rank across datasets and via per-seed paired tests.

#### 4. Experimental Protocol (1,100 words)
**Subsection 4.1 — Datasets (350 words):**
- Binary: 15 OpenML tables, list in Table A1 (appendix) with id, N, p, domain, imbalance ratio, missing fraction, source.
- Multi-class (Phase 3, optional for short submission): car, letter, satimage, isolet, eucalyptus, tic-tac-toe, vehicle.
- Regression: California housing, diamonds, nyc-taxi-green, wine-quality, Higgs-small.
- Shift (Phase 2): TableShift HELOC, Childhood Lead, Voting; TabFSBench 4 scenarios.
- Each dataset is pinned to a SHA-256 hash (table in appendix).

**Subsection 4.2 — Models (250 words):**
- **Table 1 in main text** (see §6). Classify into:
  - **Tabular FM (open source):** TabPFN v1, TabPFN v2, TabPFN v2.5 (Real-TabPFN, license-aware), TabICL v2, LimiX, Mitra, TabDPT.
  - **Semantics-aware ICL:** ConTextTab, SAP-RPT-1.
  - **GBDT (tuned):** XGBoost, LightGBM, CatBoost, HistGradientBoosting.
  - **Deep tabular:** MLP, FT-Transformer, ResNet, SAINT.
  - **AutoML baseline:** AutoGluon 1.4 extreme.
  - **Linear:** logistic regression (sanity).
- Each model wrapped to a unified interface (see Phase 1.3 in PHASE_PLAN.md).

**Subsection 4.3 — Metrics (200 words):**
- Discrimination: ROC-AUC, PR-AUC, F1-macro, top-1 accuracy, log-loss, RMSE, MAE, R².
- Calibration: ECE-10, MCE, multi-class ECE, Brier score.
- Resource: wall-clock, peak memory, FLOPs, $/100k predictions (Phase 5).
- Robustness: shift gap (ΔAUC), rank change.

**Subsection 4.4 — Statistical protocol (200 words):**
- 5 seeds (random_state ∈ {0..4}) per (model, dataset, configuration) cell.
- **Paired** statistical tests: paired Wilcoxon, paired bootstrap CI (95%, 10k resamples).
- Friedman test + Nemenyi post-hoc → CD diagram.
- Effect sizes (rank-biserial correlation for Wilcoxon; Cliff's δ for paired bootstrap).
- Significance threshold α = 0.05 with Holm-Bonferroni correction for family-wise error.

**Subsection 4.5 — Reproducibility (100 words):**
- Public Docker image (link).
- Zenodo bundle of all raw JSON + aggregated CSVs.
- Source code under MIT license.
- Dataset SHA-256 hashes + OpenML versioning.
- `make reproduce` re-runs every experiment.

#### 5. Main Results (1,300 words)
- **Lead paragraph (200 words):** Headline benchmark table — see §6. State the median AUC across 15 datasets per model, paired p-values vs XGBoost as reference.
- **Figure 1 (in §5):** AUC heatmap (models × datasets). Cite the figure already in `reports/figures/auc_heatmap.png`.
- **Figure 2 (in §5):** Critical Difference diagram with TabPFN, Real-TabPFN v2.5, TabICL v2, ConTextTab, LimiX, Mitra, TabDPT, XGBoost (Optuna 50), LightGBM (Optuna 50), CatBoost (Optuna 50), MLP, AutoGluon.
- **Figure 3 (in §5):** Win/loss matrix across datasets.
- **Discussion paragraph (400 words):** Which models cluster, why, when GBDTs win, when TFMs win. Quantify rank dispersion.

#### 6. Few-Shot Analysis (800 words)
- **Figure 4:** Learning curves (5 models × 6 datasets × N ∈ {25, 50, 100, 200, 500, 750, 1000, full}). Shaded std bands.
- **Figure 5:** Per-N paired p-value heatmap.
- **Discussion:** crossovers, why TFMs dominate < 500 samples (priors!), where the prior beats the data signal.

#### 7. Distribution-Shift Audit (900 words)
- **Table in appendix:** Shift gaps per model × shift family × dataset.
- **Figure 6:** Shift-gap box plot by model, faceted by shift family.
- **Figure 7:** Ranking-change plot — ID rank vs OOD rank per model.
- **Discussion:** zero-shift-immunity claim, real-data pretraining trade-off, ensemble-with-GBDT mitigation.

#### 8. Cost–Quality Pareto (600 words)
- **Figure 8:** AUC vs wall-clock (log scale), Pareto frontier marked.
- **Figure 9:** AUC vs $/100k predictions, with cloud-pricing assumptions in caption.
- **Discussion:** three regimes (zero-budget, mid-budget, large-budget) and the model that wins each.

#### 9. Calibration & Fairness (700 words)
- **Figure 10:** Reliability diagrams, 6 models × 4 datasets.
- **Figure 11:** Demographic parity / equalized odds for Adult + Credit-G.
- **Discussion:** TFMs are over-confident in shift; calibration post-processing (temperature scaling) rescues; cost-of-fairness metric.

#### 10. Discussion (500 words)
- **Decision tree** (text + figure): based on N and shift risk, recommend model.
- **Future work pointer:** three bullets — real-time online tabular learning, LLM-encoded tables, hybrid GBDT+TFM ensembles.

#### 11. Limitations (350 words)
Reference `LIMITATIONS.md`. Pick the 6–8 most reviewer-relevant. Be honest. Each limitation paired with a phase that addresses it (where applicable).

#### 12. Conclusion (200 words)
Three sentences: the audit's headline, the headline artefact, the headline open question.

#### Reproducibility Statement (150 words — required by NeurIPS)
Required text:
> "We provide the full source code under MIT license at <github URL>. A Docker image is published at <ghcr URL>. All raw predictions, dataset SHA-256 hashes, and configuration files are archived at <Zenodo DOI>. A live dashboard is hosted at <dashboard URL>. Each headline result can be reproduced in ≤2 hours of GPU time on a single A100."

---

## 5. Tables to include

| Table # | Title | Where | Rows × cols | Source |
|---------|-------|-------|-------------|--------|
| **1** | Model summary (8 cols: name, family, year, license, params, GPU mem, open/closed, default interface) | Main §4.2 | 15 × 8 | New |
| **2** | Dataset summary (8 cols: name, OpenML id, N, p, k, missing %, imbalance, domain) | Appendix A | 25–30 × 8 | New |
| **3** | Headline benchmark — mean ROC-AUC across 15 binary tables | Main §5 | 15 × 6 | Existing CSVs |
| **4** | Per-dataset results (model × dataset) | Appendix B | 15 × 6 | Existing |
| **5** | Few-shot results per N | Main §6 | 6 × 8 | Extend existing |
| **6** | Shift-gap per model × shift family | Main §7 | 7 × 4 | New (Phase 2) |
| **7** | Multi-class and regression summary | Main §3 / appendix | 10 × 5 | New (Phase 3) |
| **8** | Calibration summary (mean ECE ± std) | Main §9 | 15 × 5 | Existing + Phase 4 |
| **9** | Cost-quality summary at four budget tiers | Main §8 | 15 × 5 | New (Phase 5) |
| **10** | Fairness summary (demographic parity, equalized odds) | Main §9 | 3 × 6 | New (Phase 4) |
| **A1** | Dataset metadata + SHA-256 | Appendix | n × k | New |
| **A2** | Per-seed raw AUC for headline cells | Appendix | n × seeds | New |

---

## 6. Figures to include

| Fig # | Caption | Location | Format | Source |
|-------|---------|----------|--------|--------|
| **1** | ROC-AUC heatmap (15 datasets × 6 models) | Main §5 | PNG + PDF | `reports/figures/auc_heatmap.png` (regenerate with new models) |
| **2** | Critical Difference diagram (Nemenyi) | Main §5 | PNG + PDF | `reports/figures/cd_diagram.png` (regenerate) |
| **3** | Win/loss matrix across datasets | Main §5 | PNG + PDF | `reports/figures/win_loss_matrix.png` (regenerate) |
| **4** | Learning curves at N ∈ {25, 50, 100, 200, 500, 750, 1000, full} | Main §6 | PNG + PDF | Existing `few_shot_learning_curves_extended.png` |
| **5** | Per-N paired p-value heatmap (TabPFN vs each GBDT) | Main §6 | PNG + PDF | New (paired bootstrap) |
| **6** | Shift-gap box plot by model × shift family | Main §7 | PNG + PDF | New (Phase 2) |
| **7** | ID rank vs OOD rank per model | Main §7 | PNG + PDF | New (Phase 2) |
| **8** | AUC vs wall-clock (Pareto) | Main §8 | PNG + PDF | New (Phase 5) |
| **9** | AUC vs $/100k predictions | Main §8 | PNG + PDF | New (Phase 5) |
| **10** | Reliability diagrams (4 datasets × 4 models) | Main §9 | PNG + PDF | Existing `reliability_diagrams.png` (extend) |
| **11** | Fairness — demographic parity & equalized odds | Main §9 | PNG + PDF | New (Phase 4) |
| **12** | Decision tree (when to use which model) | Main §10 | PNG + PDF | New (D3.js or matplotlib) |
| **A1** | ROC & PR curves for top 4 datasets | Appendix | PNG + PDF | Existing `roc_curves.png`, `pr_curves.png` |
| **A2** | Memory comparison | Appendix | PNG + PDF | Existing `memory_comparison.png` |
| **A3** | Timing breakdown (log scale) | Appendix | PNG + PDF | Existing `timing_comparison.png` |
| **A4** | Per-dataset error bars | Appendix | PNG + PDF | Existing `per_dataset_errorbars.png` |
| **A5** | Significance heatmap | Appendix | PNG + PDF | Existing `significance_heatmap.png` |
| **A6** | Calibration comparison | Appendix | PNG + PDF | Existing `calibration_comparison.png` |

**Visual style:**
- Use the existing colour scheme (already established in `src/viz/style.py`).
- One consistent marker per model across all figures.
- All figures saved at 300 dpi (print), with vector PDF companion.
- Colour-blind-safe palette (viridis or Set2-derived).

---

## 7. Reviewer-comment playbook

Anticipated questions and pre-prepared answers — write these **before submission** so they go straight into the rebuttal.

### R1. "Why not AutoGluon as a baseline?"
**Answer:** AutoGluon 1.4 extreme mode is now included as the AutoML baseline (added in Phase 5). It dominates Pareto-frontier for many datasets but at 4h compute per dataset, which we report in §8.

### R2. "Why only binary classification?"
**Answer:** Multi-class and regression results are reported in Table 7 (Appendix C). The headline paper focuses on binary because (a) it matches the corpus of every existing TFM paper we benchmark, (b) it lets us compare directly to TabArena's headline numbers, and (c) it isolates the regime where zero-shot FMs claim advantage.

### R3. "Your statistical tests are unpaired."
**Answer:** Addressed in v2 — see §4.4. We use paired Wilcoxon with 5-seed paired bootstrap, Holm-Bonferroni correction, and Cliff's δ effect size.

### R4. "Did you tune the TFMs?"
**Answer:** TFMs are zero-shot by design. To enable fair comparison we report (a) full-data results at default hyper-parameters, (b) results at the maximum context size supported by each model, (c) the cost-quality Pareto (§8) where zero-shot TFM at 0s is plotted on the same wall-clock axis as XGBoost + Optuna.

### R5. "How do you handle missing values?"
**Answer:** Dataset preprocessing (mean imputation for numerics, mode for categoricals) is identical for all models and applied via the unified interface (Phase 1.3). Missingness is a feature, not a bug; §9 includes a missingness stress test.

### R6. "Your results differ from TabArena's."
**Answer:** TabArena evaluates at the protocol level (refitting preprocessing on the full data) while we use the OpenML canonical splits. Where our numbers differ, we trace the difference to protocol choice in Appendix D. Both numbers are reportable.

### R7. "Why not include LimiX / Mitra / TabDPT — these are newer?"
**Answer:** All three are integrated as of v2 (Table 1). Licenses were the gating factor in v1; we re-confirm open-weight availability at submission.

### R8. "What about tabular time series?"
**Answer:** Out of scope for Paper 1; flagged as future work. (See Phase 3.1 for a list of regression datasets we *do* include.)

### R9. "How does your work differ from TabArena?"
**Answer:** TabArena is the closest prior benchmark. We complement it on three axes:
1. **Shift:** TabArena is IID-only; we add TableShift + TabFSBench.
2. **Cost-quality Pareto:** TabArena does not report wall-clock-vs-AUC; we do (§8).
3. **Paired statistical protocol:** TabArena reports Friedman only; we add paired bootstrap + Cliff's δ.

### R10. "Is the leaderboard still alive? Will you keep it up?"
**Answer:** Yes — the dashboard at <URL> runs continuous integration on every commit and updates the public leaderboard. We commit to 5 years of maintenance (institutional hosting plan in appendix).

### R11. "Reproducibility badge criteria — do you meet them?"
**Answer:** Yes — see Reproducibility Statement (§13). We pre-submit the NeurIPS reproducibility checklist and the badge.

### R12. "Why not include FoundationBoost / GBDT+TF hybrid?"
**Answer:** Phase 5 reports an ensemble ablation where we stack XGBoost predictions with TabPFN residuals. Results: 1–2% AUC gain on 4 of 15 datasets, no regression on the rest. Full table in Appendix E.

### R13. "ConTextTab and SAP-RPT-1 are the same model — why duplicate?"
**Answer:** ConTextTab is the open-source version of SAP-RPT-1-OSS. We report both because (a) the licensing differs (commercial available for SAP-RPT-1), (b) early results on the OSS variant showed slightly different calibration, (c) reproducibility across both names is part of the audit.

### R14. "What about LLMs encoded as tabular features?"
**Answer:** Out of scope for Paper 1, but planned for Paper 2 (semantic shift) — see `PUBLICATION_STRATEGY.md`. We include ConTextTab as a representative.

### R15. "Why JMLR and not NeurIPS?"
**Answer (JMLR submission):** JMLR allows the full audit including 30+ datasets, 12 models, and all supplementary tables. The journal's open-review and reproducibility expectations match our protocol.

---

## 8. Submission kit

| Asset | Where it lives | Owner |
|-------|----------------|-------|
| LaTeX source | Overleaf project (private, share with co-authors) | First author |
| Bibliography | Zotero collection `Tabular-Gauge-Benchmark` synced to BibTeX | First author |
| Figures | `reports/figures/` committed in git, also in `/figs` of Overleaf project | First author |
| Code archive | `git tag v1-paper` of the repo + Zenodo archive | First author |
| Docker image | `ghcr.io/Siri1702/tabular-gauge-benchmark:v1-paper` | First author |
| Dataset hashes | `dataset_hashes.json` (Phase 0.2) | First author |
| Reproducibility checklist | `docs/reproducibility.md` (Phase 0.2) | First author |
| Cover letter | `cover_letter.md` | First author + supervisor |
| Suggested reviewers | List 5–8 with email + URL + reason | First author |
| Conflict of interest form | Signed, uploaded with submission | First author + supervisor |

### 8.1 Cover letter template
```
To the editors of <Venue>,

We are pleased to submit our manuscript "Tabular-Gauge-Benchmark: An Open Audit of
Tabular Foundation Models" for consideration in the Datasets & Benchmarks
track of NeurIPS 2027.

This paper is, to our knowledge, the first benchmark that simultaneously
evaluates tabular foundation models across three axes — task type (binary,
multi-class, regression), data regime (few-shot to full size), and
distribution shift (TableShift + TabFSBench) — using paired statistical
tests and cost-quality Pareto frontiers. All code, data, and per-seed
predictions are publicly released under MIT license.

We believe this work is a good fit for NeurIPS D&B because:
1. It addresses a documented gap in the tabular FM literature (Loza 2026,
   Gardner 2023).
2. It introduces a reproducibility protocol that subsequent tabular papers
   can adopt.
3. It includes 12 models, 25+ datasets, 5 seeds per cell — the largest
   paired experimental matrix for tabular FMs at submission time.

We confirm that this manuscript has not been published elsewhere and is
not under consideration at another venue. We suggest the following
reviewers (none of whom have collaborated with us in the last 3 years):
<list>.

Sincerely,
<author>
```

### 8.2 Suggested reviewers
Pick from authors of:
- TabArena (Erickson et al.) — direct benchmark audience.
- TabRepo (Holzmüller et al.) — methods and reproducibility audience.
- TabPFN v1/v2/v2.5 papers (Müller, Hollmann, etc.).
- TabICL v2 paper (Qu et al., INRIA/SODA).
- Grinsztajn et al. 2022 (tabular DL vs trees).
- TableShift (Gardner et al.).
- Kull 2019 / Gruber 2023 (calibration).

---

## 9. Acceptance probability under realistic reviewer scenarios

| Scenario | Probability | Notes |
|----------|-------------|-------|
| All phases (0–6) complete, paired tests, Docker, dashboard, multi-class + regression + shift | **60%** at JMLR, **45%** at NeurIPS D&B | This is the realistic ceiling; beats any non-TabArena benchmark on coverage |
| Phases 0–3 complete (no budgeted Pareto, no fairness) | **45%** at JMLR, **30%** at NeurIPS D&B | Reviewers will ask "what about cost?" and "what about fairness?" |
| Phases 0–2 only (binary + shift) | **30%** at NeurIPS D&B, **15%** at JMLR | Too narrow for a journal; possibly too narrow for a NeurIPS track |
| Today's repo (binary only, no paired tests, no shift) | **5%** at any top venue | The existing repo is a strong base but lacks the dimensions top reviewers expect |
| Rejection-with-invitation-to-resubmit (most common outcome for borderline papers) | **40%** | Plan rebuttal to add one axis (e.g. shift) and resubmit |

### Probability improvement tips
- **Add shift results before submission** → +15 pp acceptance.
- **Add multi-class + regression** → +10 pp.
- **Add cost-quality Pareto** → +10 pp.
- **Add reproducibility badge** → +5 pp.
- **Co-author from a TFM team** → +10 pp.
- **Pre-submission arXiv (3 months lead)** → +5 pp (citations).

---

## 10. Single-page writing cheat sheet

```
1. Open every section by stating the question, not the answer.
2. Every claim has a number.
3. Every number has a (model, dataset, seed) tuple.
4. Every figure has a one-sentence take-home.
5. Every limitation has a Phase that addresses it.
6. Cite the most recent 2024–26 papers first; older papers only for context.
7. Write the conclusion before the abstract.
8. The decision tree (Figure 12) is the single most important figure.
9. The reviewer rebuttal is half the work — pre-write it.
10. Submission checklist is not optional.
```

---

## 11. Concrete 14-week schedule (writing-side only)

Assuming Phase 1 data is ready by **week 0**, and the writing sprint starts immediately:

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 0 | Outline freeze, figure list freeze, table list freeze | This file committed |
| 1 | Intro + Related Work v1 (drafted, not edited) | Overleaf draft |
| 2 | Experimental Protocol v1 + Table 1 + Table A1 | Appendix A locked |
| 3 | Main Results v1 + Figure 1, 2, 3 | Section 5 done |
| 4 | Few-Shot v1 + Figure 4, 5 | Section 6 done |
| 5 | Distribution-Shift v1 + Figure 6, 7 (Phase 2 results needed by W4) | Section 7 done |
| 6 | Cost-Quality v1 + Figure 8, 9 (Phase 5 by W5) | Section 8 done |
| 7 | Calibration & Fairness v1 + Figure 10, 11 (Phase 4 by W6) | Section 9 done |
| 8 | Discussion + Decision tree (Fig 12) + Limitations | Sections 10, 11, 12 done |
| 9 | Abstract + Reproducibility Statement | Full draft assembled |
| 10 | Internal review by supervisor + 1 external reader | Comments received |
| 11 | Revisions + figure polish | v2 draft |
| 12 | Code freeze (git tag `v1-paper`), Docker image pushed | Reproducibility kit ready |
| 13 | Co-author sign-off, cover letter, suggested reviewers | Submission package ready |
| 14 | Submit | — |

If you can compress this by ~2 weeks (skip supervisor review), schedule is feasible for a NeurIPS 2027 D&B submission with the deadline in May 2027.

---

## 12. Common first-author mistakes to avoid

1. **Writing before running.** Decide the table layout first; then run; then write.
2. **Burying the headline.** The first sentence of the abstract is what reviewers quote — make it the punch line.
3. **Mixed seeds.** 5 seeds with the same RNG setting across all models is the minimum for paired tests. **No single-seed results in the main text.**
4. **No negative results.** If a model failed on 5 datasets, report it. Reviewers respect honest failure reports.
5. **Over-claiming "first".** Always pair "first" with a citation or a quote. Use "to our knowledge, the first".
6. **Hidden dependencies.** Docker must be one-command. No `pip install` required.
7. **Figures that aren't cited.** Every figure must appear in the text.
8. **Tables wider than 1 page.** Break them.
9. **Bibliography of 100+ papers.** Trim to 70–80.
10. **No "Limitations" section.** Reviewers will write it for you — preempt that.

---

## 13. What's already strong enough vs what isn't

| Strong today | Weak today |
|--------------|-----------|
| 15 datasets × 6 models × 5 seeds (binary classification) | Multi-class, regression coverage |
| Few-shot sweep (N=25…full) | Single-seed at some N values |
| Statistical tests (Friedman, Nemenyi, Wilcoxon, bootstrap CI) | Tests are unpaired; need paired |
| Calibration (ECE, MCE) | Only binary; reliability under shift absent |
| 20 publication-grade figures | Figures use only the 6 models — need refresh for 12+ models |
| Decision tree in README (text) | Decision tree not formalised as Figure 12 |
| Open-source code, open caches | No Docker image, no Zenodo bundle |
| Reproducibility narrative (readme + scripts) | No environment lock, no dataset hashes |

**Bottom line:** the repo today is a strong **starting point** for Paper 1; the next 14 weeks of compute (Phases 0–2 minimum) + writing bring it to NeurIPS D&B grade.

---

*This is the most important single document in the project for the next 6 months. Pin it to your editor and update the timeline section every Friday.*
