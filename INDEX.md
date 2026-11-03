# Tabular-Gauge-Benchmark — Documentation Index

> **Last refreshed:** 2026-11-03
> **Reading order:** read this file first. Follow the pointer that matches
> your goal; skip the others.

---

## What is in this repository

This repository benchmarks tabular foundation models against classical
baselines on a paired-by-seed protocol. The headline result is a
two-stage comparison: first we pick the better of the two open-weight
tabular foundation models (TabPFN-v2 and SAP-RPT-1) over 15 datasets and
5 seeds, then we take the winner (TabPFN-v2) and benchmark it against
four tuned baselines (XGBoost, LightGBM, CatBoost, MLP) at nine
training-set sizes on nine datasets with the same 5-seed paired
protocol. Paper 1 writes up this study.

---

## Pointer by goal

| If you want to … | Open this |
|---|---|
| **Understand Paper 1 end-to-end** (thesis, abstract, contributions, novelty, protocol, datasets, Act 1 + Act 2 + Act 3, results/figures inventories, next steps) | [`paper1/`](paper1/00_README/README.md) |
| **Verify Research Evidence** (the authoritative data and figures used in the paper) | [`docs/research/RESULTS_INDEX.md`](docs/research/RESULTS_INDEX.md) |
| **Run the benchmark** (install, customize, and generate your own results) | [`USER_GUIDE.md`](USER_GUIDE.md) |
| **Cite a specific metric** (definitions of ROC-AUC, ECE-10, log-loss, balanced accuracy, …) | [`METRICS_DOCUMENTATION.md`](METRICS_DOCUMENTATION.md) |
| **Add a dataset / model / test** | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| **Look up a paper in the related-work section** | [`REFERENCES.md`](REFERENCES.md) |

That's it. The files above are the only canonical entry points.

---

## Repository layout (canonical)

```
tabular-gauge-benchmark/
├── INDEX.md                              ← you are here
├── USER_GUIDE.md                         ← how to install & run (practitioner entry point)
├── METRICS_DOCUMENTATION.md              ← canonical metric definitions
├── CONTRIBUTING.md                       ← how to add datasets / models
├── REFERENCES.md                         ← citation pack for related work
│
├── paper1/                               ← canonical Paper-1 document set
│   ├── 00_README/README.md               ← entry point for Paper 1
│   ├── 01_agenda_and_thesis/             ← single-sentence thesis + 3-act outline
│   │   ...
│   └── 12_next_steps_checklist/          ← ordered single-action items, status-tagged
│
├── docs/
│   ├── research/                         ← research evidence & traceability
│   │   └── RESULTS_INDEX.md             ← maps paper claims to source data files
│   └── archive/                          ← legacy planning docs (read-only)
│
├── tabular_gauge/                        ← core package (model wrappers, metrics, viz)
├── experiments/                          ← runners + authoritative evidence
│   ├── paper_evidence/                  ← stable research artifacts (do not modify)
│   └── results/                          ← user-generated namespaced results
│
├── scripts/                              ← analysis scripts (see RESULTS_INDEX for canonical list)
├── reports/                              ← final publication tables and figures
└── requirements.txt
```

---

## Where to start

**If you are a Reviewer:** Start at [`docs/research/RESULTS_INDEX.md`](docs/research/RESULTS_INDEX.md) to see exactly how the paper's results map to the source data in `experiments/paper_evidence/`.

**If you are a Practitioner:** Start at [`USER_GUIDE.md`](USER_GUIDE.md) to set up the toolkit and run your own benchmarks.

**If you are reading the Paper:** Start at [`paper1/00_README/README.md`](paper1/00_README/README.md) for the conceptual tour.
