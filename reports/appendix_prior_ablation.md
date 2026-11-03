# Appendix C.6 — TabPFN-v2 In-Context-Learning Ablation

**Created:** 2026-08-19 — to address the question: *is TabPFN-v2's
low-N lead driven by in-context learning over the (X, y) context, or
by the model architecture / pre-training priors in isolation?*

We fit TabPFN-v2 twice on the **same** (X_train, dataset, seed, N):
once with the **real** training labels, once with the training
labels **permuted** (deterministically, keeping X intact). The
permutation is the standard in-context-learning ablation (Min et
al. 2022, also used in the TabPFN-v2 paper §3): if TabPFN-v2's
lead at low N comes from in-context learning over the labelled
context, the shuffled-label fit must collapse to near-chance AUC
because the labels carry no information.

If, instead, shuffled-labels AUC stays close to clean AUC, the
lead is **not** explained by in-context learning — it is either
an architectural artefact or comes from prior-data exposure
without using the context. Both would be informative for the
narrative in §1.

## C.6.1 Per-N result

| N | n pairs | mean AUC (clean) | mean AUC (shuffled) | Δ AUC | Wilcoxon p (clean > shuffled) |
|---|---:|---:|---:|---:|---:|
| 25 | 9 | 0.9028 | 0.5222 | +0.3806 | 0.001953 |
| 50 | 9 | 0.9091 | 0.5813 | +0.3278 | 0.001953 |
| 100 | 9 | 0.9144 | 0.5266 | +0.3878 | 0.001953 |
| 500 | 9 | 0.9309 | 0.5061 | +0.4248 | 0.001953 |
| full | 9 | 0.9524 | 0.4649 | +0.4874 | 0.001953 |

## C.6.2 Per-dataset result

| Dataset | mean clean | mean shuffled | Δ |
|---|---:|---:|---:|
| banknote-auth | 0.9993 | 0.5554 | +0.4440 |
| kc1 | 0.7973 | 0.4658 | +0.3315 |
| spambase | 0.9691 | 0.5395 | +0.4296 |

**Overall paired Wilcoxon (clean > shuffled): p = 2.842e-14, across 45 (dataset, seed, N) pairs.**

## C.6.3 Interpretation

* **If shuffled AUC collapses to ~0.5 across the board:** the
  in-context-learning mechanism is doing the work; pretrained
  priors alone (architecture + general training) are not enough.
  This is the standard 'ICL is the source of the advantage'
  finding for in-context learners.
* **If shuffled AUC collapses only at N ≤ 500:** the in-context
  mechanism matters most when the labelled context is small;
  pretrained priors carry the rest. This is consistent with a
  pretrained prior that 'kicks in' once the in-context signal
  gets weak.
* **If shuffled AUC stays close to clean AUC:** the in-context
  mechanism is *not* the source of the lead — the lead is
  either an architectural artefact or a prior-data exposure
  effect that doesn't depend on the actual context labels. We
  would then need to weaken §1's priors narrative.
* **Sanity check:** if clean AUC is also ~0.5, the model is
  broken on our setup and the ablation is uninformative.

## C.6.4 Sources

* Ablation CSV: `experiments/results/few_shot/tabpfn_shuffled_labels.csv`
* Summary JSON: `reports/tabpfn_shuffled_labels.json`
* Reproduce: `python scripts/tabpfn_shuffled_labels_ablation.py`
