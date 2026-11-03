"""Compute the SAP-RPT-1 mean for each metric across 75 Stage-I cells."""
import pandas as pd

df = pd.read_csv("experiments/results/aggregated/results.csv")
s = df[df["model"] == "SAP-RPT-1"].copy()
print(f"SAP-RPT-1 cells: {len(s)}")
print(f"  ({s['dataset'].nunique()} datasets × {s['seed'].nunique()} seeds)")
print()

# Per-metric summary
metrics = [
    "roc_auc",
    "avg_precision",
    "pr_auc",
    "brier_score",
    "log_loss_val",
    "balanced_accuracy",
    "peak_memory_mb",
    "ms_per_test_row",
    "ece_10",
]
for m in metrics:
    if m in s.columns:
        v = s[m]
        ok = v.dropna()
        n_ok = len(ok)
        if n_ok:
            print(f"  {m:<22}  mean={ok.mean():.4f}   std={ok.std():.4f}   "
                  f"median={ok.median():.4f}   n={n_ok}/{len(v)}")
        else:
            print(f"  {m:<22}  no valid values")

print()
print("Per-dataset SAP-RPT-1 mean ROC-AUC:")
g = s.groupby("dataset")["roc_auc"].agg(["mean", "std", "count"])
print(g.to_string())

# Also confirm: what does TabPFN look like at the Stage-I level?
print()
print("TabPFN Stage-I cells (for reference):")
t = df[df["model"] == "TabPFN"]
print(f"  n = {len(t)}")
print(f"  mean ROC-AUC across all Stage-I cells: {t['roc_auc'].mean():.4f}")
