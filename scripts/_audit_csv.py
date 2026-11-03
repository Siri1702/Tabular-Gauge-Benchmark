"""Audit Stage-II CSV: row count, schema, per-(model,dataset,train_size,seed)
completeness, key summary stats. Output as JSON + Markdown for citation
in the paper's appendix."""
import json
import os
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\siri\Desktop\Tabfm-benchmark")
STAGE2 = ROOT / "experiments" / "results" / "few_shot" / "multiseed_few_shot_results.csv"
OUT_JSON = ROOT / "reports" / "stage2_csv_audit.json"
OUT_MD_PATH = ROOT / "reports" / "appendix_stage2_csv_metadata.md"

assert STAGE2.exists(), f"missing {STAGE2}"

df = pd.read_csv(STAGE2)
n_rows, n_cols = df.shape

# Schema
schema = {c: str(df[c].dtype) for c in df.columns}

# Completeness
unique_models    = sorted(df["model"].unique().tolist())
unique_datasets  = sorted(df["dataset"].unique().tolist())
unique_sizes     = sorted(df["train_size"].astype(str).unique().tolist(),
                         key=lambda x: (isinstance(x, str) and x == "full",
                                        int(x) if x.isdigit() else 0))
unique_seeds     = sorted(df["seed"].unique().tolist())

# Per-cell counts
cell_counts = (df.groupby(["model", "dataset", "train_size", "seed"])
               .size()
               .reset_index(name="n"))
multi = cell_counts[cell_counts["n"] > 1]
missing = cell_counts[cell_counts["n"] == 0]

# Sanity counts
expected = (len(unique_models) * len(unique_datasets)
            * len(unique_sizes) * len(unique_seeds))

# AUC stats per model
auc_per_model = (df.groupby("model")["roc_auc"]
                 .agg(["mean", "median", "min", "max", "std", "count"])
                 .reset_index())

# Per-N AUC for TabPFN + best GBDT
tabpfn = df[df["model"] == "TabPFN"]
tabpfn_auc_by_n = (tabpfn.groupby("train_size")["roc_auc"]
                   .agg(["mean", "std", "count"])
                   .reset_index())

# Per-dataset AUC for TabPFN
tabpfn_auc_by_ds = (tabpfn.groupby("dataset")["roc_auc"]
                    .agg(["mean", "std", "count"])
                    .reset_index())

# Memory and time check
mem_col = "peak_memory_mb" if "peak_memory_mb" in df.columns else None
time_col = "predict_time_sec" if "predict_time_sec" in df.columns else None

OUT_JSON.write_text(json.dumps({
    "path": str(STAGE2),
    "n_rows": int(n_rows),
    "n_cols": int(n_cols),
    "schema": schema,
    "unique_models": unique_models,
    "unique_datasets": unique_datasets,
    "unique_train_sizes": [str(x) for x in unique_sizes],
    "unique_seeds": unique_seeds,
    "expected_cells": expected,
    "duplicate_cells": int(len(multi)),
    "auc_per_model": auc_per_model.to_dict(orient="records"),
    "tabpfn_auc_by_train_size": tabpfn_auc_by_n.to_dict(orient="records"),
    "tabpfn_auc_by_dataset": tabpfn_auc_by_ds.to_dict(orient="records"),
    "memory_col_present": mem_col is not None,
    "time_col_present":   time_col is not None,
}, indent=2), encoding="utf-8")

with OUT_MD_PATH.open("w", encoding="utf-8") as fh:
    fh.write("# Appendix B — Stage-II CSV Metadata Audit\n\n")
    fh.write(f"- Path: `{STAGE2}`\n")
    fh.write(f"- Rows: **{n_rows}**\n")
    fh.write(f"- Cols: **{n_cols}**\n")
    fh.write(f"- Expected total cells: **{expected}** = "
             f"{len(unique_models)} models × {len(unique_datasets)} datasets × "
             f"{len(unique_sizes)} train sizes × {len(unique_seeds)} seeds\n")
    fh.write(f"- Duplicate cells: {len(multi)} (must be 0)\n\n")
    fh.write("## Columns\n\n")
    for c, t in schema.items():
        fh.write(f"- `{c}` (`{t}`)\n")
    fh.write("\n## AUC summary per model\n\n")
    fh.write("| Model | n | mean AUC | median | min | max | std |\n")
    fh.write("|---|---:|---:|---:|---:|---:|---:|\n")
    for _, r in auc_per_model.iterrows():
        fh.write(f"| {r['model']} | {int(r['count'])} | "
                 f"{r['mean']:.4f} | {r['median']:.4f} | "
                 f"{r['min']:.4f} | {r['max']:.4f} | {r['std']:.4f} |\n")
    fh.write("\n## TabPFN AUC by train size\n\n")
    fh.write("| Train size | n | mean | std |\n|---|---:|---:|---:|\n")
    for _, r in tabpfn_auc_by_n.iterrows():
        fh.write(f"| {r['train_size']} | {int(r['count'])} | "
                 f"{r['mean']:.4f} | {r['std']:.4f} |\n")
    fh.write("\n## TabPFN AUC by dataset (across all N + seeds)\n\n")
    fh.write("| Dataset | n | mean | std |\n|---|---:|---:|---:|\n")
    for _, r in tabpfn_auc_by_ds.iterrows():
        fh.write(f"| {r['dataset']} | {int(r['count'])} | "
                 f"{r['mean']:.4f} | {r['std']:.4f} |\n")
    fh.write(f"\n## Notes for the paper\n\n")
    fh.write(f"- Memory column present: **{mem_col is not None}** "
             f"(`{mem_col}`).\n")
    fh.write(f"- Inference time column present: **{time_col is not None}** "
             f"(`{time_col}`).\n")
    fh.write(f"- Total raw cells = {n_rows}.\n")
    fh.write(f"- Zero duplicates = {len(multi) == 0}.\n")

print(f"Wrote {OUT_JSON}")
print(f"Wrote {OUT_MD_PATH}")
print(f"Rows: {n_rows}  Cols: {n_cols}  Unique models: {unique_models}")
print(f"Train sizes: {unique_sizes}")
print(f"Seeds: {unique_seeds}")
print(f"Datasets: {unique_datasets}")
