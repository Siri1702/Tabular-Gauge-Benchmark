"""Inventory: paper1 drafts, result CSVs, figures."""
import os

ROOT = r"C:\Users\siri\Desktop\Tabfm-benchmark"

print("===== paper1 DRAFTS =====")
for root, dirs, files in os.walk(os.path.join(ROOT, "paper1")):
    for f in sorted(files):
        if f.endswith(".md"):
            p = os.path.join(root, f)
            size = os.path.getsize(p)
            rel = os.path.relpath(p, ROOT)
            with open(p, "r", encoding="utf-8") as fh:
                head = " ".join(fh.read(160).split())
            print(f"  {size:>7d}  {rel}")
            print(f"           {head[:140]}")

print("\n===== RESULT CSVs =====")
csv_dir = os.path.join(ROOT, "experiments", "results")
for d in ("few_shot", "aggregated", "act1", "act2", "act3"):
    p = os.path.join(csv_dir, d)
    if os.path.isdir(p):
        for f in sorted(os.listdir(p)):
            if f.endswith(".csv"):
                fp = os.path.join(p, f)
                rel = os.path.relpath(fp, ROOT)
                try:
                    with open(fp, "r", encoding="utf-8") as fh:
                        rows = sum(1 for _ in fh) - 1
                except Exception:
                    rows = -1
                size = os.path.getsize(fp)
                print(f"  {rows:>6d} rows  {size:>8d} B  {rel}")

print("\n===== FIGURES =====")
fig_root = os.path.join(ROOT, "reports", "figures")
fig_count = 0
for root, dirs, files in os.walk(fig_root):
    for f in sorted(files):
        if f.endswith(".png") or f.endswith(".pdf"):
            fp = os.path.join(root, f)
            size = os.path.getsize(fp)
            print(f"  {size:>8d} B  {os.path.relpath(fp, ROOT)}")
            fig_count += 1
print(f"Total figures: {fig_count}")
