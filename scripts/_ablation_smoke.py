"""Quick smoke test of the ablation script's fit_eval — one cell."""
import sys
sys.path.insert(0, r"C:\Users\siri\Desktop\Tabfm-benchmark")
sys.path.insert(0, r"C:\Users\siri\Desktop\Tabfm-benchmark\scripts")
from tabpfn_shuffled_labels_ablation import fit_eval
import time
t0 = time.perf_counter()
for mode in (False, True):
    row = fit_eval("banknote-auth", 1462, 0, 50, mode)
    print(f"  mode={'shuffled' if mode else 'clean'}  auc={row['auc']:.4f}  ece={row['ece_10']:.4f}  "
          f"fit={row['fit_time_sec']:.2f}s  pred={row['predict_time_sec']:.2f}s  wall={time.perf_counter()-t0:.1f}s")
    t0 = time.perf_counter()
print("SMOKE OK")
