import os
os.environ.setdefault('TABPFN_ALLOW_CPU_LARGE_DATASET', '1')
import sys; sys.path.insert(0, '.')
from tabular_gauge.models.tabpfn_model import TabPFNWrapper
from tabular_gauge.evaluation.metrics import compute_metrics
from tabular_gauge.data.loader import DataLoader
import pandas as pd
print('imports OK')
DATASETS = [
    ('australian',    40536),
    ('banknote-auth',  1462),
    ('kc1',            1067),
    ('ozone-level',    1487),
    ('phoneme',        1489),
    ('sick',           1050),
    ('spambase',         44),
    ('titanic',       40945),
]
for name, did in DATASETS:
    dl = DataLoader(test_size=0.2, random_state=42)
    ds = dl.load_openml(did, name)
    print(f'  {name:18s}: rows={ds.n_rows:5d}  X_train={ds.X_train.shape}  pos-rate={float(ds.y_train.mean()):.3f}')
csv = 'experiments/results/few_shot/multiseed_few_shot_results.csv'
print('CSV exists?', os.path.exists(csv))
df = pd.read_csv(csv)
print('cols:', sorted([c for c in df.columns if c in {
    'roc_auc','peak_memory_mb','fit_time_sec','context_used','was_subsampled',
    'context_original','actual_n_train','model','train_size','dataset','seed'}]))
tab_full = df[(df['model']=='TabPFN') & (df['train_size']=='full')]
print('Stage-II TabPFN / full rows:', len(tab_full))
print('Actual n_train per dataset in Stage-II:')
print(tab_full.groupby('dataset')['actual_n_train'].first().sort_values().to_string())
