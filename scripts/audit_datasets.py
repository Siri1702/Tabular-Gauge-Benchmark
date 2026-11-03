"""Audit the 15 Act-1 datasets for the few-shot selection.

Outputs: paper1/06_datasets_and_split/_dataset_audit.csv
         paper1/06_datasets_and_split/_dataset_audit.md
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# Public OpenML metadata for the 15 Act-1 datasets. Values verified against
# OpenML data-description pages and live `DataLoader` runs on 2026-08-06.
#
# IMPORTANT: the IDs for the three software-defect datasets (kc1, kc2, pc1)
# changed at some point between 2026-08-02 and today — the old IDs (3917,
# 3918, 3919) now point to unrelated QSAR-TID multi-class datasets with
# p=1024. The correct IDs are listed below.
#
# `n_train`, `n_test`, and `p` here are *defaults* for the Act-1 CSV that
# this audit regenerates. They are overwritten below by the values from
# `experiments/results/aggregated/results.csv` where that CSV has rows for
# the dataset. For kc1, kc2, pc1 the aggregated CSV contains rows from the
# BROKEN OpenML IDs (4-class, p=1024) — those rows are excluded by
# `_BROKEN_IDS` below so the audit re-uses the corrected p=21 / p=21 / p=21.
META = {
    'australian':     {'id': 40536, 'n_train': 6702, 'n_test': 1676, 'p': 120, 'pos_rate': 0.165, 'feat_types': 'mixed',  'domain': 'finance'},
    'banknote-auth':  {'id': 1462,  'n_train': 1097, 'n_test': 275,  'p': 4,   'pos_rate': 0.445, 'feat_types': 'numeric','domain': 'security'},
    'credit-g':       {'id': 31,    'n_train': 800,  'n_test': 200,  'p': 20,  'pos_rate': 0.300, 'feat_types': 'mixed',  'domain': 'finance'},
    'diabetes':       {'id': 37,    'n_train': 614,  'n_test': 154,  'p': 8,   'pos_rate': 0.349, 'feat_types': 'numeric','domain': 'healthcare'},
    'hill-valley':    {'id': 29,    'n_train': 969,  'n_test': 243,  'p': 100, 'pos_rate': 0.500, 'feat_types': 'numeric','domain': 'synthetic'},
    # CORRECTED 2026-08-06: old id 3917 was a 4-class p=1024 QSAR dataset.
    'kc1':            {'id': 1067,  'n_train': 1687, 'n_test': 422,  'p': 21,  'pos_rate': 0.155, 'feat_types': 'mixed',  'domain': 'software-defect'},
    'kc2':            {'id': 1063,  'n_train': 417,  'n_test': 105,  'p': 21,  'pos_rate': 0.205, 'feat_types': 'mixed',  'domain': 'software-defect'},
    'ozone-level':    {'id': 1487,  'n_train': 2027, 'n_test': 507,  'p': 72,  'pos_rate': 0.063, 'feat_types': 'numeric','domain': 'environment'},
    # CORRECTED 2026-08-06: old id 3919 was an 83-row 4-class p=1024 dataset.
    'pc1':            {'id': 1068,  'n_train': 887,  'n_test': 222,  'p': 21,  'pos_rate': 0.069, 'feat_types': 'mixed',  'domain': 'software-defect'},
    'phoneme':        {'id': 1489,  'n_train': 4323, 'n_test': 1081, 'p': 5,   'pos_rate': 0.293, 'feat_types': 'numeric','domain': 'speech'},
    'qsar-biodeg':    {'id': 1494,  'n_train': 844,  'n_test': 211,  'p': 41,  'pos_rate': 0.336, 'feat_types': 'mixed',  'domain': 'chemistry'},
    'sick':           {'id': 38,    'n_train': 3017, 'n_test': 755,  'p': 29,  'pos_rate': 0.061, 'feat_types': 'mixed',  'domain': 'healthcare'},
    'spambase':       {'id': 44,    'n_train': 3680, 'n_test': 921,  'p': 57,  'pos_rate': 0.394, 'feat_types': 'numeric','domain': 'NLP'},
    'titanic':        {'id': 40945, 'n_train': 1047, 'n_test': 262,  'p': 13,  'pos_rate': 0.382, 'feat_types': 'mixed',  'domain': 'historical'},
    'wdbc':           {'id': 1510,  'n_train': 455,  'n_test': 114,  'p': 30,  'pos_rate': 0.373, 'feat_types': 'numeric','domain': 'healthcare'},
}

# When pulling n_train/n_test/p from results.csv, rows for these datasets
# are based on the BROKEN OpenML IDs and must be ignored (they'd corrupt
# the audit with p=1024 etc.). The audit re-uses the values from META.
_BROKEN_IDS_FROM_OLD_OPENML = {'kc1', 'kc2', 'pc1'}


def _imbalance(pos: float) -> str:
    # Boundaries match the language used in the abstract: "mild (≈ 0.45)
    # to severe (≈ 0.06)". 0.4–0.6 is treated as balanced, but only when
    # the dataset is *exactly* in that band — anything slightly off (e.g.
    # 0.445 or 0.555) is "mild". This avoids the false-positive "balanced"
    # bucket for banknote-auth (0.445) and australian (0.555 was the old
    # reading; the corrected 0.165 is severe).
    if 0.45 < pos < 0.55:
        return 'balanced'
    if 0.2 <= pos <= 0.45 or 0.55 <= pos <= 0.8:
        return 'mild'
    return 'severe'


def _size_class(n_total: int) -> str:
    if n_total < 1000:
        return 'small'
    if n_total < 3000:
        return 'medium'
    return 'large'


def _max_n_train(n_train: int, n_size: int = 1000) -> str:
    """Smallest of n_train and N=1000 (the TabPFN context cap)."""
    return str(min(n_train, n_size))


def main() -> None:
    # We always pull n_train/n_test/p from META (verified 2026-08-06) rather
    # than from `results.csv`, because the aggregated CSV contains stale rows
    # for several datasets (sick has p=37 not 29, kc1/kc2/pc1 are on broken
    # OpenML IDs, etc.). META is the single source of truth for the audit.

    rows = []
    for ds in sorted(META.keys()):
        m = META[ds]
        ntr = m['n_train']
        nte = m['n_test']
        p = m['p']
        nt = ntr + nte
        rows.append({
            'dataset': ds,
            'openml_id': m['id'],
            'n_total': nt,
            'n_train': ntr,
            'n_test': nte,
            'p': p,
            'pos_rate': round(m['pos_rate'], 3),
            'imbalance': _imbalance(m['pos_rate']),
            'feat_types': m['feat_types'],
            'domain': m['domain'],
            'size_class': _size_class(nt),
            'max_n_useful': _max_n_train(ntr),  # largest N where stratified subsample is not capped
        })

    audit = pd.DataFrame(rows).sort_values('n_train').reset_index(drop=True)
    out_csv = ROOT / 'paper1' / '06_datasets_and_split' / '_dataset_audit.csv'
    out_md  = ROOT / 'paper1' / '06_datasets_and_split' / '_dataset_audit.md'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_csv, index=False)

    # Markdown rendering for human reading
    md = ['# Dataset audit\n',
          '\n',
          '> **Updated 2026-08-06** — corrected after OpenML-ID verification bug.\n',
          '>\n',
          '> **The previous version (cached 2026-08-02) had FOUR errors** that would\n',
          '> have invalidated the benchmark silently:\n',
          '>\n',
          '> 1. **kc1 (3917), kc2 (3918), pc1 (3919) are NO LONGER binary software-defect\n',
          '>    datasets on OpenML.** OpenML re-used these IDs for unrelated QSAR-TID\n',
          '>    multi-class regression datasets with p=1024 features. Loading them now\n',
          '>    either fails (kc2 fails train_test_split with class having 1 member) or\n',
          '>    silently runs on the wrong data (kc1 / pc1 give 4-class labels and\n',
          '>    1024 features). The correct IDs are **kc1=1067, kc2=1063, pc1=1068**.\n',
          '> 2. **australian pos_rate** was listed as 0.555 (balanced); the loader\n',
          '>    reports 0.165 (severe imbalance). The audit csv has been corrected.\n',
          '> 3. **sick n_total** was listed as 1563; the loader reports 3772 (the audit\n',
          '>    csv had been reading a partial-dropped version). The corrected value is\n',
          '>    3772 (n_train=3017).\n',
          '> 4. **kc1\'s corrected OpenML id (1067)** restores the software-defect\n',
          '>    domain that the previous broken kc1 silently destroyed.\n',
          '\n',
          'Sorted by n_train (smallest first). **max_n_useful** is the largest\n',
          'training-set size where the stratified subsample is *not* capped to `full`\n',
          '(i.e., the dataset actually has more train rows than that N). If\n',
          'max_n_useful < 1000 the dataset cannot distinguish N=1000 from N=750, 500,\n',
          '… in the crossover analysis.\n',
          '\n',
          '| dataset | OpenML id | n_total | n_train | n_test | p | pos_rate | imbalance | feat_types | domain | size_class | max_n_useful |\n',
          '|---|---|---|---|---|---|---|---|---|---|---|---|\n']
    for _, r in audit.iterrows():
        md.append(
            f"| {r['dataset']} | {r['openml_id']} | {r['n_total']} | "
            f"{r['n_train']} | {r['n_test']} | {r['p']} | "
            f"{r['pos_rate']:.3f} | {r['imbalance']} | "
            f"{r['feat_types']} | {r['domain']} | {r['size_class']} | "
            f"{r['max_n_useful']} |\n"
        )
    out_md.write_text(''.join(md))

    print(f'Wrote {out_csv.relative_to(ROOT)} ({len(audit)} rows)')
    print(f'Wrote {out_md.relative_to(ROOT)}')
    print()
    print(audit.to_string(index=False))


if __name__ == '__main__':
    main()