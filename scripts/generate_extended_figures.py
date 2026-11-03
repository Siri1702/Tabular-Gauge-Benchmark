"""
Extended Few-Shot Analysis - Publication-Ready Visualizations
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Publication-ready style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Load data
df = pd.read_csv('experiments/results/few_shot/extended_few_shot_results.csv')
df['dataset_clean'] = df['dataset'].str.extract(r'^([^_]+)')[0]
df['n_train_actual'] = pd.to_numeric(df['n_train'], errors='coerce')

# Standard sizes
std_sizes = [25, 50, 100, 200, 300, 500, 750, 1000]
df_std = df[df['n_train_actual'].isin(std_sizes)].copy()

# Color palette (from validated reference)
colors = {
    'TabPFN': '#2a78d6',
    'XGBoost': '#e34948',
    'LightGBM': '#eda100',
    'CatBoost': '#4a3aa7',
    'MLP': '#1baf7a'
}
markers = {'TabPFN': 'o', 'XGBoost': 's', 'LightGBM': '^', 'CatBoost': 'D', 'MLP': 'v'}

x_positions = list(range(len(std_sizes)))

# ============ PLOT 1: LEARNING CURVES ============
fig, ax = plt.subplots(figsize=(11, 6.5))

perf_data = df_std.groupby(['n_train_actual', 'model'])['roc_auc'].agg(['mean', 'std']).reset_index()

for model in ['TabPFN', 'XGBoost', 'LightGBM', 'CatBoost', 'MLP']:
    mdata = perf_data[perf_data['model'] == model].sort_values('n_train_actual')
    x_vals = [std_sizes.index(n) for n in mdata['n_train_actual']]
    y_vals = mdata['mean'].values
    yerr = mdata['std'].values

    ax.errorbar(x_vals, y_vals, yerr=yerr,
                label=model, color=colors[model],
                marker=markers[model], markersize=8,
                linewidth=2.2, capsize=3, capthick=1.5, markeredgecolor='white', markeredgewidth=0.5)

ax.set_xticks(x_positions)
ax.set_xticklabels(std_sizes)
ax.set_xlabel('Training Set Size (N)', fontweight='bold')
ax.set_ylabel('ROC-AUC', fontweight='bold')
ax.set_title('Learning Curves: Model Performance vs Training Size\n(Mean +/- SD across 6 datasets)', fontweight='bold', pad=12)

ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=False, fontsize=9)
ax.set_ylim(0.55, 1.02)
ax.set_xlim(-0.3, len(std_sizes)-0.7)
ax.grid(True, alpha=0.25, linestyle='-', linewidth=0.5)
ax.set_facecolor('#fafafa')

ax.axvspan(-0.5, 1.5, alpha=0.08, color='#2a78d6')
ax.axvspan(4.5, 5.5, alpha=0.08, color='#e34948')
ax.text(0.5, 0.57, 'Extreme Few-Shot', fontsize=9, alpha=0.7, ha='center', color='#2a78d6')
ax.text(5, 0.57, 'Convergence Zone', fontsize=9, alpha=0.7, ha='center', color='#e34948')

plt.tight_layout()
plt.savefig('reports/figures/few_shot_learning_curves_extended.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('reports/figures/few_shot_learning_curves_extended.pdf', bbox_inches='tight', facecolor='white')
plt.close()
print('Saved: few_shot_learning_curves_extended.png/pdf')

# ============ PLOT 2: CROSSOVER ANALYSIS ============
fig, ax = plt.subplots(figsize=(10, 5.5))

crossover = df_std.groupby(['n_train_actual', 'model'])['roc_auc'].mean().unstack()[['TabPFN', 'XGBoost']]
crossover = crossover.loc[std_sizes]

ax.plot(x_positions, crossover['TabPFN'].values, 'o-',
        label='TabPFN', color=colors['TabPFN'], linewidth=2.5, markersize=10, markeredgecolor='white', markeredgewidth=1)
ax.plot(x_positions, crossover['XGBoost'].values, 's-',
        label='XGBoost', color=colors['XGBoost'], linewidth=2.5, markersize=10, markeredgecolor='white', markeredgewidth=1)

for i in range(len(std_sizes)):
    tp = crossover['TabPFN'].iloc[i]
    xgb = crossover['XGBoost'].iloc[i]
    if tp > xgb:
        ax.fill_between([i-0.25, i+0.25], [tp]*2, [xgb]*2,
                       color=colors['TabPFN'], alpha=0.15)

ax.set_xticks(x_positions)
ax.set_xticklabels(std_sizes)
ax.set_xlabel('Training Set Size (N)', fontweight='bold')
ax.set_ylabel('ROC-AUC', fontweight='bold')
ax.set_title('TabPFN vs XGBoost: Crossover Analysis (Shaded = TabPFN advantage)', fontweight='bold', pad=12)
ax.legend(loc='lower right', frameon=True, fontsize=10)
ax.set_ylim(0.68, 1.0)
ax.grid(True, alpha=0.25)
ax.set_facecolor('#fafafa')

for i, n in enumerate(std_sizes):
    diff = crossover['TabPFN'].iloc[i] - crossover['XGBoost'].iloc[i]
    if abs(diff) > 0.02:
        ax.annotate(f'{diff:+.1%}', xy=(i, max(crossover['TabPFN'].iloc[i], crossover['XGBoost'].iloc[i]) + 0.02),
                   ha='center', fontsize=8, color='#333')

plt.tight_layout()
plt.savefig('reports/figures/few_shot_crossover_extended.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved: few_shot_crossover_extended.png')

# ============ PLOT 3: PERFORMANCE HEATMAP ============
fig, ax = plt.subplots(figsize=(11, 5))

pivot = df_std.pivot_table(values='roc_auc', index='model', columns='n_train_actual', aggfunc='mean')
pivot = pivot[std_sizes]
pivot = pivot.loc[['TabPFN', 'XGBoost', 'LightGBM', 'CatBoost', 'MLP']]

im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=0.6, vmax=1.0)

for i in range(len(pivot)):
    for j in range(len(std_sizes)):
        val = pivot.iloc[i, j]
        text_color = 'white' if val > 0.88 or val < 0.7 else 'black'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=9, color=text_color, fontweight='bold')

ax.set_xticks(range(len(std_sizes)))
ax.set_xticklabels(std_sizes)
ax.set_yticks(range(len(pivot)))
ax.set_yticklabels(pivot.index)
ax.set_xlabel('Training Set Size (N)', fontweight='bold')
ax.set_ylabel('Model', fontweight='bold')
ax.set_title('Model Performance Heatmap: Mean AUC by Training Size', fontweight='bold', pad=12)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('ROC-AUC', fontsize=10)

plt.tight_layout()
plt.savefig('reports/figures/few_shot_heatmap_extended.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved: few_shot_heatmap_extended.png')

# ============ PLOT 4: TIMING COMPARISON ============
fig, ax = plt.subplots(figsize=(10, 5.5))

timing = df_std.groupby(['n_train_actual', 'model'])['total_time_sec'].mean().unstack()
timing = timing.loc[std_sizes]

for model in ['TabPFN', 'XGBoost', 'LightGBM', 'CatBoost', 'MLP']:
    ax.plot(x_positions, timing[model].values, 'o-',
            label=model, color=colors[model], linewidth=2, markersize=6)

ax.set_xticks(x_positions)
ax.set_xticklabels(std_sizes)
ax.set_xlabel('Training Set Size (N)', fontweight='bold')
ax.set_ylabel('Total Time (seconds, log scale)', fontweight='bold')
ax.set_title('Training Time Comparison: TabPFN vs Traditional ML Models', fontweight='bold', pad=12)
ax.legend(loc='upper left', frameon=True, fontsize=9)
ax.set_yscale('log')
ax.grid(True, alpha=0.25)
ax.set_facecolor('#fafafa')

plt.tight_layout()
plt.savefig('reports/figures/few_shot_timing_extended.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('Saved: few_shot_timing_extended.png')

print('\nAll publication-ready figures generated!')
