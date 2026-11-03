"""
Detailed Few-Shot Learning Analysis Report Generator

Loads existing results and generates comprehensive analysis and visualizations.
"""

import sys
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_results():
    """Load the few-shot results."""
    df = pd.read_csv(project_root / "experiments" / "results" / "few_shot" / "quick_few_shot_results.csv")

    # Clean train_size - convert to numeric
    df['train_size_num'] = df['train_size'].apply(
        lambda x: int(x) if str(x).isdigit() else 9999  # 'full' -> large number
    )

    # Clean dataset names
    df['dataset_clean'] = df['dataset'].str.replace('_N.*', '', regex=True)

    return df


def analyze_by_training_size(df):
    """Analyze performance by training size."""
    print("\n" + "="*70)
    print("ANALYSIS 1: PERFORMANCE BY TRAINING SIZE")
    print("="*70)

    # Mean ROC-AUC by training size and model
    summary = df.groupby(["train_size_num", "model"])["roc_auc"].mean().unstack()

    # Sort by training size
    summary = summary.sort_index()

    print("\nMean ROC-AUC by Training Size:")
    print("-" * 60)
    print(summary.round(4).to_string())

    # Find best model at each training size
    print("\nBest Model at Each Training Size:")
    for size in summary.index:
        row = summary.loc[size]
        best_model = row.idxmax()
        best_auc = row.max()
        print(f"  N={size:>5}: {best_model:10s} with AUC={best_auc:.4f}")


def analyze_by_dataset(df):
    """Analyze performance by dataset."""
    print("\n" + "="*70)
    print("ANALYSIS 2: PERFORMANCE BY DATASET")
    print("="*70)

    # Clean dataset names
    df['dataset_clean'] = df['dataset'].str.replace('_N.*', '', regex=True)

    # Mean performance by dataset
    dataset_summary = df.groupby("dataset_clean")["roc_auc"].mean().sort_values(ascending=False)

    print("\nMean ROC-AUC by Dataset:")
    for ds, auc in dataset_summary.items():
        print(f"  {ds:20s}: {auc:.4f}")


def crossover_analysis(df):
    """Perform crossover analysis."""
    print("\n" + "="*70)
    print("ANALYSIS 3: CROSSOVER ANALYSIS (TabPFN vs XGBoost)")
    print("="*70)

    # Pivot by train_size and model
    pivot = df.pivot_table(
        index="train_size_num",
        columns="model",
        values="roc_auc",
        aggfunc="mean"
    ).sort_index()

    print("\nTabPFN vs XGBoost at each training size:")
    print("-" * 60)
    print(f"{'N':>6} | {'TabPFN':>8} | {'XGBoost':>8} | {'Winner':>10} | {'Diff':>8}")
    print("-" * 60)

    crossovers = []

    for size in pivot.index:
        tabpfn = pivot.loc[size, "TabPFN"]
        xgb = pivot.loc[size, "XGBoost"]
        diff = tabpfn - xgb
        winner = "TabPFN" if tabpfn > xgb else "XGBoost"

        print(f"{size:>6} | {tabpfn:>8.4f} | {xgb:>8.4f} | {winner:>10} | {diff:>+8.4f}")

        crossovers.append({
            'train_size': size,
            'TabPFN': tabpfn,
            'XGBoost': xgb,
            'winner': winner,
            'diff': diff
        })

    # Find crossover points
    print("\n" + "-" * 60)
    print("CROSSOVER POINTS:")
    prev_winner = None
    for c in crossovers:
        if prev_winner and c['winner'] != prev_winner:
            print(f"  CROSSOVER DETECTED at N={c['train_size']}: {prev_winner} -> {c['winner']}")
        prev_winner = c['winner']


def model_ranking_analysis(df):
    """Analyze model rankings across training sizes."""
    print("\n" + "="*70)
    print("ANALYSIS 4: MODEL RANKINGS ACROSS TRAINING SIZES")
    print("="*70)

    # Rank models at each training size
    pivot = df.pivot_table(
        index="train_size_num",
        columns="model",
        values="roc_auc",
        aggfunc="mean"
    ).sort_index()

    # Rank (1 = best)
    ranks = pivot.rank(axis=1, ascending=False)

    print("\nModel Rankings (1=best, 5=worst):")
    print(ranks.round(1).to_string())

    # Average rank
    print("\nAverage Rank:")
    avg_ranks = ranks.mean().sort_values()
    for model, rank in avg_ranks.items():
        print(f"  {model:10s}: {rank:.2f}")


def create_visualizations(df):
    """Create publication-ready visualizations."""
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {'TabPFN': '#FF6B6B', 'XGBoost': '#4ECDC4', 'LightGBM': '#45B7D1',
              'CatBoost': '#96CEB4', 'MLP': '#DDA0DD'}

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Learning Curves
    ax1 = axes[0, 0]
    pivot = df.pivot_table(index="train_size_num", columns="model", values="roc_auc", aggfunc="mean").sort_index()

    for model in pivot.columns:
        ax1.plot(pivot.index, pivot[model], marker='o', label=model,
                 color=colors.get(model, '#333'), linewidth=2, markersize=8)

    ax1.set_xlabel("Training Set Size", fontsize=12)
    ax1.set_ylabel("ROC-AUC", fontsize=12)
    ax1.set_title("Learning Curves: ROC-AUC vs Training Size", fontsize=14)
    ax1.set_xscale('log')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # 2. Model comparison bar chart by training size
    ax2 = axes[0, 1]
    pivot_plot = pivot.copy()
    pivot_plot.index = [str(x) for x in pivot_plot.index]
    pivot_plot.plot(kind='bar', ax=ax2, color=[colors.get(m, '#333') for m in pivot_plot.columns])
    ax2.set_xlabel("Training Set Size", fontsize=12)
    ax2.set_ylabel("ROC-AUC", fontsize=12)
    ax2.set_title("Model Comparison by Training Size", fontsize=14)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
    ax2.legend(loc='lower right')

    # 3. Heatmap
    ax3 = axes[1, 0]
    heatmap_data = pivot.T
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax3,
                vmin=0.5, vmax=1.0, cbar_kws={'label': 'ROC-AUC'})
    ax3.set_title("ROC-AUC Heatmap: Model x Training Size", fontsize=14)
    ax3.set_xlabel("Training Set Size")
    ax3.set_ylabel("Model")

    # 4. Win count
    ax4 = axes[1, 1]
    pivot_winners = pivot.idxmax(axis=1)
    win_counts = pivot_winners.value_counts()
    win_counts.plot(kind='bar', ax=ax4, color=[colors.get(m, '#333') for m in win_counts.index])
    ax4.set_xlabel("Model")
    ax4.set_ylabel("Number of Wins")
    ax4.set_title("Win Count by Model (Across Training Sizes)", fontsize=14)
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    # Save
    output_path = project_root / "reports" / "figures" / "few_shot_learning_curves.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")

    plt.close()


def generate_summary_statistics(df):
    """Generate summary statistics table."""
    print("\n" + "="*70)
    print("ANALYSIS 5: SUMMARY STATISTICS")
    print("="*70)

    # Overall model performance
    print("\nOverall Model Performance (Mean +/- Std):")
    overall = df.groupby("model")["roc_auc"].agg(['mean', 'std', 'min', 'max'])
    print(overall.round(4).to_string())

    # By training size regime
    print("\nPerformance by Training Size Regime:")
    df['regime'] = pd.cut(df['train_size_num'],
                          bins=[0, 50, 100, 200, 500, 1000, 10000],
                          labels=['25-50', '50-100', '100-200', '200-500', '500-1000', 'full'])

    regime_summary = df.groupby(["regime", "model"])["roc_auc"].mean().unstack()
    print(regime_summary.round(4).to_string())


def main():
    print("="*70)
    print("FEW-SHOT LEARNING ANALYSIS - DETAILED REPORT")
    print("="*70)

    # Load results
    df = load_results()
    print(f"\nLoaded {len(df)} results")
    print(f"Datasets: {df['dataset_clean'].nunique()}")
    print(f"Training sizes: {sorted(df['train_size_num'].unique())}")
    print(f"Models: {df['model'].unique().tolist()}")

    # Run analyses
    analyze_by_training_size(df)
    analyze_by_dataset(df)
    crossover_analysis(df)
    model_ranking_analysis(df)
    generate_summary_statistics(df)

    # Create visualizations
    create_visualizations(df)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nVisualization saved to: reports/figures/few_shot_learning_curves.png")


if __name__ == "__main__":
    main()