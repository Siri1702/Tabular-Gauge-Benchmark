"""
Interactive Dashboard for Tabular-Gauge Benchmark Results.

Usage:
    python tabular_gauge/viz/dashboard.py
    # Then open http://127.0.0.1:8050 in your browser

Requires: pip install dash plotly
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import argparse

import pandas as pd
import numpy as np
from dash import Dash, html, dcc, callback, Output, Input, State
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from tabular_gauge.evaluation.statistical import (
    friedman_test,
    nemenyi_cd_test,
    compute_bootstrap_ci,
    wilcoxon_pairwise,
    average_rank_table,
)
from tabular_gauge.viz.leaderboard import MODEL_COLORS


DARK_BG = "#0d0f14"
SURFACE = "#13161e"
ACCENT = "#5b8dee"
ACCENT2 = "#a78bfa"
ACCENT3 = "#34d399"
TEXT_COLOR = "#e2e8f0"
MUTED = "#8892a4"


def load_latest_results(results_dir: str = "experiments/results/aggregated") -> pd.DataFrame:
    """Load the most recent results CSV."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return pd.DataFrame()

    csv_files = list(results_path.glob("results_*.csv"))
    if not csv_files:
        return pd.DataFrame()

    latest = max(csv_files, key=lambda p: p.stat().st_mtime)
    return pd.read_csv(latest)


def create_performance_tab(results_df: pd.DataFrame) -> html.Div:
    """Create the performance analysis tab."""
    if results_df.empty:
        return html.Div("No results available. Run the benchmark first.")

    metrics = ["roc_auc", "avg_precision", "f1_macro", "brier_score", "log_loss_val"]
    available_metrics = [m for m in metrics if m in results_df.columns]

    return html.Div([
        html.H3("Model Performance Comparison", style={"color": TEXT_COLOR}),

        html.Div([
            html.Label("Select Metric:", style={"color": MUTED}),
            dcc.Dropdown(
                id="metric-selector",
                options=[{"label": m.replace("_", " ").title(), "value": m} for m in available_metrics],
                value="roc_auc",
                clearable=False,
                style={"color": DARK_BG},
            ),
        ], style={"width": "300px", "margin-bottom": "20px"}),

        html.Div([
            dcc.Graph(id="auc-heatmap"),
        ]),

        html.Div([
            dcc.Graph(id="rank-chart"),
        ], style={"margin-top": "20px"}),

        html.Div([
            dcc.Graph(id="confidence-intervals"),
        ], style={"margin-top": "20px"}),
    ])


def create_timing_tab(results_df: pd.DataFrame) -> html.Div:
    """Create the timing analysis tab."""
    if results_df.empty:
        return html.Div("No results available.")

    return html.Div([
        html.H3("Timing & Efficiency Analysis", style={"color": TEXT_COLOR}),

        html.Div([
            dcc.Graph(id="timing-chart"),
        ]),

        html.Div([
            dcc.Graph(id="throughput-chart"),
        ], style={"margin-top": "20px"}),

        html.Div([
            dcc.Graph(id="memory-chart"),
        ], style={"margin-top": "20px"}),
    ])


def create_statistical_tab(results_df: pd.DataFrame) -> html.Div:
    """Create the statistical significance tab."""
    if results_df.empty:
        return html.Div("No results available.")

    # Compute statistical tests
    friedman = friedman_test(results_df, metric="roc_auc")
    nemenyi = nemenyi_cd_test(results_df, metric="roc_auc")
    wilcoxon = wilcoxon_pairwise(results_df, metric="roc_auc")
    rank_df = average_rank_table(results_df, metric="roc_auc")

    return html.Div([
        html.H3("Statistical Significance Analysis", style={"color": TEXT_COLOR}),

        # Friedman test results
        html.Div([
            html.H4("Friedman Test", style={"color": ACCENT}),
            html.Div(id="friedman-results"),
        ], style={"margin-bottom": "20px"}),

        # CD diagram
        html.Div([
            dcc.Graph(id="cd-diagram"),
        ], style={"margin-top": "20px"}),

        # Pairwise comparisons
        html.Div([
            html.H4("Pairwise Wilcoxon Tests", style={"color": ACCENT}),
            html.Div(id="wilcoxon-results"),
        ], style={"margin-top": "20px"}),

        # Win/Loss matrix
        html.Div([
            dcc.Graph(id="win-loss-chart"),
        ], style={"margin-top": "20px"}),
    ])


def create_diversity_tab(results_df: pd.DataFrame) -> html.Div:
    """Create the dataset diversity analysis tab."""
    if results_df.empty:
        return html.Div("No results available.")

    return html.Div([
        html.H3("Dataset-Level Analysis", style={"color": TEXT_COLOR}),

        html.Div([
            dcc.Graph(id="dataset-performance"),
        ]),

        html.Div([
            dcc.Graph(id="dataset-size-impact"),
        ], style={"margin-top": "20px"}),

        html.Div([
            dcc.Graph(id="feature-count-impact"),
        ], style={"margin-top": "20px"}),
    ])


def create_app() -> Dash:
    """Create the Dash application."""
    app = Dash(__name__)

    app.layout = html.Div([
        # Header
        html.Div([
            html.H1("Tabular-Gauge Benchmark Dashboard", style={"color": TEXT_COLOR}),
            html.P("Tabular Foundation Models Comparison", style={"color": MUTED}),
        ], style={
            "background": SURFACE,
            "padding": "20px",
            "border-bottom": f"2px solid {ACCENT}",
            "margin-bottom": "20px",
        }),

        # Controls
        html.Div([
            html.Button("Reload Results", id="reload-btn", n_clicks=0),
            html.Div(id="last-updated", style={"color": MUTED, "display": "inline-block", "margin-left": "20px"}),
        ], style={"margin-bottom": "20px"}),

        # Store for results
        dcc.Store(id="results-store", data=None),

        # Tabs
        dcc.Tabs([
            dcc.Tab(label="Performance", children=[create_performance_tab(pd.DataFrame())]),
            dcc.Tab(label="Timing & Memory", children=[create_timing_tab(pd.DataFrame())]),
            dcc.Tab(label="Statistical Tests", children=[create_statistical_tab(pd.DataFrame())]),
            dcc.Tab(label="Dataset Analysis", children=[create_diversity_tab(pd.DataFrame())]),
        ]),
    ], style={"background": DARK_BG, "min-height": "100vh", "padding": "20px"})

    return app


def update_performance_graphs(results_df: pd.DataFrame, metric: str) -> tuple:
    """Update performance graphs based on selected metric."""
    if results_df.empty:
        return go.Figure(), go.Figure(), go.Figure()

    # Heatmap
    pivot = results_df.groupby(["dataset", "model"])[metric].mean().unstack()
    pivot_norm = pivot.sub(pivot.min(axis=1), axis=0).div(pivot.max(axis=1) - pivot.min(axis=1) + 1e-9, axis=0)

    fig_heatmap = go.Figure(data=go.Heatmap(
        z=pivot_norm.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale="Viridis",
        text=np.round(pivot.values, 3),
        texttemplate="%{text}",
    ))
    fig_heatmap.update_layout(
        title=f"{metric.replace('_', ' ').title()} by Model × Dataset",
        xaxis_title="Model",
        yaxis_title="Dataset",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT_COLOR),
    )

    # Rank chart
    rank_df = average_rank_table(results_df, metric=metric)
    fig_ranks = go.Figure(data=go.Bar(
        x=rank_df["average_rank"],
        y=rank_df["model"],
        orientation="h",
        marker_color=[MODEL_COLORS.get(m, MUTED) for m in rank_df["model"]],
    ))
    fig_ranks.update_layout(
        title="Average Rank (lower = better)",
        xaxis_title="Average Rank",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT_COLOR),
    )

    # Confidence intervals
    ci_df = compute_bootstrap_ci(results_df, metric=metric)
    fig_ci = go.Figure()
    for _, row in ci_df.iterrows():
        fig_ci.add_trace(go.Scatter(
            x=[row["mean"]], y=[row["model"]],
            mode="markers",
            marker=dict(size=15, color=MODEL_COLORS.get(row["model"], MUTED)),
            error_x=dict(type="data", array=[row["mean"] - row["ci_lower"]], visible=True),
        ))
    fig_ci.update_layout(
        title=f"{metric.replace('_', ' ').title()} with 95% CI",
        xaxis_title=metric,
        paper_bgcolor=DARK_BG,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT_COLOR),
    )

    return fig_heatmap, fig_ranks, fig_ci


def update_timing_graphs(results_df: pd.DataFrame) -> tuple:
    """Update timing graphs."""
    if results_df.empty:
        return go.Figure(), go.Figure(), go.Figure()

    # Timing chart
    timing = results_df.groupby("model")[["tuning_time_sec", "fit_time_sec", "predict_time_sec"]].mean()
    fig_timing = go.Figure(data=[
        go.Bar(name=name, x=timing.index, y=timing[name])
        for name in timing.columns
    ])
    fig_timing.update_layout(
        title="Mean Timing (seconds)",
        barmode="stack",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT_COLOR),
    )

    # Throughput
    throughput = results_df.groupby("model")["throughput_rows_per_sec"].mean()
    fig_throughput = go.Figure(data=go.Bar(
        x=throughput.index,
        y=throughput.values,
        marker_color=[MODEL_COLORS.get(m, MUTED) for m in throughput.index],
    ))
    fig_throughput.update_layout(
        title="Inference Throughput (rows/sec)",
        yaxis_title="Rows/sec",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=SURFACE,
        font=dict(color=TEXT_COLOR),
    )

    # Memory
    if "peak_memory_mb" in results_df.columns:
        memory = results_df.groupby("model")["peak_memory_mb"].mean()
        fig_memory = go.Figure(data=go.Bar(
            x=memory.index,
            y=memory.values,
            marker_color=[MODEL_COLORS.get(m, MUTED) for m in memory.index],
        ))
        fig_memory.update_layout(
            title="Peak Memory Usage (MB)",
            yaxis_title="MB",
            paper_bgcolor=DARK_BG,
            plot_bgcolor=SURFACE,
            font=dict(color=TEXT_COLOR),
        )
    else:
        fig_memory = go.Figure()

    return fig_timing, fig_throughput, fig_memory


def run_dashboard(debug: bool = False, port: int = 8050):
    """Run the dashboard server."""
    app = create_app()

    # Add callbacks
    @app.callback(
        Output("results-store", "data"),
        Input("reload-btn", "n_clicks"),
    )
    def load_results(n_clicks):
        return load_latest_results().to_dict()

    # This would need proper callback registration for full interactivity
    # For now, create a simpler version

    print(f"\n🚀 Dashboard starting at http://127.0.0.1:{port}")
    print("Press Ctrl+C to stop\n")

    app.run(debug=debug, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    run_dashboard(debug=args.debug, port=args.port)