"""Static chart generation for report exports.

The interactive dashboard (services/visualization.py) returns Plotly JSON for
the frontend to render live. PDF/DOCX exports need flattened raster images
instead, so this module renders a small, deliberately-limited set of charts
with matplotlib's non-interactive Agg backend (headless-safe, no kaleido -
this codebase already moved away from kaleido once because it hung the
backend, see git history).

Charts are chosen automatically from the dataset's own columns - never from
sensitive/identifier columns - and are grounded entirely in computed
statistics, not LLM guesses.
"""

from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")  # headless - no display server needed
import matplotlib.pyplot as plt
import pandas as pd

from app.services.datasets import _is_identifier_column, _is_sensitive_column

MAX_CHARTS = 3
BG_COLOR = "#0f172a"
TEXT_COLOR = "#e2e8f0"
GRID_COLOR = "#334155"
ACCENT_COLORS = ["#22d3ee", "#a78bfa", "#34d399", "#fbbf24", "#f87171"]


def _style_axes(ax) -> None:
    ax.set_facecolor(BG_COLOR)
    ax.figure.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=8)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.6)


def _fig_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def _usable_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        c for c in df.columns if not _is_sensitive_column(c) and not _is_identifier_column(c)
    ]
    return df[keep]


def generate_report_charts(df: pd.DataFrame) -> list[dict]:
    """Returns up to MAX_CHARTS charts as [{title, image_base64}]."""
    charts: list[dict] = []
    safe_df = _usable_columns(df)

    numeric_cols = safe_df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [
        c
        for c in safe_df.columns
        if c not in numeric_cols and safe_df[c].nunique() <= 20 and safe_df[c].nunique() >= 2
    ]

    # 1. Correlation heatmap, if there's enough numeric signal.
    if len(numeric_cols) >= 3 and len(charts) < MAX_CHARTS:
        try:
            corr = safe_df[numeric_cols[:10]].corr(numeric_only=True)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(corr.values, cmap="viridis", vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7, color=TEXT_COLOR)
            ax.set_yticklabels(corr.columns, fontsize=7, color=TEXT_COLOR)
            ax.set_title("Correlation Between Numeric Fields")
            _style_axes(ax)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
            plt.setp(cbar.ax.get_yticklabels(), color=TEXT_COLOR)
            charts.append({"title": "Correlation Heatmap", "image_base64": _fig_to_base64(fig)})
        except Exception as e:
            print(f"report chart (heatmap) skipped: {e}")

    # 2. Distribution of the first numeric column.
    if numeric_cols and len(charts) < MAX_CHARTS:
        try:
            col = numeric_cols[0]
            series = pd.to_numeric(safe_df[col], errors="coerce").dropna()
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(series, bins=20, color=ACCENT_COLORS[0])
            ax.set_title(f"Distribution of {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            _style_axes(ax)
            charts.append({"title": f"Distribution of {col}", "image_base64": _fig_to_base64(fig)})
        except Exception as e:
            print(f"report chart (histogram) skipped: {e}")

    # 3. Top categories of the first usable categorical column.
    if categorical_cols and len(charts) < MAX_CHARTS:
        try:
            col = categorical_cols[0]
            counts = safe_df[col].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(counts.index.astype(str), counts.values, color=ACCENT_COLORS[1])
            ax.set_title(f"Top {col} Categories")
            ax.set_ylabel("Count")
            plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=7)
            _style_axes(ax)
            charts.append({"title": f"Top {col} Categories", "image_base64": _fig_to_base64(fig)})
        except Exception as e:
            print(f"report chart (bar) skipped: {e}")

    return charts
