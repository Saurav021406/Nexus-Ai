"""Visualization Agent: turns a dataset (+ either explicit chart params or a
natural-language request) into a rendered chart. Charts are rendered with
matplotlib directly in-process (no subprocess/browser involved, unlike
Plotly's kaleido export) and returned as base64 PNG - the frontend just
needs an <img> tag."""

import base64
import io
import json

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for a server process
import matplotlib.pyplot as plt
import pandas as pd
import google.generativeai as genai

from app.config import settings

genai.configure(api_key=settings.gemini_api_key)

VALID_CHART_TYPES = {"histogram", "bar", "line", "scatter", "heatmap"}

BG_COLOR = "#0f172a"
TEXT_COLOR = "#e2e8f0"
GRID_COLOR = "#334155"
ACCENT_COLORS = ["#60a5fa", "#a78bfa", "#34d399", "#fbbf24", "#f87171", "#22d3ee"]


def _new_figure():
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=110)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.5)
    return fig, ax


def _fig_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def generate_chart(
    df: pd.DataFrame,
    chart_type: str,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    agg: str = "sum",
    title: str | None = None,
) -> str:
    if chart_type not in VALID_CHART_TYPES:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    fig, ax = _new_figure()

    if chart_type == "histogram":
        if not x or x not in df.columns:
            raise ValueError("Histogram needs a valid 'x' column")
        series = pd.to_numeric(df[x], errors="coerce").dropna()
        ax.hist(series, bins=20, color=ACCENT_COLORS[0], edgecolor=BG_COLOR)
        ax.set_xlabel(x)
        ax.set_ylabel("Count")
        ax.set_title(title or f"Distribution of {x}")

    elif chart_type == "bar":
        if not x or x not in df.columns:
            raise ValueError("Bar chart needs a valid 'x' column")
        if y and y in df.columns:
            agg_func = {"sum": "sum", "mean": "mean", "count": "count", "median": "median"}.get(agg, "sum")
            grouped = df.groupby(x)[y].agg(agg_func).sort_values(ascending=False).head(25)
            ax.bar(grouped.index.astype(str), grouped.values, color=ACCENT_COLORS[1])
            ax.set_ylabel(f"{y} ({agg})")
            ax.set_title(title or f"{y} ({agg}) by {x}")
        else:
            counts = df[x].value_counts().head(25)
            ax.bar(counts.index.astype(str), counts.values, color=ACCENT_COLORS[1])
            ax.set_ylabel("Count")
            ax.set_title(title or f"Count by {x}")
        ax.set_xlabel(x)
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right")

    elif chart_type == "line":
        if not y or y not in df.columns:
            raise ValueError("Line chart needs a valid 'y' column")
        working = df.copy()
        if x and x in df.columns:
            try:
                parsed = pd.to_datetime(working[x], errors="coerce")
                if parsed.notna().sum() > 0:
                    working[x] = parsed
            except Exception:
                pass
            working = working.sort_values(x)
            ax.plot(working[x], working[y], color=ACCENT_COLORS[2], linewidth=2)
            ax.set_xlabel(x)
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        else:
            working = working.reset_index()
            ax.plot(working.index, working[y], color=ACCENT_COLORS[2], linewidth=2)
            ax.set_xlabel("Row index")
        ax.set_ylabel(y)
        ax.set_title(title or f"{y} trend")

    elif chart_type == "scatter":
        if not x or not y or x not in df.columns or y not in df.columns:
            raise ValueError("Scatter chart needs valid 'x' and 'y' columns")
        if color and color in df.columns:
            categories = df[color].astype(str).unique()[:6]
            for i, cat in enumerate(categories):
                subset = df[df[color].astype(str) == cat]
                ax.scatter(subset[x], subset[y], label=cat, color=ACCENT_COLORS[i % len(ACCENT_COLORS)], alpha=0.75)
            legend = ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR, fontsize=8)
            for text in legend.get_texts():
                text.set_color(TEXT_COLOR)
        else:
            ax.scatter(df[x], df[y], color=ACCENT_COLORS[0], alpha=0.75)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(title or f"{y} vs {x}")

    elif chart_type == "heatmap":
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            raise ValueError("Need at least 2 numeric columns for a correlation heatmap")
        corr = numeric_df.corr(numeric_only=True).round(2)
        im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr.columns)
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                ax.text(j, i, corr.values[i, j], ha="center", va="center", color=TEXT_COLOR, fontsize=8)
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
        plt.setp(cbar.ax.get_yticklabels(), color=TEXT_COLOR)
        ax.set_title(title or "Correlation heatmap")

    return _fig_to_base64(fig)


def suggest_chart_from_request(columns_info: list[dict], nl_request: str) -> dict:
    prompt = f"""You are a data visualization assistant. A user described a chart
they want in plain language. Based on the available columns below, decide the
best chart to build.

Available columns (name: type):
{json.dumps(columns_info, indent=2)}

User request: "{nl_request}"

Respond ONLY in this exact JSON format, no extra text:
{{
  "chart_type": "one of: histogram, bar, line, scatter, heatmap",
  "x": "column name or null",
  "y": "column name or null",
  "agg": "one of: sum, mean, count, median",
  "title": "a short descriptive chart title",
  "reasoning": "one short sentence explaining the choice"
}}"""

    model = genai.GenerativeModel("gemini-3-flash-preview")
    response = model.generate_content(prompt)
    text = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(text)


def generate_dashboard(df: pd.DataFrame) -> list[dict]:
    charts_config = []

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in df.select_dtypes(include="object").columns if df[c].nunique() <= 30]

    for col in numeric_cols[:2]:
        charts_config.append({"chart_type": "histogram", "x": col, "y": None, "title": f"Distribution of {col}"})

    if categorical_cols:
        cat = categorical_cols[0]
        charts_config.append({"chart_type": "bar", "x": cat, "y": None, "title": f"Count by {cat}"})

    if categorical_cols and numeric_cols:
        charts_config.append(
            {
                "chart_type": "bar",
                "x": categorical_cols[0],
                "y": numeric_cols[0],
                "title": f"{numeric_cols[0]} by {categorical_cols[0]}",
            }
        )

    if len(numeric_cols) >= 2:
        charts_config.append({"chart_type": "heatmap", "x": None, "y": None, "title": "Correlation heatmap"})
        charts_config.append(
            {
                "chart_type": "scatter",
                "x": numeric_cols[0],
                "y": numeric_cols[1],
                "title": f"{numeric_cols[1]} vs {numeric_cols[0]}",
            }
        )

    results = []
    for cfg in charts_config[:6]:
        try:
            image_b64 = generate_chart(
                df,
                chart_type=cfg["chart_type"],
                x=cfg.get("x"),
                y=cfg.get("y"),
                title=cfg.get("title"),
            )
            results.append({**cfg, "image_base64": image_b64})
        except Exception:
            continue

    return results
