"""Visualization Agent: turns a dataset (+ either explicit chart params or a
natural-language request) into one or more interactive charts. Charts are
built with Plotly and returned as Plotly's native JSON figure spec
(data + layout) - the frontend renders it directly with react-plotly.js,
which gives zoom/pan/hover/export-to-PNG for free.

Each chart also gets a short, rule-based "insight" - computed directly from
the data (never guessed by an LLM), so explanations are always accurate.
"""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.services.consensus import get_consensus_json

VALID_CHART_TYPES = {"histogram", "bar", "line", "scatter", "heatmap", "pie", "box", "area"}

BG_COLOR = "#0f172a"
TEXT_COLOR = "#e2e8f0"
GRID_COLOR = "#334155"
ACCENT_COLORS = ["#60a5fa", "#a78bfa", "#34d399", "#fbbf24", "#f87171", "#22d3ee"]


def _base_layout(title: str | None) -> dict:
    return dict(
        title=dict(text=title or "", font=dict(color=TEXT_COLOR, size=15)),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=TEXT_COLOR, size=12),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        margin=dict(l=50, r=30, t=50, b=50),
        legend=dict(font=dict(color=TEXT_COLOR)),
    )


def generate_chart(
    df: pd.DataFrame,
    chart_type: str,
    x: str | None = None,
    y: str | None = None,
    color: str | None = None,
    agg: str = "sum",
    title: str | None = None,
) -> dict:
    """Returns a Plotly figure as a plain JSON-serializable dict: {"data": [...], "layout": {...}}"""
    if chart_type not in VALID_CHART_TYPES:
        raise ValueError(f"Unsupported chart type: {chart_type}")

    fig = go.Figure()

    if chart_type == "histogram":
        if not x or x not in df.columns:
            raise ValueError("Histogram needs a valid 'x' column")
        series = pd.to_numeric(df[x], errors="coerce").dropna()
        fig.add_trace(go.Histogram(x=series, nbinsx=20, marker_color=ACCENT_COLORS[0]))
        fig.update_layout(**_base_layout(title or f"Distribution of {x}"))
        fig.update_xaxes(title_text=x)
        fig.update_yaxes(title_text="Count")

    elif chart_type == "bar":
        if not x or x not in df.columns:
            raise ValueError("Bar chart needs a valid 'x' column")
        if y and y in df.columns:
            agg_func = {"sum": "sum", "mean": "mean", "count": "count", "median": "median"}.get(agg, "sum")
            grouped = df.groupby(x)[y].agg(agg_func).sort_values(ascending=False).head(25)
            fig.add_trace(go.Bar(x=grouped.index.astype(str), y=grouped.values, marker_color=ACCENT_COLORS[1]))
            fig.update_yaxes(title_text=f"{y} ({agg})")
            fig.update_layout(**_base_layout(title or f"{y} ({agg}) by {x}"))
        else:
            counts = df[x].value_counts().head(25)
            fig.add_trace(go.Bar(x=counts.index.astype(str), y=counts.values, marker_color=ACCENT_COLORS[1]))
            fig.update_yaxes(title_text="Count")
            fig.update_layout(**_base_layout(title or f"Count by {x}"))
        fig.update_xaxes(title_text=x, tickangle=-40)

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
            fig.add_trace(
                go.Scatter(x=working[x], y=working[y], mode="lines", line=dict(color=ACCENT_COLORS[2], width=2))
            )
            fig.update_xaxes(title_text=x)
        else:
            working = working.reset_index()
            fig.add_trace(
                go.Scatter(x=working.index, y=working[y], mode="lines", line=dict(color=ACCENT_COLORS[2], width=2))
            )
            fig.update_xaxes(title_text="Row index")
        fig.update_yaxes(title_text=y)
        fig.update_layout(**_base_layout(title or f"{y} trend"))

    elif chart_type == "area":
        if not y or y not in df.columns:
            raise ValueError("Area chart needs a valid 'y' column")
        working = df.copy()
        if x and x in df.columns:
            try:
                parsed = pd.to_datetime(working[x], errors="coerce")
                if parsed.notna().sum() > 0:
                    working[x] = parsed
            except Exception:
                pass
            working = working.sort_values(x)
            x_vals = working[x]
            fig.update_xaxes(title_text=x)
        else:
            working = working.reset_index()
            x_vals = working.index
            fig.update_xaxes(title_text="Row index")
        fig.add_trace(
            go.Scatter(
                x=x_vals, y=working[y], mode="lines", fill="tozeroy",
                line=dict(color=ACCENT_COLORS[2], width=2),
                fillcolor="rgba(52, 211, 153, 0.25)",
            )
        )
        fig.update_yaxes(title_text=y)
        fig.update_layout(**_base_layout(title or f"{y} over time"))

    elif chart_type == "scatter":
        if not x or not y or x not in df.columns or y not in df.columns:
            raise ValueError("Scatter chart needs valid 'x' and 'y' columns")
        if color and color in df.columns:
            categories = df[color].astype(str).unique()[:6]
            for i, cat in enumerate(categories):
                subset = df[df[color].astype(str) == cat]
                fig.add_trace(
                    go.Scatter(
                        x=subset[x],
                        y=subset[y],
                        mode="markers",
                        name=str(cat),
                        marker=dict(color=ACCENT_COLORS[i % len(ACCENT_COLORS)], opacity=0.75, size=7),
                    )
                )
        else:
            fig.add_trace(
                go.Scatter(
                    x=df[x], y=df[y], mode="markers", marker=dict(color=ACCENT_COLORS[0], opacity=0.75, size=7)
                )
            )
        fig.update_xaxes(title_text=x)
        fig.update_yaxes(title_text=y)
        fig.update_layout(**_base_layout(title or f"{y} vs {x}"))

    elif chart_type == "pie":
        if not x or x not in df.columns:
            raise ValueError("Pie chart needs a valid 'x' column")
        if y and y in df.columns:
            grouped = df.groupby(x)[y].sum().sort_values(ascending=False).head(12)
            labels, values = grouped.index.astype(str), grouped.values
        else:
            counts = df[x].value_counts().head(12)
            labels, values = counts.index.astype(str), counts.values
        fig.add_trace(
            go.Pie(
                labels=labels, values=values, hole=0.35,
                marker=dict(colors=ACCENT_COLORS * 3),
                textfont=dict(color=TEXT_COLOR),
            )
        )
        fig.update_layout(**_base_layout(title or f"Share by {x}"))

    elif chart_type == "box":
        if not y or y not in df.columns:
            raise ValueError("Box plot needs a valid 'y' column")
        if x and x in df.columns:
            categories = df[x].astype(str).unique()[:12]
            for i, cat in enumerate(categories):
                subset = df[df[x].astype(str) == cat]
                fig.add_trace(
                    go.Box(y=subset[y], name=str(cat), marker_color=ACCENT_COLORS[i % len(ACCENT_COLORS)])
                )
            fig.update_xaxes(title_text=x)
        else:
            fig.add_trace(go.Box(y=df[y], name=y, marker_color=ACCENT_COLORS[0]))
        fig.update_yaxes(title_text=y)
        fig.update_layout(**_base_layout(title or f"Spread of {y}"))

    elif chart_type == "heatmap":
        numeric_df = df.select_dtypes(include="number")
        if numeric_df.shape[1] < 2:
            raise ValueError("Need at least 2 numeric columns for a correlation heatmap")
        corr = numeric_df.corr(numeric_only=True).round(2)
        fig.add_trace(
            go.Heatmap(
                z=corr.values,
                x=list(corr.columns),
                y=list(corr.columns),
                colorscale="RdBu",
                zmin=-1,
                zmax=1,
                text=corr.values,
                texttemplate="%{text}",
                textfont=dict(color=TEXT_COLOR, size=10),
                colorbar=dict(tickfont=dict(color=TEXT_COLOR)),
            )
        )
        fig.update_layout(**_base_layout(title or "Correlation heatmap"))
        fig.update_xaxes(tickangle=-45)

    return json.loads(fig.to_json())


def explain_chart(
    df: pd.DataFrame,
    chart_type: str,
    x: str | None = None,
    y: str | None = None,
    agg: str = "sum",
) -> str:
    """Computes a short, factual explanation of what the chart shows.
    Purely rule-based (pandas math), so it's always accurate - no LLM call
    needed per chart, which also keeps dashboard generation fast."""
    try:
        if chart_type == "histogram" and x in df.columns:
            s = pd.to_numeric(df[x], errors="coerce").dropna()
            if len(s) == 0:
                return ""
            return f"{x} ranges from {s.min():.1f} to {s.max():.1f}, averaging {s.mean():.1f}."

        if chart_type in ("bar", "pie") and x in df.columns:
            if y and y in df.columns:
                agg_func = {"sum": "sum", "mean": "mean", "count": "count", "median": "median"}.get(agg, "sum")
                grouped = df.groupby(x)[y].agg(agg_func).sort_values(ascending=False)
                top = grouped.index[0]
                share = grouped.iloc[0] / grouped.sum() * 100 if grouped.sum() else 0
                return f"'{top}' leads with {grouped.iloc[0]:,.1f} ({share:.0f}% of the total)."
            else:
                counts = df[x].value_counts()
                top = counts.index[0]
                share = counts.iloc[0] / counts.sum() * 100 if counts.sum() else 0
                return f"'{top}' is the most common value, appearing {counts.iloc[0]} times ({share:.0f}%)."

        if chart_type in ("line", "area") and y in df.columns:
            s = pd.to_numeric(df[y], errors="coerce").dropna()
            if len(s) < 2:
                return ""
            change = s.iloc[-1] - s.iloc[0]
            direction = "up" if change > 0 else "down" if change < 0 else "flat"
            return f"{y} trends {direction} overall, from {s.iloc[0]:.1f} to {s.iloc[-1]:.1f}."

        if chart_type == "scatter" and x in df.columns and y in df.columns:
            xs = pd.to_numeric(df[x], errors="coerce")
            ys = pd.to_numeric(df[y], errors="coerce")
            corr = xs.corr(ys)
            if pd.isna(corr):
                return ""
            strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.3 else "weak"
            direction = "positive" if corr > 0 else "negative"
            return f"{x} and {y} show a {strength} {direction} relationship (correlation {corr:.2f})."

        if chart_type == "box" and y in df.columns:
            s = pd.to_numeric(df[y], errors="coerce").dropna()
            if len(s) == 0:
                return ""
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            return f"Median {y} is {s.median():.1f}, with the middle 50% between {q1:.1f} and {q3:.1f}."

        if chart_type == "heatmap":
            numeric_df = df.select_dtypes(include="number")
            if numeric_df.shape[1] < 2:
                return ""
            corr = numeric_df.corr(numeric_only=True).abs()
            corr_vals = corr.where(~np.eye(len(corr), dtype=bool))
            max_pair = corr_vals.stack().idxmax()
            max_val = corr_vals.stack().max()
            return f"Strongest relationship: {max_pair[0]} and {max_pair[1]} (correlation {max_val:.2f})."
    except Exception:
        return ""

    return ""


def suggest_multiple_charts_from_request(columns_info: list[dict], nl_request: str) -> list[dict]:
    """Uses the Consensus Engine (Groq + NVIDIA Nemotron + MiniMax-M3) to
    translate a natural-language request into one or more chart specs - a
    single query like 'show me sales trends' can reasonably map to 2-3
    complementary charts. Python still does all the actual
    rendering/computation - the AI only decides WHAT to chart."""
    prompt = f"""You are a data visualization assistant. A user described what they
want to see in plain language. Based on the available columns below, decide
the best chart(s) to build - usually 1 chart, but use 2 or 3 complementary
charts if the request genuinely calls for multiple angles (e.g. "trends and
breakdown" or "show me an overview").

Available columns (name: type):
{json.dumps(columns_info, indent=2)}

User request: "{nl_request}"

Respond ONLY in this exact JSON format, no extra text - "charts" is a list
of 1 to 3 chart specs:
{{
  "charts": [
    {{
      "chart_type": "one of: histogram, bar, line, area, scatter, heatmap, pie, box",
      "x": "column name or null",
      "y": "column name or null",
      "agg": "one of: sum, mean, count, median",
      "title": "a short descriptive chart title",
      "reasoning": "one short sentence explaining this chart's choice"
    }}
  ]
}}"""

    parsed = get_consensus_json(prompt, temperature=0.7, max_tokens=1024)
    return parsed.get("charts", [])[:3]


def generate_dashboard(df: pd.DataFrame) -> list[dict]:
    charts_config = []

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in df.select_dtypes(include="object").columns if df[c].nunique() <= 30]

    for col in numeric_cols[:2]:
        charts_config.append({"chart_type": "histogram", "x": col, "y": None, "title": f"Distribution of {col}"})

    if categorical_cols:
        cat = categorical_cols[0]
        charts_config.append({"chart_type": "pie", "x": cat, "y": None, "title": f"Share by {cat}"})

    if categorical_cols and numeric_cols:
        charts_config.append(
            {
                "chart_type": "bar",
                "x": categorical_cols[0],
                "y": numeric_cols[0],
                "title": f"{numeric_cols[0]} by {categorical_cols[0]}",
            }
        )

    if numeric_cols:
        charts_config.append(
            {"chart_type": "box", "x": None, "y": numeric_cols[0], "title": f"Spread of {numeric_cols[0]}"}
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
            figure = generate_chart(
                df,
                chart_type=cfg["chart_type"],
                x=cfg.get("x"),
                y=cfg.get("y"),
                title=cfg.get("title"),
            )
            insight = explain_chart(df, cfg["chart_type"], cfg.get("x"), cfg.get("y"))
            results.append({**cfg, "figure": figure, "insight": insight})
        except Exception:
            continue

    return results
