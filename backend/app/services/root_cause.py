"""Root Cause Analysis (one of the 3 starred ⭐⭐⭐ features in the master
vision doc). Instead of just reporting "revenue dropped 12%", this finds
WHICH segments actually drove that change - "primarily driven by the West
region (-8 points of the -12%) and Product X (-5 points)" - by comparing
two time periods and decomposing the change across every other column in
the dataset.

    dataset + metric column + date column
        -> split into two adjacent time periods (before / after)
        -> for every other categorical/low-cardinality column:
               group by (period, segment value), sum/average the metric
               compute each segment's delta and % contribution to the
               OVERALL change (not just its own % change - a segment can
               grow while still being unimportant to the total swing)
        -> rank every segment across every column by contribution
        -> explain_root_cause(): Business Analyst LLM layer - plain
           English, grounded ONLY in the exact numbers above, same
           discipline every other specialist in this codebase follows

This is pure pandas arithmetic (no model training, no LLM call) up until
the final narrative step - same "free where code can determine it for
free" philosophy as Domain Gate, Evidence Gate, and AutoML's own
ID-column/leakage detection elsewhere in this codebase.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.services.consensus import get_consensus_json

# A column this unique per row is almost certainly an identifier, not a
# real segment - grouping by it would produce one row per group, which is
# meaningless for "what segment drove this change." Same threshold-based
# philosophy as automl.py's ID-column detection, kept independent here
# since root cause analysis doesn't need automl's full feature-matrix
# machinery, just this one check.
ID_LIKE_NAME_PATTERN = re.compile(r"(^id$|_id$|^uuid$|_uuid$|^guid$)", re.IGNORECASE)

# A numeric column with this few distinct values is treated as a segment
# (e.g. a 1-5 rating, a store number) rather than a continuous measurement
# - same threshold automl.py's detect_problem_type() uses for the same
# "is this actually a category" judgment call.
MAX_NUMERIC_SEGMENT_VALUES = 15
MAX_SEGMENT_VALUES_PER_COLUMN = 30  # a column with more distinct values than this is too granular to summarize usefully
TOP_N_CONTRIBUTORS = 15


@dataclass
class SegmentContribution:
    segment_column: str
    segment_value: str
    before: float
    after: float
    delta: float
    contribution_pct: float | None  # None only when overall_change is exactly 0 (contribution is undefined)


@dataclass
class RootCauseResult:
    metric_column: str
    date_column: str
    aggregation: str
    direction: str
    period_before: dict[str, Any]
    period_after: dict[str, Any]
    overall_before: float
    overall_after: float
    overall_change: float
    overall_pct_change: float | None
    segment_columns_analyzed: list[str]
    top_contributors: list[SegmentContribution] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _is_segment_column(series: pd.Series, name: str) -> bool:
    if series.nunique(dropna=True) > MAX_SEGMENT_VALUES_PER_COLUMN:
        return False
    if pd.api.types.is_numeric_dtype(series):
        return series.nunique(dropna=True) <= MAX_NUMERIC_SEGMENT_VALUES
    return True


def _detect_segment_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    columns = []
    for col in df.columns:
        if col in exclude:
            continue
        if ID_LIKE_NAME_PATTERN.search(str(col)):
            continue
        if _is_segment_column(df[col], col):
            columns.append(col)
    return columns


def analyze_root_cause(
    df: pd.DataFrame,
    metric_column: str,
    date_column: str,
    aggregation: str = "sum",
    direction: str = "decrease",
    segment_columns: list[str] | None = None,
) -> RootCauseResult:
    """Splits the dataset's date range into two equal-length ADJACENT
    periods (earlier half = "before", later half = "after") rather than a
    fixed window like "last 30 days" - this adapts to whatever time span
    the dataset actually covers instead of assuming one.

    direction="decrease" ranks the segments that dropped the most first
    (the worst offenders behind a decline); direction="increase" ranks
    the segments that grew the most first (the best contributors behind a
    rise) - same underlying computation either way, just sorted for what
    the question actually being asked.
    """
    if metric_column not in df.columns:
        raise ValueError(f"Metric column '{metric_column}' is not in this dataset.")
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' is not in this dataset.")
    if aggregation not in ("sum", "mean"):
        raise ValueError("aggregation must be 'sum' or 'mean'.")
    if direction not in ("decrease", "increase"):
        raise ValueError("direction must be 'decrease' or 'increase'.")
    if not pd.api.types.is_numeric_dtype(df[metric_column]):
        raise ValueError(f"Metric column '{metric_column}' must be numeric to aggregate.")

    working = df.copy()
    working[date_column] = pd.to_datetime(working[date_column], errors="coerce")
    working = working.dropna(subset=[date_column, metric_column])

    if len(working) < 10:
        raise ValueError(
            f"Only {len(working)} usable rows with both a valid date and metric value - too few to "
            "split into before/after periods reliably (need at least 10)."
        )

    min_date, max_date = working[date_column].min(), working[date_column].max()
    if min_date == max_date:
        raise ValueError(f"Every row in '{date_column}' has the same date - there's no time range to split.")

    midpoint = min_date + (max_date - min_date) / 2
    before_mask = working[date_column] < midpoint
    before_df, after_df = working[before_mask], working[~before_mask]

    if len(before_df) == 0 or len(after_df) == 0:
        raise ValueError("The date range couldn't be split into two non-empty periods.")

    agg_fn = "sum" if aggregation == "sum" else "mean"
    overall_before = float(before_df[metric_column].agg(agg_fn))
    overall_after = float(after_df[metric_column].agg(agg_fn))
    overall_change = overall_after - overall_before
    overall_pct_change = round((overall_change / overall_before) * 100, 2) if overall_before != 0 else None

    resolved_segment_columns = segment_columns or _detect_segment_columns(
        working, exclude={metric_column, date_column}
    )
    warnings: list[str] = []
    if not resolved_segment_columns:
        warnings.append(
            "No usable segment columns were found (either too many distinct values, or every other "
            "column looks like an identifier) - only the overall before/after totals are available."
        )

    all_contributions: list[SegmentContribution] = []
    for col in resolved_segment_columns:
        before_grouped = before_df.groupby(col, dropna=False)[metric_column].agg(agg_fn)
        after_grouped = after_df.groupby(col, dropna=False)[metric_column].agg(agg_fn)
        all_values = set(before_grouped.index) | set(after_grouped.index)

        for value in all_values:
            before_value = float(before_grouped.get(value, 0.0))
            after_value = float(after_grouped.get(value, 0.0))
            delta = after_value - before_value
            contribution_pct = round((delta / overall_change) * 100, 2) if overall_change != 0 else None
            all_contributions.append(
                SegmentContribution(
                    segment_column=col,
                    segment_value=str(value),
                    before=round(before_value, 4),
                    after=round(after_value, 4),
                    delta=round(delta, 4),
                    contribution_pct=contribution_pct,
                )
            )

    reverse = direction == "increase"
    all_contributions.sort(key=lambda c: c.delta, reverse=reverse)
    top_contributors = all_contributions[:TOP_N_CONTRIBUTORS]

    return RootCauseResult(
        metric_column=metric_column,
        date_column=date_column,
        aggregation=aggregation,
        direction=direction,
        period_before={"start": str(min_date.date()), "end": str((midpoint).date()), "n_rows": len(before_df)},
        period_after={"start": str((midpoint).date()), "end": str(max_date.date()), "n_rows": len(after_df)},
        overall_before=round(overall_before, 4),
        overall_after=round(overall_after, 4),
        overall_change=round(overall_change, 4),
        overall_pct_change=overall_pct_change,
        segment_columns_analyzed=resolved_segment_columns,
        top_contributors=top_contributors,
        warnings=warnings,
    )


def explain_root_cause(result: RootCauseResult) -> dict:
    """The Business Analyst layer: plain-English narrative grounded ONLY
    in the exact numbers already computed above - same never-invent-a-
    number discipline every other specialist in this codebase follows."""
    contributors_summary = "\n".join(
        f"- {c.segment_column} = \"{c.segment_value}\": {c.before} -> {c.after} "
        f"(delta {c.delta:+}, {c.contribution_pct}% of the total change)"
        for c in result.top_contributors[:10]
    )

    prompt = f"""You are a Business Analyst explaining WHY a metric changed, not just that it changed.
Use ONLY the exact numbers given below - never invent, estimate, or round differently.

Metric: {result.metric_column} ({result.aggregation})
Period before ({result.period_before['start']} to {result.period_before['end']}): {result.overall_before}
Period after ({result.period_after['start']} to {result.period_after['end']}): {result.overall_after}
Overall change: {result.overall_change} ({result.overall_pct_change}%)

TOP CONTRIBUTING SEGMENTS (sorted by impact on the {result.direction}):
{contributors_summary or "No segment breakdown was available."}

Write a plain-English root cause explanation: state the overall change, then name the 2-4 segments
that actually drove it, using their exact contribution numbers. Don't just restate the table -
synthesize what it MEANS (e.g. "the drop was concentrated in one region" vs "broadly distributed
across all segments").

Respond ONLY in this exact JSON format, no extra text:
{{
  "summary": "one paragraph explaining what changed and why, using only exact numbers above",
  "key_metrics": ["plain-English restatement of a key contributor 1", "contributor 2", "contributor 3"],
  "recommendation": "one concrete next step tied to the actual root cause found"
}}"""

    return get_consensus_json(prompt, temperature=1, max_tokens=2048)
