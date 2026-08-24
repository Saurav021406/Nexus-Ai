"""Data Engineer service: data quality analysis and cleaning operations.

Kept as pure pandas functions (no I/O, no HTTP) so this can be reused later
by other callers (a future Manager Agent, ML pipeline, etc) without
depending on FastAPI request/response objects.
"""

import numpy as np
import pandas as pd

IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3.0


def analyze_data_quality(df: pd.DataFrame) -> dict:
    """Read-only inspection - reports issues without changing the data."""
    missing = {}
    for col in df.columns:
        count = int(df[col].isna().sum())
        if count > 0:
            missing[col] = {"count": count, "pct": round(float(df[col].isna().mean()) * 100, 2)}

    duplicate_count = int(df.duplicated().sum())

    outliers = {}
    outliers_zscore = {}
    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr != 0:
            lower = q1 - IQR_MULTIPLIER * iqr
            upper = q3 + IQR_MULTIPLIER * iqr
            count = int(((series < lower) | (series > upper)).sum())
            if count > 0:
                outliers[col] = {"count": count, "lower_bound": round(float(lower), 2), "upper_bound": round(float(upper), 2)}

        std = series.std()
        if std and std > 0:
            z_scores = (series - series.mean()) / std
            z_count = int((z_scores.abs() > ZSCORE_THRESHOLD).sum())
            if z_count > 0:
                outliers_zscore[col] = {"count": z_count, "threshold": ZSCORE_THRESHOLD}

    type_issues = []

    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna()
        if len(sample) == 0:
            continue
        numeric_parsed = pd.to_numeric(sample, errors="coerce")
        if numeric_parsed.notna().mean() > 0.9:
            type_issues.append({"column": col, "detected_as": "numeric", "currently": "text"})
            continue
        date_parsed = pd.to_datetime(sample, format="mixed", errors="coerce")
        if date_parsed.notna().mean() > 0.9:
            type_issues.append({"column": col, "detected_as": "date", "currently": "text"})

    total_issues = len(missing) + (1 if duplicate_count else 0) + len(outliers) + len(type_issues)
    # Simple, explainable scoring - not a statistically rigorous metric, just
    # a quick "how messy is this dataset" signal for the UI.
    score = max(0, 100 - (len(missing) * 8) - (5 if duplicate_count else 0) - (len(outliers) * 5) - (len(type_issues) * 5))

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_values": missing,
        "duplicate_rows": duplicate_count,
        "outliers": outliers,
        "outliers_zscore": outliers_zscore,
        "type_issues": type_issues,
        "quality_score": score,
    }


def clean_dataset(df: pd.DataFrame, options: dict) -> tuple[pd.DataFrame, dict]:
    """Applies the requested cleaning steps and returns (cleaned_df, report).

    options:
        fill_missing: bool
        missing_strategy: 'mean' | 'median' | 'mode' | 'drop_rows'
        remove_duplicates: bool
        fix_types: bool
    """
    before_rows = len(df)
    before_missing = int(df.isna().sum().sum())
    before_duplicates = int(df.duplicated().sum())

    cleaned = df.copy()
    report_steps = []

    if options.get("fix_types"):
        fixed_columns = []
        for col in cleaned.select_dtypes(include="object").columns:
            sample = cleaned[col].dropna()
            if len(sample) == 0:
                continue
            numeric_parsed = pd.to_numeric(sample, errors="coerce")
            if numeric_parsed.notna().mean() > 0.9:
                cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
                fixed_columns.append(col)
                continue
            date_parsed = pd.to_datetime(sample, format="mixed", errors="coerce")
            if date_parsed.notna().mean() > 0.9:
                cleaned[col] = pd.to_datetime(cleaned[col], format="mixed", errors="coerce")
                fixed_columns.append(col)
        report_steps.append({"step": "fix_types", "columns_fixed": fixed_columns})

    if options.get("fill_missing"):
        strategy = options.get("missing_strategy", "mean")
        filled_columns = []

        if strategy == "drop_rows":
            rows_before = len(cleaned)
            cleaned = cleaned.dropna()
            report_steps.append({"step": "fill_missing", "strategy": "drop_rows", "rows_dropped": rows_before - len(cleaned)})
        else:
            for col in cleaned.columns:
                if cleaned[col].isna().sum() == 0:
                    continue
                if pd.api.types.is_numeric_dtype(cleaned[col]):
                    if strategy == "median":
                        value = cleaned[col].median()
                    else:
                        value = cleaned[col].mean()
                    cleaned[col] = cleaned[col].fillna(value)
                else:
                    mode_values = cleaned[col].mode()
                    value = mode_values.iloc[0] if len(mode_values) > 0 else "Unknown"
                    cleaned[col] = cleaned[col].fillna(value)
                filled_columns.append(col)
            report_steps.append({"step": "fill_missing", "strategy": strategy, "columns_filled": filled_columns})

    if options.get("remove_duplicates"):
        rows_before = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        report_steps.append({"step": "remove_duplicates", "rows_removed": rows_before - len(cleaned)})

    cleaned = cleaned.reset_index(drop=True)

    report = {
        "steps_applied": report_steps,
        "before": {
            "row_count": before_rows,
            "missing_values": before_missing,
            "duplicate_rows": before_duplicates,
        },
        "after": {
            "row_count": len(cleaned),
            "missing_values": int(cleaned.isna().sum().sum()),
            "duplicate_rows": int(cleaned.duplicated().sum()),
        },
    }

    return cleaned, report
