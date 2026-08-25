import io
import re

import pandas as pd
from fastapi import HTTPException
from app.supabase_client import supabase_admin

BUCKET_NAME = "datasets"

# A model never needs direct identifiers to explain aggregate trends. These
# fields are excluded from LLM context, even when they contain only a few values.
SENSITIVE_COLUMN_MARKERS = (
    "address",
    "aadhaar",
    "birth",
    "date of birth",
    "dob",
    "email",
    "ip address",
    "name",
    "passport",
    "phone",
    "mobile",
    "social security",
    "ssn",
    "user id",
    "customer id",
    "employee id",
    "patient id",
)
IDENTIFIER_COLUMN_MARKERS = (" id", "_id", "uuid", "identifier")
MAX_CATEGORICAL_COLUMNS = 12
MAX_CATEGORIES_PER_COLUMN = 20


def get_dataset_record(dataset_id: str, user_id: str) -> dict:
    """Return metadata only when the dataset belongs to the calling user."""
    try:
        result = (
            supabase_admin.table("datasets")
            .select("id, filename, storage_path, user_id")
            .eq("id", dataset_id)
            .eq("user_id", user_id)
            .single()
            .execute()
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {e}")

    dataset = result.data
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return dataset


def get_dataset_dataframe(dataset_id: str, user_id: str) -> pd.DataFrame:
    """Load the full CSV/Excel from storage, never the small frontend preview."""
    dataset = get_dataset_record(dataset_id, user_id)

    path_lower = dataset["storage_path"].lower()
    if path_lower.endswith((".pdf", ".docx")):
        raise HTTPException(
            status_code=400,
            detail=(
                "This is a document dataset (PDF/Word), not a spreadsheet - tabular "
                "analysis isn't applicable here. Document-based analysis (RAG) is "
                "being built separately."
            ),
        )

    try:
        raw_bytes = supabase_admin.storage.from_(BUCKET_NAME).download(dataset["storage_path"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not fetch dataset file: {e}")

    try:
        if path_lower.endswith(".xlsx") or path_lower.endswith(".xls"):
            return pd.read_excel(io.BytesIO(raw_bytes))
        return pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse stored file: {e}")


def is_document_dataset(dataset_id: str, user_id: str) -> bool:
    dataset = get_dataset_record(dataset_id, user_id)
    return dataset["storage_path"].lower().endswith((".pdf", ".docx"))


def get_document_text(dataset_id: str, user_id: str) -> str:
    """Returns the full extracted text for a document dataset (Step 2's
    output), read back from the `analysis` blob saved at upload time -
    avoids re-downloading and re-parsing the original PDF/DOCX on every
    call. Used by later RAG steps (chunking/embeddings/retrieval)."""
    import json

    result = (
        supabase_admin.table("datasets")
        .select("analysis, storage_path")
        .eq("id", dataset_id)
        .eq("user_id", user_id)
        .single()
        .execute()
    )
    dataset = result.data
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    if not dataset["storage_path"].lower().endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="This dataset is not a document.")

    try:
        analysis = json.loads(dataset["analysis"]) if dataset.get("analysis") else {}
    except Exception:
        analysis = {}

    text = analysis.get("extracted_text")
    if not text:
        raise HTTPException(status_code=500, detail="No extracted text found for this document.")
    return text


def _normalise_column_name(column: object) -> str:
    value = str(column).replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", value.lower()).strip()


def _is_sensitive_column(column: object) -> bool:
    normalised = _normalise_column_name(column)
    return any(marker in normalised for marker in SENSITIVE_COLUMN_MARKERS)


def _is_identifier_column(column: object) -> bool:
    normalised = _normalise_column_name(column)
    return normalised == "id" or any(marker in normalised for marker in IDENTIFIER_COLUMN_MARKERS)


def _looks_like_direct_identifier(value: object) -> bool:
    text = str(value)
    return bool(
        re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", text)
        or re.search(r"\+?\d[\d ()-]{7,}\d", text)
    )


def build_data_summary(df: pd.DataFrame) -> str:
    """Builds a text summary with EXACT statistics computed in pandas (never
    guessed by the LLM). Raw dataset rows are intentionally not included in the
    model context, which keeps analysis accurate while reducing PII exposure."""
    lines = [
        f"Total rows: {len(df)}",
        f"Total columns: {len(df.columns)}",
        f"Column names: {', '.join(str(column) for column in df.columns)}",
        "",
    ]

    numeric_cols = df.select_dtypes(include="number").columns
    numeric_cols_filtered = [
        col for col in numeric_cols if not _is_sensitive_column(col) and not _is_identifier_column(col)
    ]
    if len(numeric_cols_filtered) > 0:
        lines.append("Numeric column statistics (computed precisely from the FULL dataset):")
        for col in numeric_cols_filtered:
            s = df[col].dropna()
            if len(s) == 0:
                continue
            lines.append(
                f"- {col}: sum={s.sum():.2f}, mean={s.mean():.2f}, "
                f"min={s.min():.2f}, max={s.max():.2f}, count={len(s)}"
            )
        lines.append("")

    # Pairwise correlations - without this, "which factor most affects X"
    # style questions can't actually be answered (means alone don't say
    # anything about how columns move together). Pandas computes this
    # exactly from the full data, same as every other number in this
    # summary - nothing here is estimated or guessed by a model.
    if len(numeric_cols_filtered) >= 2:
        corr_matrix = df[numeric_cols_filtered].corr(numeric_only=True)
        lines.append("Pairwise correlations between numeric columns (Pearson r, -1 to 1; closer to +/-1 = stronger relationship):")
        seen_pairs = set()
        for col_a in numeric_cols_filtered:
            for col_b in numeric_cols_filtered:
                if col_a == col_b or (col_b, col_a) in seen_pairs:
                    continue
                seen_pairs.add((col_a, col_b))
                value = corr_matrix.loc[col_a, col_b]
                if pd.notna(value):
                    lines.append(f"- {col_a} vs {col_b}: r={value:.3f}")
        lines.append("")

    categorical_cols = df.select_dtypes(exclude="number").columns
    included_categorical_columns = 0
    for col in categorical_cols:
        if included_categorical_columns >= MAX_CATEGORICAL_COLUMNS:
            break
        if _is_sensitive_column(col) or _is_identifier_column(col):
            continue
        nunique = df[col].nunique()
        if 0 < nunique <= MAX_CATEGORIES_PER_COLUMN:
            counts = df[col].value_counts()
            if any(_looks_like_direct_identifier(value) for value in counts.index):
                continue
            lines.append(f"'{col}' breakdown (exact count per value):")
            for val, cnt in counts.items():
                lines.append(f"  - {val}: {cnt}")
            lines.append("")
            included_categorical_columns += 1

    lines.append(
        "Privacy note: raw rows and direct identifiers are intentionally excluded from this model context."
    )

    return "\n".join(lines)
