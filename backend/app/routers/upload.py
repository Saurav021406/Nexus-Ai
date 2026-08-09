import io
import json
import uuid

import pandas as pd
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException

from app.deps import get_current_user
from app.supabase_client import supabase_admin

router = APIRouter(prefix="/datasets", tags=["datasets"])

BUCKET_NAME = "datasets"
MAX_FILE_SIZE_MB = 25
ALLOWED_EXTENSIONS = (".csv", ".xlsx", ".xls")


def _dtype_to_simple(dtype) -> str:
    """Map pandas dtypes to simple labels for the frontend."""
    dtype_str = str(dtype)
    if "int" in dtype_str or "float" in dtype_str:
        return "numeric"
    if "datetime" in dtype_str:
        return "datetime"
    if "bool" in dtype_str:
        return "boolean"
    return "text"


def analyze_csv(df: pd.DataFrame) -> dict:
    """Compute basic EDA stats used by later steps (domain detection etc)."""
    columns_info = []
    for col in df.columns:
        series = df[col]
        col_info = {
            "name": col,
            "dtype": _dtype_to_simple(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_pct": round(float(series.isna().mean()) * 100, 2),
            "unique_count": int(series.nunique()),
        }
        if _dtype_to_simple(series.dtype) == "numeric":
            desc = series.describe()
            col_info["stats"] = {
                "mean": round(float(desc.get("mean", 0)), 2),
                "min": round(float(desc.get("min", 0)), 2),
                "max": round(float(desc.get("max", 0)), 2),
                "std": round(float(desc.get("std", 0)), 2),
            }
        columns_info.append(col_info)

    preview = df.head(10).fillna("").astype(str).to_dict(orient="records")

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns_info,
        "preview_rows": preview,
    }


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    user=Depends(get_current_user),
):
    if not file.filename or not file.filename.lower().endswith(ALLOWED_EXTENSIONS):
        raise HTTPException(status_code=400, detail="Only CSV or XLSX files are supported right now")

    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400, detail=f"File too large ({size_mb:.1f}MB). Max is {MAX_FILE_SIZE_MB}MB."
        )

    is_excel = file.filename.lower().endswith((".xlsx", ".xls"))
    try:
        if is_excel:
            df = pd.read_excel(io.BytesIO(raw_bytes))
        else:
            df = pd.read_csv(io.BytesIO(raw_bytes))
    except Exception:
        file_type = "Excel" if is_excel else "CSV"
        raise HTTPException(status_code=400, detail=f"Could not parse {file_type} file. Please check the format.")

    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded file appears to be empty")

    dataset_id = str(uuid.uuid4())
    safe_filename = file.filename.replace("/", "_").replace("\\", "_")
    storage_path = f"{user.id}/{dataset_id}_{safe_filename}"

    content_type = (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if is_excel else "text/csv"
    )
    try:
        supabase_admin.storage.from_(BUCKET_NAME).upload(
            storage_path,
            raw_bytes,
            {"content-type": content_type},
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Could not store the uploaded file. Please try again.")

    analysis = analyze_csv(df)

    try:
        supabase_admin.table("datasets").insert(
            {
                "id": dataset_id,
                "user_id": user.id,
                "filename": file.filename,
                "storage_path": storage_path,
                "row_count": analysis["row_count"],
                "column_count": analysis["column_count"],
                "analysis": json.dumps(analysis),
            }
        ).execute()
    except Exception as e:
        # Don't fail the whole request if only the metadata save fails - the
        # file is already safely in storage and we can still return analysis.
        print(f"Warning: could not save dataset metadata: {e}")

    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        **analysis,
    }


@router.get("")
async def list_datasets(user=Depends(get_current_user)):
    """Dataset history - only ever returns datasets owned by the current user."""
    try:
        result = (
            supabase_admin.table("datasets")
            .select("id, filename, row_count, column_count, created_at")
            .eq("user_id", user.id)
            .order("created_at", desc=True)
            .execute()
        )
        return result.data
    except Exception:
        raise HTTPException(status_code=500, detail="Could not load your datasets. Please try again.")


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str, user=Depends(get_current_user)):
    """Re-open a previously uploaded dataset using its saved analysis (no
    need to re-parse the CSV). Ownership is enforced via the user_id filter."""
    try:
        result = (
            supabase_admin.table("datasets")
            .select("id, filename, storage_path, analysis, user_id")
            .eq("id", dataset_id)
            .eq("user_id", user.id)
            .single()
            .execute()
        )
    except Exception:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = result.data
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        analysis = json.loads(dataset["analysis"]) if dataset.get("analysis") else None
    except Exception:
        analysis = None

    if not analysis:
        raise HTTPException(status_code=500, detail="This dataset's profile could not be loaded")

    return {
        "dataset_id": dataset["id"],
        "filename": dataset["filename"],
        **analysis,
    }


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str, user=Depends(get_current_user)):
    """Deletes both the stored file and its metadata row. Ownership is
    enforced - a user can only delete their own datasets."""
    try:
        result = (
            supabase_admin.table("datasets")
            .select("id, storage_path, user_id")
            .eq("id", dataset_id)
            .eq("user_id", user.id)
            .single()
            .execute()
        )
    except Exception:
        raise HTTPException(status_code=404, detail="Dataset not found")

    dataset = result.data
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        supabase_admin.storage.from_(BUCKET_NAME).remove([dataset["storage_path"]])
    except Exception as e:
        print(f"Warning: could not delete storage file: {e}")

    try:
        supabase_admin.table("datasets").delete().eq("id", dataset_id).eq("user_id", user.id).execute()
    except Exception:
        raise HTTPException(status_code=500, detail="Could not delete dataset record")

    return {"deleted": True, "dataset_id": dataset_id}
