"""Data quality inspection and cleaning endpoints."""

import io

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool  # <-- Added this missing import
from pydantic import BaseModel

from app.deps import get_current_user
from app.services.datasets import get_dataset_dataframe, get_dataset_record, BUCKET_NAME
from app.services.cleaning import analyze_data_quality, clean_dataset
from app.supabase_client import supabase_admin

router = APIRouter(prefix="/clean", tags=["cleaning"])


class DatasetIdRequest(BaseModel):
    dataset_id: str


class CleanRequest(BaseModel):
    dataset_id: str
    fill_missing: bool = True
    missing_strategy: str = "mean"  # mean | median | mode | drop_rows
    remove_duplicates: bool = True
    fix_types: bool = True


@router.post("/quality")
async def check_quality(payload: DatasetIdRequest, user=Depends(get_current_user)):
    """Read-only - reports issues without modifying anything."""
    # Fixed: wrapped get_dataset_dataframe
    df = await run_in_threadpool(get_dataset_dataframe, payload.dataset_id, user.id)
    return analyze_data_quality(df)


@router.post("/apply")
async def apply_cleaning(payload: CleanRequest, user=Depends(get_current_user)):
    # Fixed: wrapped get_dataset_dataframe
    df = await run_in_threadpool(get_dataset_dataframe, payload.dataset_id, user.id)

    options = {
        "fill_missing": payload.fill_missing,
        "missing_strategy": payload.missing_strategy,
        "remove_duplicates": payload.remove_duplicates,
        "fix_types": payload.fix_types,
    }
    cleaned_df, report = clean_dataset(df, options)

    # Save the cleaned version as a separate file - the original upload is
    # never overwritten, so the user can always go back to the raw data.
    
    # Fixed: wrapped get_dataset_record because it is also a blocking network call
    dataset = await run_in_threadpool(get_dataset_record, payload.dataset_id, user.id)
    
    cleaned_path = f"{user.id}/{payload.dataset_id}_cleaned.csv"

    csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
    try:
        supabase_admin.storage.from_(BUCKET_NAME).upload(
            cleaned_path,
            csv_bytes,
            {"content-type": "text/csv", "upsert": "true"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save cleaned dataset: {e}")

    try:
        supabase_admin.table("datasets").update(
            {
                "cleaned_storage_path": cleaned_path,
                "cleaning_report": report,
            }
        ).eq("id", payload.dataset_id).eq("user_id", user.id).execute()
    except Exception as e:
        print(f"Warning: could not save cleaning report: {e}")

    preview = cleaned_df.head(10).fillna("").astype(str).to_dict(orient="records")

    return {
        "dataset_id": payload.dataset_id,
        "filename": dataset["filename"],
        "report": report,
        "preview_rows": preview,
        "columns": list(cleaned_df.columns),
    }