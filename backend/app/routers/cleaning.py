"""Data quality inspection and cleaning endpoints."""

import io

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool  # <-- Added this missing import
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.deps import get_current_user
from app.services import approvals
from app.services.datasets import get_dataset_dataframe, get_dataset_record, BUCKET_NAME
from app.services.cleaning import analyze_data_quality, clean_dataset
from app.supabase_client import supabase_admin

router = APIRouter(prefix="/clean", tags=["cleaning"])

RESOURCE_TYPE = "cleaning"


class DatasetIdRequest(BaseModel):
    dataset_id: str


class CleanRequest(BaseModel):
    dataset_id: str
    fill_missing: bool = True
    missing_strategy: str = "mean"  # mean | median | mode | drop_rows
    remove_duplicates: bool = True
    fix_types: bool = True


class DownloadCleanedRequest(BaseModel):
    dataset_id: str
    version_id: str | None = None  # None = most recent version


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

    # Fixed: wrapped get_dataset_record because it is also a blocking network call
    dataset = await run_in_threadpool(get_dataset_record, payload.dataset_id, user.id)

    def _save() -> dict:
        # Version history (Human Approval Hooks pattern, same table
        # routers/agent.py already uses for Multi-Agent results - see
        # services/approvals.py): each cleaning run gets its OWN storage
        # path (suffixed with the version number) instead of overwriting
        # the last one, so re-cleaning with different settings doesn't
        # silently discard an earlier result the user might want back.
        version_number = approvals.get_next_version_number(RESOURCE_TYPE, payload.dataset_id, user.id)
        cleaned_path = f"{user.id}/{payload.dataset_id}_cleaned_v{version_number}.csv"

        csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
        try:
            supabase_admin.storage.from_(BUCKET_NAME).upload(
                cleaned_path,
                csv_bytes,
                {"content-type": "text/csv", "upsert": "true"},
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not save cleaned dataset: {e}")

        version = approvals.create_version(
            resource_type=RESOURCE_TYPE,
            resource_id=payload.dataset_id,
            user_id=user.id,
            content={"report": report, "storage_path": cleaned_path, "options": options},
            dataset_id=payload.dataset_id,
        )

        # Also keep updating the datasets table's own cleaned_storage_path/
        # cleaning_report fields (always pointing at the LATEST version) -
        # unchanged from before, so anything else that already reads those
        # two columns keeps working exactly as it did.
        try:
            supabase_admin.table("datasets").update(
                {"cleaned_storage_path": cleaned_path, "cleaning_report": report}
            ).eq("id", payload.dataset_id).eq("user_id", user.id).execute()
        except Exception as e:
            print(f"Warning: could not save cleaning report: {e}")

        return version

    version = await run_in_threadpool(_save)

    preview = cleaned_df.head(10).fillna("").astype(str).to_dict(orient="records")

    return {
        "dataset_id": payload.dataset_id,
        "filename": dataset["filename"],
        "report": report,
        "preview_rows": preview,
        "columns": list(cleaned_df.columns),
        "version_id": version["id"],
        "version_number": version["version_number"],
    }


@router.get("/versions")
async def list_cleaning_versions(dataset_id: str, user=Depends(get_current_user)):
    """History of every past cleaning run for this dataset (Human Approval
    Hooks pattern) - lets the UI show "Version 3 (2 min ago)" etc. instead
    of only ever exposing the single most recent result."""
    versions = await run_in_threadpool(approvals.list_versions, RESOURCE_TYPE, dataset_id, user.id)
    return {
        "versions": [
            {
                "id": v["id"],
                "version_number": v["version_number"],
                "created_at": v["created_at"],
                "report": v["content"].get("report"),
                "options": v["content"].get("options"),
            }
            for v in versions
        ]
    }


@router.post("/download")
async def download_cleaned(payload: DownloadCleanedRequest, user=Depends(get_current_user)):
    """Downloads a cleaned CSV that /clean/apply already saved to storage -
    this endpoint doesn't clean anything itself, it just serves back what's
    already there, same StreamingResponse + Content-Disposition pattern
    routers/report.py already uses for PDF/DOCX downloads.

    With no version_id given, downloads the most recent version (via the
    datasets table's cleaned_storage_path, unchanged fast path). With a
    version_id, downloads that SPECIFIC historical version instead - see
    /clean/versions for the list to pick from.
    """

    def _fetch() -> tuple[str, bytes]:
        try:
            dataset_result = (
                supabase_admin.table("datasets")
                .select("filename, cleaned_storage_path")
                .eq("id", payload.dataset_id)
                .eq("user_id", user.id)
                .single()
                .execute()
            )
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {e}")

        dataset_row = dataset_result.data
        if not dataset_row:
            raise HTTPException(status_code=404, detail="Dataset not found")

        if payload.version_id:
            version = approvals.get_version(payload.version_id, user.id)
            if not version or version["resource_type"] != RESOURCE_TYPE or version["resource_id"] != payload.dataset_id:
                raise HTTPException(status_code=404, detail="Cleaning version not found")
            storage_path = version["content"].get("storage_path")
        else:
            storage_path = dataset_row.get("cleaned_storage_path")

        if not storage_path:
            raise HTTPException(
                status_code=400,
                detail="No cleaned version exists yet for this dataset - run 'Apply cleaning' first.",
            )

        try:
            raw_bytes = supabase_admin.storage.from_(BUCKET_NAME).download(storage_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not download the cleaned dataset: {e}")

        return dataset_row["filename"], raw_bytes

    filename, raw_bytes = await run_in_threadpool(_fetch)

    base_name = filename.rsplit(".", 1)[0] if "." in filename else filename
    download_filename = f"{base_name}_cleaned.csv"

    return StreamingResponse(
        io.BytesIO(raw_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{download_filename}"'},
    )
