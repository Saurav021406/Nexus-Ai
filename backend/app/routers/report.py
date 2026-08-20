"""Report Agent endpoints.

Flow: the frontend already has the Manager's final_output (from
POST /domain/analyze) sitting in state. It posts that here once to get back
structured report content (the "draft"). Human Approval Hooks: every
generate/resubmit call now also persists a new version row (see
services/report_versions.py) so the report can be approved, rejected (with a
reason, then regenerated or edited), or reviewed via version history -
without ever losing a prior version. PDF/DOCX export is blocked server-side
until a version's approval_status is 'approved'.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io

from app.agents.report import generate_report_content
from app.deps import get_current_user
from app.services.datasets import get_dataset_dataframe
from app.services.report_charts import generate_report_charts
from app.services.report_render import render_docx, render_pdf
from app.services import report_versions

router = APIRouter(prefix="/report", tags=["report"])


class GenerateReportRequest(BaseModel):
    dataset_id: str
    filename: str
    analysis: dict[str, Any]


class ExportReportRequest(BaseModel):
    report: dict[str, Any]


class RejectReportRequest(BaseModel):
    reason: str


class ResubmitReportRequest(BaseModel):
    executive_summary: str


def _version_response(version: dict[str, Any]) -> dict[str, Any]:
    """Flatten a report_versions row into the shape the frontend already
    expects from /generate (the content fields), plus approval metadata."""
    return {
        **version["content"],
        "id": version["id"],
        "version_number": version["version_number"],
        "approval_status": version["approval_status"],
        "rejection_reason": version.get("rejection_reason"),
        "created_at": version.get("created_at"),
    }


@router.post("/generate")
async def generate_report(payload: GenerateReportRequest, user=Depends(get_current_user)):
    if not payload.analysis:
        raise HTTPException(status_code=400, detail="analysis payload is required")

    try:
        # FIXED: Variable name is now 'dataframe' to match the next line
        dataframe = await run_in_threadpool(get_dataset_dataframe, payload.dataset_id, user.id)
        # FIXED: Wrapped chart generation too to avoid CPU blocking
        charts = await run_in_threadpool(generate_report_charts, dataframe)
    except Exception as e:
        # Charts are a nice-to-have, never block report generation on them.
        print(f"Chart generation skipped: {e}")
        charts = []

    content = await run_in_threadpool(
        generate_report_content,
        analysis=payload.analysis,
        filename=payload.filename,
        dataset_id=payload.dataset_id,
        charts=charts,
    )

    try:
        # FIXED: Wrapped database call
        version = await run_in_threadpool(
            report_versions.create_version, payload.dataset_id, user.id, content
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save report version: {e}")

    return _version_response(version)


@router.get("/versions")
async def get_versions(dataset_id: str, user=Depends(get_current_user)):
    # FIXED: Wrapped database call
    versions = await run_in_threadpool(report_versions.list_versions, dataset_id, user.id)
    return {"versions": [_version_response(v) for v in versions]}


@router.post("/{report_id}/approve")
async def approve_report(report_id: str, user=Depends(get_current_user)):
    try:
        # FIXED: Wrapped database call
        version = await run_in_threadpool(
            report_versions.set_status, report_id, user.id, "approved"
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Report version not found: {e}")
    return _version_response(version)


@router.post("/{report_id}/reject")
async def reject_report(report_id: str, payload: RejectReportRequest, user=Depends(get_current_user)):
    if not payload.reason.strip():
        raise HTTPException(status_code=400, detail="A rejection reason is required")
    try:
        # FIXED: Wrapped database call
        version = await run_in_threadpool(
            report_versions.set_status, report_id, user.id, "rejected", payload.reason.strip()
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Report version not found: {e}")
    return _version_response(version)


@router.post("/{report_id}/resubmit")
async def resubmit_report(report_id: str, payload: ResubmitReportRequest, user=Depends(get_current_user)):
    """Edit the executive summary of a rejected (or any prior) report and
    resubmit it as a new pending version. The version being edited stays in
    history untouched."""
    
    # FIXED: Wrapped database call
    previous = await run_in_threadpool(report_versions.get_version, report_id, user.id)
    if not previous:
        raise HTTPException(status_code=404, detail="Report version not found")

    if not payload.executive_summary.strip():
        raise HTTPException(status_code=400, detail="Executive summary cannot be empty")

    new_content = {**previous["content"], "executive_summary": payload.executive_summary.strip()}

    try:
        # FIXED: Wrapped database call
        version = await run_in_threadpool(
            report_versions.create_version, previous["dataset_id"], user.id, new_content
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save report version: {e}")

    return _version_response(version)


@router.post("/pdf")
async def export_pdf(payload: ExportReportRequest, user=Depends(get_current_user)):
    if payload.report.get("approval_status") != "approved":
        raise HTTPException(status_code=400, detail="This report must be approved before it can be downloaded")

    try:
        # FIXED: PDF rendering is CPU heavy, wrapped it to prevent lag
        pdf_bytes = await run_in_threadpool(render_pdf, payload.report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF export failed: {str(e)}")

    filename = f"{payload.report.get('filename', 'report')}_report.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.post("/docx")
async def export_docx(payload: ExportReportRequest, user=Depends(get_current_user)):
    if payload.report.get("approval_status") != "approved":
        raise HTTPException(status_code=400, detail="This report must be approved before it can be downloaded")

    try:
        # FIXED: DOCX rendering wrapped to prevent lag
        docx_bytes = await run_in_threadpool(render_docx, payload.report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DOCX export failed: {str(e)}")

    filename = f"{payload.report.get('filename', 'report')}_report.docx"
    return StreamingResponse(
        io.BytesIO(docx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )