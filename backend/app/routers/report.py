"""Report Agent endpoints.

Flow: the frontend already has the Manager's final_output (from
POST /domain/analyze) sitting in state. It posts that here once to get back
structured report content (the "draft"), which can be shown to the user for
approval/edits, then exported to PDF and/or DOCX without re-running the LLM.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io

from app.agents.report import generate_report_content
from app.deps import get_current_user
from app.services.datasets import get_dataset_dataframe
from app.services.report_charts import generate_report_charts
from app.services.report_render import render_docx, render_pdf

router = APIRouter(prefix="/report", tags=["report"])


class GenerateReportRequest(BaseModel):
    dataset_id: str
    filename: str
    analysis: dict[str, Any]


class ExportReportRequest(BaseModel):
    report: dict[str, Any]


@router.post("/generate")
async def generate_report(payload: GenerateReportRequest, user=Depends(get_current_user)):
    if not payload.analysis:
        raise HTTPException(status_code=400, detail="analysis payload is required")

    try:
        dataframe = get_dataset_dataframe(payload.dataset_id, user.id)
        charts = generate_report_charts(dataframe)
    except Exception as e:
        # Charts are a nice-to-have, never block report generation on them.
        print(f"Chart generation skipped: {e}")
        charts = []

    report = generate_report_content(
        analysis=payload.analysis,
        filename=payload.filename,
        dataset_id=payload.dataset_id,
        charts=charts,
    )
    return report


@router.post("/pdf")
async def export_pdf(payload: ExportReportRequest, user=Depends(get_current_user)):
    try:
        pdf_bytes = render_pdf(payload.report)
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
    try:
        docx_bytes = render_docx(payload.report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DOCX export failed: {str(e)}")

    filename = f"{payload.report.get('filename', 'report')}_report.docx"
    return StreamingResponse(
        io.BytesIO(docx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
