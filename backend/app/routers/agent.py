"""Intent-aware agent endpoint (Section 31 of the Phase 4 spec).

This is additive: /domain/analyze (routers/domain.py) and the existing "AI
analysis" tab are completely untouched and keep working exactly as before.
/agent/run is the new entry point for the Manager Agent v2, which reasons
over an actual free-text user_query instead of pure domain routing.

Currently synchronous - it runs the full workflow and returns the final
result in one request/response, same as /domain/analyze does today. Async
polling (GET /agent/runs/{workflow_id}) and persistence to Supabase are
later steps (Section 28/31) once the Task Planner/Executor exist and
workflows can genuinely run long enough to need polling.

Document support (Optimization Step 3): _build_initial_state() used to
unconditionally call get_dataset_dataframe(), which raises an
HTTPException for any PDF/DOCX dataset, so document Q&A never worked
through this endpoint. It now branches on is_document_dataset(): documents
go through the same Hybrid Retrieval (services/retrieval) + Evidence Gate
(services/evidence_gate) used by routers/chat.py, and the retrieved
excerpts are handed to the Manager as state.data_summary - every
specialist agent (agents/registry.py) already only ever reads
state.data_summary as plain text, so nothing downstream needs to change to
support documents. Domain Gate also now runs on both paths before any
specialist ever executes, since previously it didn't run here at all - an
off-topic question would run the entire Manager/Planner/Executor pipeline
before failing.
"""

from __future__ import annotations

import asyncio
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agents.domain_gate import check_domain_relevance
from app.agents.events import make_event
from app.agents.manager_v2 import run_manager_v2
from app.agents.state import WorkflowState
from app.deps import get_current_user
from app.services import approvals
from app.services.datasets import (
    build_data_summary,
    get_dataset_dataframe,
    get_dataset_record,
    get_document_text,
    is_document_dataset,
)
from app.services.domain_router import classify_dataset
from app.services.evidence_gate import check_evidence
from app.services.reranker import rerank_chunks
from app.services.retrieval import hybrid_search

router = APIRouter(prefix="/agent", tags=["agent"])

CANDIDATE_CHUNKS = 15
TOP_K_CHUNKS = 5
# Domain Gate only needs enough text to build a vocabulary to compare the
# query against - capping this keeps large documents from slowing down a
# check that's meant to be the cheapest step in the whole pipeline.
MAX_GATE_DOCUMENT_CHARS = 6000


class AgentRunRequest(BaseModel):
    query: str
    dataset_id: str


class RoutingRejected(Exception):
    """Raised by _build_initial_state() when Domain Gate or (for document
    datasets) Evidence Gate rejects the query - caught by the endpoints
    below so a rejection never reaches run_manager_v2() at all."""


def _document_state(payload: AgentRunRequest, user_id: str, query: str) -> WorkflowState:
    document_text = get_document_text(payload.dataset_id, user_id)

    gate = check_domain_relevance(
        query, dataset_columns=[], data_summary=document_text[:MAX_GATE_DOCUMENT_CHARS]
    )
    if not gate["in_domain"]:
        raise RoutingRejected(gate["reason"])

    candidates = hybrid_search(payload.dataset_id, user_id, query, CANDIDATE_CHUNKS)
    chunks = rerank_chunks(query, candidates, TOP_K_CHUNKS)

    evidence = check_evidence(chunks, query=query)
    if not evidence["has_evidence"]:
        raise RoutingRejected(evidence["reason"])

    # Specialists (agents/registry.py) only ever consume state.data_summary
    # as plain text - the retrieved excerpts slot in there directly, so no
    # specialist/executor code needs to change to support documents.
    context = "\n\n".join(
        f"[Excerpt {i + 1}] {chunk.get('chunk_text', '')}" for i, chunk in enumerate(chunks)
    )
    classification = {
        "primary_domain": "Document",
        "secondary_domains": [],
        "tags": [],
        "confidence": round(evidence.get("best_similarity") or 0.5, 2),
        "reasoning": "Document dataset - context comes from hybrid retrieval + reranking, not schema-based domain detection.",
        "agent_domains": ["Document"],
    }

    return WorkflowState(
        dataset_id=payload.dataset_id,
        user_id=user_id,
        data_summary=context,
        classification=classification,
        user_query=query,
        dataset_columns=[],
    )


def _tabular_state(payload: AgentRunRequest, user_id: str, query: str) -> WorkflowState:
    dataframe = get_dataset_dataframe(payload.dataset_id, user_id)
    data_summary = build_data_summary(dataframe)

    gate = check_domain_relevance(
        query, dataset_columns=[str(c) for c in dataframe.columns], data_summary=data_summary
    )
    if not gate["in_domain"]:
        raise RoutingRejected(gate["reason"])

    # Domain detection still runs, but only as context for the Manager's
    # reasoning now - not as the sole basis for agent selection.
    metadata = get_dataset_record(payload.dataset_id, user_id)
    route = classify_dataset(metadata["filename"], dataframe)

    return WorkflowState(
        dataset_id=payload.dataset_id,
        user_id=user_id,
        data_summary=data_summary,
        classification=route.to_dict(),
        user_query=query,
        dataset_columns=[str(c) for c in dataframe.columns],
    )


def _build_initial_state(payload: AgentRunRequest, user_id: str) -> WorkflowState:
    query = payload.query.strip()
    if is_document_dataset(payload.dataset_id, user_id):
        return _document_state(payload, user_id, query)
    return _tabular_state(payload, user_id, query)


def _rejected_result(reason: str) -> dict:
    """Same response shape run_manager_v2() already uses for an
    input-security block, so the frontend handles every rejection reason
    (domain gate, evidence gate, input security) identically."""
    return {
        "workflow_id": str(uuid.uuid4()),
        "status": "rejected",
        "result": {
            "user_query": "",
            "goal": "",
            "error": reason,
            "summary": "",
            "key_metrics": [],
            "recommendation": "",
            "participating_agents": [],
            "specialist_reports": [],
        },
        "approval": None,
    }


def _create_pending_approval(final_state: WorkflowState, user_id: str) -> dict | None:
    """Generic Human Approval hook (Phase 4 roadmap item): every completed
    Multi-Agent run gets a pending approval row, the same way report
    generation already does for reports (see services/report_versions.py).
    Never blocks/fails the main response - approval is a nice-to-have on
    top of a result the user can already see."""
    if final_state.final_output is None:
        return None
    try:
        return approvals.create_version(
            resource_type="agent_workflow",
            resource_id=final_state.workflow_id,
            user_id=user_id,
            content=final_state.final_output,
            dataset_id=final_state.dataset_id,
        )
    except Exception as e:
        print(f"Could not create pending approval for workflow {final_state.workflow_id}: {e}")
        return None


@router.post("/run")
async def run_agent(payload: AgentRunRequest, user=Depends(get_current_user)):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    try:
        state = await run_in_threadpool(_build_initial_state, payload, user.id)
    except RoutingRejected as e:
        return _rejected_result(str(e))

    final_state = await run_manager_v2(state)

    if final_state.final_output is None:
        raise HTTPException(status_code=500, detail="Manager produced no output")

    approval = _create_pending_approval(final_state, user.id)

    return {
        "workflow_id": final_state.workflow_id,
        "status": final_state.status,
        "result": final_state.final_output,
        "approval": approval,
    }


@router.post("/run/stream")
async def run_agent_stream(payload: AgentRunRequest, user=Depends(get_current_user)):
    """Section 30: live workflow progress. Same work as /agent/run, but
    streams a Server-Sent-Events line per stage as it actually happens
    (Manager planning, each task starting/completing/failing/retrying/
    skipping per wave, quality check, final synthesis) instead of making
    the frontend wait for one big response at the end.

    Uses a plain fetch()-readable text/event-stream rather than requiring
    the browser's EventSource API, since EventSource can't send a POST
    body and we need query + dataset_id in the request.
    """
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query is required")

    try:
        state = await run_in_threadpool(_build_initial_state, payload, user.id)
    except RoutingRejected as e:
        # Capture the message as a plain string NOW: Python deletes the
        # exception variable `e` the moment this except block ends, but
        # rejected_stream() is a generator that only actually runs later
        # (when StreamingResponse iterates it) - by then `e` is gone,
        # which raises NameError. Closing over `rejection_reason` instead
        # (a plain str, not tied to the except block's lifetime) avoids that.
        rejection_reason = str(e)

        async def rejected_stream():
            yield f"data: {json.dumps({'type': '__done__', **_rejected_result(rejection_reason)}, default=str)}\n\n"

        return StreamingResponse(
            rejected_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    queue: asyncio.Queue = asyncio.Queue()

    async def emit(event_type: str, agent: str, message: str, data: dict | None = None) -> None:
        await queue.put(make_event(event_type, agent, message, data))

    async def run_and_finish() -> None:
        try:
            final_state = await run_manager_v2(state, emit)
            approval = _create_pending_approval(final_state, state.user_id)
            await queue.put({
                "type": "__done__",
                "workflow_id": final_state.workflow_id,
                "status": final_state.status,
                "result": final_state.final_output,
                "approval": approval,
            })
        except Exception as e:
            await queue.put({"type": "__error__", "message": str(e)})

    async def event_stream():
        task = asyncio.create_task(run_and_finish())
        try:
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event, default=str)}\n\n"
                if event.get("type") in ("__done__", "__error__"):
                    break
        finally:
            await task

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
