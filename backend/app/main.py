import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import health
from app.routers import upload
from app.routers import domain
from app.routers import chat
from app.routers import forecast
from app.routers import cleaning
from app.routers import eda
from app.routers import viz
from app.routers import report
from app.routers import agent
from app.routers import approvals
from app.routers import automation
from app.routers import automl
from app.routers import stats
from app.routers import root_cause
from app.routers import scenario
from app.routers import recommendations
from app.routers import meeting_intelligence
from app.routers import email_generator
from app.routers import monitoring
from app.routers import audit_logs
from app.services.audit_logs import record_audit_event
from app.services.monitoring import monitoring_loop


@asynccontextmanager
async def lifespan(app: FastAPI):
    monitor_task = asyncio.create_task(monitoring_loop())
    try:
        yield
    finally:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Nexus AI Backend", version="0.1.0", lifespan=lifespan)


async def _write_audit_event(request: Request, status_code: int, duration_ms: int) -> None:
    user_id = getattr(request.state, "user_id", None)
    if not user_id or request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
        return

    route = request.scope.get("route")
    endpoint = getattr(route, "path", request.url.path)
    action = f"{request.method} {endpoint}"
    await run_in_threadpool(
        record_audit_event,
        user_id,
        action,
        endpoint,
        status_code,
        duration_ms,
    )


@app.middleware("http")
async def audit_authenticated_mutations(request: Request, call_next):
    started_at = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        await _write_audit_event(request, 500, duration_ms)
        raise

    duration_ms = int((time.perf_counter() - started_at) * 1000)
    await _write_audit_event(request, response.status_code, duration_ms)
    return response


app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(domain.router)
app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(forecast.router)
app.include_router(cleaning.router)
app.include_router(eda.router)
app.include_router(viz.router)
app.include_router(report.router)
app.include_router(agent.router)
app.include_router(approvals.router)
app.include_router(automation.router)
app.include_router(automl.router)
app.include_router(stats.router)
app.include_router(root_cause.router)
app.include_router(scenario.router)
app.include_router(recommendations.router)
app.include_router(meeting_intelligence.router)
app.include_router(email_generator.router)
app.include_router(monitoring.router)
app.include_router(audit_logs.router)


@app.get("/")
async def root():
    return {"message": "Nexus AI backend is running. See /docs for API docs."}
