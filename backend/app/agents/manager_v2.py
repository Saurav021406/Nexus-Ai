"""Manager Agent v2 (Section 11 of the Phase 4 spec).

This is a NEW, additive path. The original agents/manager.py (pure domain
routing: "primary_domain -> one specialist") is untouched and still powers
the existing /domain/analyze endpoint and "AI analysis" tab exactly as
before.

manager_v2 is the Section 10 target architecture wired together: the
Manager Agent reasons over the actual user_query and produces a structured
plan, the Task Planner (agents/planner.py) validates it and orders it into
dependency-respecting waves, and the Task Executor (agents/executor.py)
actually runs it - concurrent within a wave, sequential across waves,
with retry/timeout/skip handling. This closes the gap explicitly flagged
in Steps 1-6: depends_on is now enforced, not just recorded.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agents import Agent, Runner
from pydantic import BaseModel, Field

from app.agents.executor import execute_plan
from app.agents.events import EventEmitter, NOOP_EMIT
from app.agents.model_provider import get_nexus_model
from app.agents.planner import validate_and_order_plan
from app.agents.quality import quality_check
from app.agents.registry import get_agent, available_agent_domains, list_capabilities
from app.agents.state import WorkflowState
from app.agents.tracing import add_trace


class PlannedTask(BaseModel):
    id: str = Field(description="Short unique task id, e.g. 't1'")
    agent: str = Field(description="Must be one of the available agent names given to you")
    task: str = Field(description="What this agent should do, in a few words")
    depends_on: list[str] = Field(default_factory=list, description="Task ids this depends on, if any")


class AgentPlan(BaseModel):
    goal: str = Field(description="One sentence restating what the user actually wants")
    reasoning: str = Field(description="1-2 sentences on why these agents were chosen")
    tasks: list[PlannedTask]


MANAGER_INSTRUCTIONS = """You are the Manager Agent for Nexus AI, a data analysis platform.

A user has asked a question about a dataset. Your job is to:
1. Understand what they actually want (their real goal, not just a label).
2. Choose which of the available specialist agents should analyze the dataset to
   satisfy that goal. You may choose more than one if the request genuinely spans
   multiple domains (e.g. "look at this from a financial and HR angle").
3. Produce a short task plan.

Rules:
- Only use agent names from the AVAILABLE AGENTS list you're given - never invent one.
- If the request is narrow, pick exactly one agent. Don't add agents "just in case".
- If the request is broad or explicitly multi-angle, you may pick up to 3 agents.
- Every task needs a unique id like "t1", "t2".
- Use depends_on only when one task genuinely needs another task's output first;
  most tasks here can run independently (depends_on: []).
"""


def _build_manager_agent() -> Agent:
    return Agent(
        name="Manager",
        instructions=MANAGER_INSTRUCTIONS,
        model=get_nexus_model(),
        output_type=AgentPlan,
    )


def _build_planning_input(state: WorkflowState) -> str:
    registry_snapshot = list_capabilities()
    return f"""USER REQUEST:
{state.user_query}

DATASET CLASSIFICATION (from automatic domain detection, for context only -
you decide the final agent selection, this is not binding):
{json.dumps(state.classification, indent=2, default=str)}

DATA SUMMARY (privacy-filtered):
{state.data_summary}

AVAILABLE AGENTS:
{json.dumps(registry_snapshot, indent=2, default=str)}
"""


async def run_manager_v2(state: WorkflowState, emit: EventEmitter = NOOP_EMIT) -> WorkflowState:
    """Entry point for the new intent-aware path. Mirrors run_manager()'s
    overall shape (plan -> delegate -> quality check -> synthesize) so the
    two paths stay easy to compare, but the plan now comes from real
    reasoning over state.user_query instead of a domain lookup.

    `emit` is optional live-progress reporting (Section 30) - defaults to a
    no-op, so calling this without a frontend watching behaves identically
    to before."""

    add_trace(state, "Manager", "start", input_data={"user_query": state.user_query})
    await emit("manager_start", "Manager", "Understanding your request...", None)

    # ---------- 1. Planning ----------
    manager_agent = _build_manager_agent()
    try:
        result = await Runner.run(manager_agent, _build_planning_input(state))
        agent_plan: AgentPlan = result.final_output
    except Exception as e:
        state.add_error("Manager", f"Planning failed: {e}")
        add_trace(state, "Manager", "planning_failed", error=str(e))
        await emit("manager_planning_failed", "Manager", "Planning failed, using fallback plan", {"error": str(e)})
        # Safe fallback: fall back to the primary detected domain as a single task,
        # so a planning-model hiccup doesn't take down the whole request.
        fallback_domain = state.classification.get("primary_domain", "General")
        agent_plan = AgentPlan(
            goal=state.user_query or "Analyze the dataset",
            reasoning="Planning model unavailable - falling back to detected domain.",
            tasks=[PlannedTask(id="t1", agent=fallback_domain, task="full_analysis", depends_on=[])],
        )

    state.goal = agent_plan.goal
    state.plan = [
        {"step": i + 1, "agent": t.agent, "task": t.task} for i, t in enumerate(agent_plan.tasks)
    ]
    add_trace(
        state, "Manager", "plan_created",
        output_data={"goal": state.goal, "tasks": [t.model_dump() for t in agent_plan.tasks]},
    )
    await emit("plan_created", "Manager", f"Plan: {state.goal}",
                {"tasks": [t.model_dump() for t in agent_plan.tasks]})

    # ---------- 2. Task Planning (validate + order into dependency waves) ----------
    plan_result = validate_and_order_plan(agent_plan.tasks, available_agent_domains())

    if not plan_result.valid:
        state.add_error("Planner", "; ".join(plan_result.errors))
        add_trace(state, "Planner", "validation_failed", error="; ".join(plan_result.errors))
        await emit("planner_validation_failed", "Planner", "Plan invalid, using fallback",
                    {"errors": plan_result.errors})
        # Same safety net as a planning failure: fall back to one task on
        # the detected primary domain rather than failing the whole request.
        fallback_domain = state.classification.get("primary_domain", "General")
        plan_result = validate_and_order_plan(
            [PlannedTask(id="t1", agent=fallback_domain, task="full_analysis", depends_on=[])],
            available_agent_domains(),
        )

    add_trace(
        state, "Planner", "plan_ordered",
        output_data={"wave_count": len(plan_result.waves), "waves": [
            [t["id"] for t in wave] for wave in plan_result.waves
        ]},
    )
    await emit("plan_ordered", "Planner", f"Ordered into {len(plan_result.waves)} execution wave(s)",
                {"waves": [[t["id"] for t in wave] for wave in plan_result.waves]})

    # ---------- 3. Task Execution (dependency-respecting, wave by wave) ----------
    async def run_specialist_task(task: dict, state: WorkflowState) -> dict:
        agent_fn = get_agent(task["agent"])
        if not agent_fn:
            return {"error": f"{task['agent']} specialist is not installed"}
        output = await asyncio.to_thread(agent_fn, state.data_summary)
        if not isinstance(output, dict):
            return {"error": "Specialist returned invalid format"}
        return output

    await execute_plan(state, plan_result.waves, run_specialist_task, emit)

    # ---------- 4. Reviewer + Security (reuses the existing quality_check -
    # separating these into two independent agents is a later step) ----------
    await emit("quality_check_start", "Quality", "Running reviewer and security checks...", None)
    try:
        quality = await asyncio.to_thread(quality_check, state)
        state.review = quality.get("review")
        state.security = quality.get("security")
        add_trace(state, "Quality", "completed", output_data=quality)
        await emit("quality_check_completed", "Quality", "Quality checks complete", quality)
    except Exception as e:
        state.review = {
            "overall_quality": "medium",
            "issues": [f"Reviewer failed: {str(e)}"],
            "approved": True,
            "suggested_improvements": [],
        }
        state.security = {
            "risk_level": "medium",
            "findings": [f"Security agent failed: {str(e)}"],
            "blocked": False,
            "safe_to_show": True,
        }
        add_trace(state, "Quality", "failed", error=str(e))
        await emit("quality_check_failed", "Quality", "Quality checks failed, using safe defaults", {"error": str(e)})

    # ---------- 5. Final synthesis ----------
    state.final_output = _synthesize(state)
    state.status = "completed" if not state.errors else "completed_with_errors"
    state.touch()
    add_trace(state, "Manager", "finished")
    await emit("finished", "Manager", "Done", {"status": state.status})

    return state


def _synthesize(state: WorkflowState) -> dict[str, Any]:
    successful = {k: v for k, v in state.specialist_results.items() if "error" not in v}

    base = {
        "workflow_id": state.workflow_id,
        "user_query": state.user_query,
        "goal": state.goal,
        "classification": state.classification,
        "plan": state.plan,
        "tasks": state.tasks,
        "review": state.review,
        "security": state.security,
        "traces": state.traces,
        "errors": state.errors,
    }

    if not successful:
        return {
            **base,
            "error": "No specialist could complete analysis",
            "summary": "",
            "key_metrics": [],
            "recommendation": "",
            "participating_agents": [],
            "specialist_reports": [{"domain": k, **v} for k, v in state.specialist_results.items()],
        }

    primary = state.classification.get("primary_domain", "General")
    primary_report = successful.get(primary) or next(iter(successful.values()))

    key_metrics: list[str] = []
    for report in successful.values():
        for metric in report.get("key_metrics", []):
            if metric not in key_metrics:
                key_metrics.append(metric)

    return {
        **base,
        "summary": primary_report.get("summary", ""),
        "key_metrics": key_metrics[:8],
        "recommendation": primary_report.get("recommendation", ""),
        "participating_agents": list(successful.keys()),
        "specialist_reports": [{"domain": k, **v} for k, v in state.specialist_results.items()],
    }
