"""Manager Agent v2 (Section 11 of the Phase 4 spec).

This is a NEW, additive path. The original agents/manager.py (pure domain
routing: "primary_domain -> one specialist") is untouched and still powers
the existing /domain/analyze endpoint and "AI analysis" tab exactly as
before.

manager_v2 is the first step toward Section 10's target architecture: a
Manager that reasons over the actual user_query (not just a pre-computed
domain label) and produces a structured, dependency-aware plan (Section 11's
JSON shape), then still uses the existing domain specialists to execute it.

Scope note (Steps 1-6 of the spec's Section 41 implementation order): this
module builds the Manager Agent itself. It does NOT yet include:
  - a standalone Task Planner module (Section 13 / Step 7)
  - a real dependency-respecting Task Executor (Section 14 / Step 8)
Task execution below runs every planned task concurrently, same as the
original manager.py. The plan's `depends_on` metadata is captured and
stored in state.tasks so the future Task Executor can consume it - but
nothing enforces it yet. This is called out explicitly rather than silently
half-implementing dependency ordering.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from agents import Agent, Runner
from pydantic import BaseModel, Field

from app.agents.model_provider import get_nexus_model
from app.agents.quality import quality_check
from app.agents.registry import get_agent, list_capabilities
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


async def run_manager_v2(state: WorkflowState) -> WorkflowState:
    """Entry point for the new intent-aware path. Mirrors run_manager()'s
    overall shape (plan -> delegate -> quality check -> synthesize) so the
    two paths stay easy to compare, but the plan now comes from real
    reasoning over state.user_query instead of a domain lookup."""

    add_trace(state, "Manager", "start", input_data={"user_query": state.user_query})

    # ---------- 1. Planning ----------
    manager_agent = _build_manager_agent()
    try:
        result = await Runner.run(manager_agent, _build_planning_input(state))
        agent_plan: AgentPlan = result.final_output
    except Exception as e:
        state.add_error("Manager", f"Planning failed: {e}")
        add_trace(state, "Manager", "planning_failed", error=str(e))
        # Safe fallback: fall back to the primary detected domain as a single task,
        # so a planning-model hiccup doesn't take down the whole request.
        fallback_domain = state.classification.get("primary_domain", "General")
        agent_plan = AgentPlan(
            goal=state.user_query or "Analyze the dataset",
            reasoning="Planning model unavailable - falling back to detected domain.",
            tasks=[PlannedTask(id="t1", agent=fallback_domain, task="full_analysis", depends_on=[])],
        )

    state.goal = agent_plan.goal
    state.tasks = [t.model_dump() for t in agent_plan.tasks]
    state.plan = [
        {"step": i + 1, "agent": t.agent, "task": t.task} for i, t in enumerate(agent_plan.tasks)
    ]
    add_trace(state, "Manager", "plan_created", output_data={"goal": state.goal, "tasks": state.tasks})

    # ---------- 2. Delegation ----------
    # NOTE: runs every task concurrently regardless of depends_on - see module
    # docstring. A real dependency-respecting executor is the next step.
    async def run_one(task: PlannedTask) -> tuple[str, dict]:
        agent_fn = get_agent(task.agent)
        if not agent_fn:
            result = {"error": f"{task.agent} specialist is not installed"}
            add_trace(state, task.agent, "missing", error=result["error"])
            return task.agent, result
        try:
            output = await asyncio.to_thread(agent_fn, state.data_summary)
            if not isinstance(output, dict):
                output = {"error": "Specialist returned invalid format"}
            add_trace(state, task.agent, "completed", output_data=output)
            return task.agent, output
        except Exception as e:
            output = {"error": f"Specialist failed: {str(e)}"}
            add_trace(state, task.agent, "failed", error=str(e))
            return task.agent, output

    if not agent_plan.tasks:
        state.add_error("Manager", "Plan produced no tasks")
    else:
        results = await asyncio.gather(*(run_one(t) for t in agent_plan.tasks))
        state.specialist_results = {domain: report for domain, report in results}

        for task in state.tasks:
            domain = task["agent"]
            task["status"] = "failed" if "error" in state.specialist_results.get(domain, {}) else "completed"

    # ---------- 3. Reviewer + Security (reuses the existing quality_check -
    # separating these into two independent agents is a later step) ----------
    try:
        quality = await asyncio.to_thread(quality_check, state)
        state.review = quality.get("review")
        state.security = quality.get("security")
        add_trace(state, "Quality", "completed", output_data=quality)
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

    # ---------- 4. Final synthesis ----------
    state.final_output = _synthesize(state)
    state.status = "completed" if not state.errors else "completed_with_errors"
    state.touch()
    add_trace(state, "Manager", "finished")

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
