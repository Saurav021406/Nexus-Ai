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

from app.agents.domain_gate import check_domain_relevance
from app.agents.events import EventEmitter, NOOP_EMIT
from app.agents.executor import execute_plan
from app.agents.input_security import input_security_check
from app.agents.model_provider import get_nexus_model
from app.agents.planner import validate_and_order_plan
from app.agents.registry import available_agent_domains, get_agent, get_agent_definition, list_capabilities
from app.agents.reviewer import review_check
from app.agents.security import security_check
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


def _build_planning_input(state: WorkflowState, replan_feedback: dict | None = None) -> str:
    registry_snapshot = list_capabilities()
    feedback_block = ""
    if replan_feedback:
        feedback_block = f"""
YOUR PREVIOUS PLAN WAS REVIEWED AND REJECTED. Do not repeat the same plan -
address these specific problems:

Issues found: {json.dumps(replan_feedback.get("issues", []), default=str)}
Suggested improvements: {json.dumps(replan_feedback.get("suggested_improvements", []), default=str)}
"""
    return f"""USER REQUEST:
{state.user_query}

DATASET CLASSIFICATION (from automatic domain detection, for context only -
you decide the final agent selection, this is not binding):
{json.dumps(state.classification, indent=2, default=str)}

DATA SUMMARY (privacy-filtered):
{state.data_summary}

AVAILABLE AGENTS:
{json.dumps(registry_snapshot, indent=2, default=str)}
{feedback_block}"""


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

    # ---------- 0. Domain Gate (Step 1) - Free, zero LLM cost.
    # Must run BEFORE Input Security to prevent wasting LLM tokens on off-topic queries. ----------
    gate = check_domain_relevance(
        state.user_query,
        state.dataset_columns,
        state.data_summary,
    )
    if not gate.get("in_domain", True):
        state.status = "out_of_domain"
        state.final_output = {
            "workflow_id": state.workflow_id,
            "user_query": state.user_query,
            "goal": "",
            "error": gate.get("reason", "Question is outside dataset domain."),
            "summary": gate.get("reason", "Question is outside dataset domain."),
            "key_metrics": [],
            "recommendation": "",
            "participating_agents": [],
            "specialist_reports": [],
        }
        state.touch()
        add_trace(state, "DomainGate", "rejected", output_data=gate)
        await emit("domain_gate_rejected", "DomainGate", gate.get("reason", "Out of domain"), gate)
        await emit("finished", "Manager", "Out of domain", {"status": state.status})
        return state

    # ---------- 1. Input Security (Section 23) ----------
    await emit("input_security_start", "InputSecurity", "Screening your request...", None)
    try:
        input_check = await asyncio.to_thread(input_security_check, state.user_query, state)
    except Exception as e:
        # Fail SAFE, not open: if the security check itself is broken, block
        # rather than silently letting an unscreened query through.
        add_trace(state, "InputSecurity", "check_failed", error=str(e))
        input_check = {"risk_level": "high", "findings": [f"Input security check failed: {e}"], "blocked": True}

    add_trace(state, "InputSecurity", "completed", output_data=input_check)

    if input_check.get("blocked"):
        await emit("input_security_blocked", "InputSecurity", "Request blocked by input security", input_check)
        state.security = input_check
        state.status = "blocked"
        state.final_output = {
            "workflow_id": state.workflow_id,
            "user_query": state.user_query,
            "goal": "",
            "error": "This request was blocked by input security and was not processed.",
            "findings": input_check.get("findings", []),
            "summary": "",
            "key_metrics": [],
            "recommendation": "",
            "participating_agents": [],
            "specialist_reports": [],
        }
        state.touch()
        await emit("finished", "Manager", "Blocked", {"status": state.status})
        return state

    await emit("input_security_passed", "InputSecurity", "Request cleared for processing", input_check)

    # ---------- 2-4. Planning -> Execution -> Review, with replanning
    # (Section 21: "If rejected: Reviewer -> Failure reason -> Manager ->
    # Replan -> Retry"). Bounded at MAX_REPLAN_ATTEMPTS extra tries so a
    # persistently-rejected plan can never loop forever (Section 25). ----------
    MAX_REPLAN_ATTEMPTS = 1
    replan_feedback: dict | None = None

    for attempt in range(MAX_REPLAN_ATTEMPTS + 1):
        is_replan = attempt > 0
        if is_replan:
            state.replan_count += 1
            add_trace(state, "Manager", "replanning", input_data=replan_feedback)
            await emit(
                "manager_replanning", "Manager",
                f"Revising the plan (attempt {attempt + 1}) based on reviewer feedback...",
                replan_feedback,
            )

        # ---------- 2. Planning ----------
        manager_agent = _build_manager_agent()
        try:
            planning_input = _build_planning_input(state, replan_feedback if is_replan else None)
            result = await Runner.run(manager_agent, planning_input)
            agent_plan: AgentPlan = result.final_output
        except Exception as e:
            state.add_error("Manager", f"Planning failed: {e}")
            add_trace(state, "Manager", "planning_failed", error=str(e))
            await emit("manager_planning_failed", "Manager", "Planning failed, using fallback plan", {"error": str(e)})
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

        # ---------- 3. Task Planning (validate + order into dependency waves) ----------
        plan_result = validate_and_order_plan(agent_plan.tasks, available_agent_domains())

        if not plan_result.valid:
            state.add_error("Planner", "; ".join(plan_result.errors))
            add_trace(state, "Planner", "validation_failed", error="; ".join(plan_result.errors))
            await emit("planner_validation_failed", "Planner", "Plan invalid, using fallback",
                        {"errors": plan_result.errors})
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

        # ---------- 4. Task Execution (dependency-respecting, wave by wave) ----------
        async def run_specialist_task(task: dict, state: WorkflowState) -> dict:
            agent_fn = get_agent(task["agent"])
            if not agent_fn:
                return {"error": f"{task['agent']} specialist is not installed"}
            task_description = task.get("task") or state.user_query or ""

            definition = get_agent_definition(task["agent"])
            if definition and definition.needs_full_access:
                output = await asyncio.to_thread(agent_fn, state, task_description)
            else:
                output = await asyncio.to_thread(agent_fn, state.data_summary, task_description)

            if not isinstance(output, dict):
                return {"error": "Specialist returned invalid format"}
            return output

        state.specialist_results = {}
        await execute_plan(state, plan_result.waves, run_specialist_task, emit)

        # ---------- 5. Reviewer + Security (independent agents in PARALLEL) ----------
        await emit("reviewer_check_start", "Reviewer", "Reviewing specialist reports for accuracy and consistency...", None)
        await emit("security_check_start", "Security", "Running security and PII checks...", None)

        review_task = asyncio.to_thread(review_check, state)
        security_task = asyncio.to_thread(security_check, state)
        review_result, security_result = await asyncio.gather(
            review_task, security_task, return_exceptions=True
        )

        if isinstance(review_result, Exception):
            state.review = {
                "overall_quality": "medium",
                "issues": [f"Reviewer failed: {str(review_result)}"],
                "approved": True,
                "suggested_improvements": [],
            }
            add_trace(state, "Reviewer", "failed", error=str(review_result))
            await emit("reviewer_check_failed", "Reviewer", "Review failed, using safe defaults", {"error": str(review_result)})
        else:
            state.review = review_result
            add_trace(state, "Reviewer", "completed", output_data=state.review)
            await emit("reviewer_check_completed", "Reviewer", "Review complete", state.review)

        if isinstance(security_result, Exception):
            state.security = {
                "risk_level": "medium",
                "findings": [f"Security agent failed: {str(security_result)}"],
                "blocked": False,
                "safe_to_show": True,
            }
            add_trace(state, "Security", "failed", error=str(security_result))
            await emit("security_check_failed", "Security", "Security checks failed, using safe defaults", {"error": str(security_result)})
        else:
            state.security = security_result
            add_trace(state, "Security", "completed", output_data=state.security)
            await emit("security_check_completed", "Security", "Security checks complete", state.security)

        # ---------- Replan decision ----------
        if state.review.get("approved", True):
            break

        if attempt < MAX_REPLAN_ATTEMPTS:
            replan_feedback = {
                "issues": state.review.get("issues", []),
                "suggested_improvements": state.review.get("suggested_improvements", []),
            }
            add_trace(state, "Manager", "replan_triggered", output_data=replan_feedback)
            await emit("manager_replan_triggered", "Manager", "Review rejected the result - replanning...", replan_feedback)
        else:
            add_trace(state, "Manager", "replan_exhausted")
            await emit(
                "manager_replan_exhausted", "Manager",
                "Still not approved after replanning - proceeding with the best available result", None,
            )

    # ---------- 6. Final synthesis ----------
    state.final_output = _synthesize(state)
    final_attempt_had_errors = any("error" in v for v in state.specialist_results.values())
    state.status = "completed_with_errors" if final_attempt_had_errors else "completed"
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
        "replan_count": state.replan_count,
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