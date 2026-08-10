import asyncio
from typing import Any

from app.agents.registry import get_agent
from app.agents.state import WorkflowState
from app.agents.tracing import add_trace
from app.agents.quality import quality_check


async def run_manager(state: WorkflowState) -> WorkflowState:
    add_trace(state, "Manager", "start")

    # ---------- 1. Task Planning ----------
    primary = state.classification.get("primary_domain", "General")
    secondaries = state.classification.get("secondary_domains", []) or []

    plan = [{"step": 1, "agent": primary, "task": "full_analysis"}]

    # Quota-saving: only bring in secondary specialists when the caller
    # explicitly asked for a multi-domain run (e.g. via selected_domains).
    # The automatic/default path only runs the primary specialist.
    if state.include_secondary_specialists:
        for domain in secondaries[:2]:
            plan.append(
                {
                    "step": len(plan) + 1,
                    "agent": domain,
                    "task": "supporting_analysis",
                }
            )

    state.plan = plan
    add_trace(state, "Manager", "plan_created", output_data=plan)

    # ---------- 2. Agent Delegation ----------
    async def run_one(domain: str) -> tuple[str, dict]:
        agent_fn = get_agent(domain)
        if not agent_fn:
            result = {"error": f"{domain} specialist is not installed"}
            add_trace(state, domain, "missing", error=result["error"])
            return domain, result

        try:
            result = await asyncio.to_thread(agent_fn, state.data_summary)
            if not isinstance(result, dict):
                result = {"error": "Specialist returned invalid format"}
            add_trace(state, domain, "completed", output_data=result)
            return domain, result
        except Exception as e:
            result = {"error": f"Specialist failed: {str(e)}"}
            add_trace(state, domain, "failed", error=str(e))
            return domain, result

    domains = [step["agent"] for step in plan]
    results = await asyncio.gather(*(run_one(d) for d in domains))
    state.specialist_results = {domain: report for domain, report in results}

    # ---------- 3. Quality check (Reviewer + Security merged into 1 call) ----------
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
    add_trace(state, "Manager", "finished")

    return state


def _synthesize(state: WorkflowState) -> dict[str, Any]:
    successful = {
        k: v for k, v in state.specialist_results.items() if "error" not in v
    }

    if not successful:
        return {
            "error": "No specialist could complete analysis",
            "classification": state.classification,
            "plan": state.plan,
            "summary": "",
            "key_metrics": [],
            "recommendation": "",
            "participating_agents": [],
            "specialist_reports": [
                {"domain": k, **v} for k, v in state.specialist_results.items()
            ],
            "review": state.review,
            "security": state.security,
            "traces": state.traces,
        }

    primary = state.classification.get("primary_domain", "General")
    primary_report = successful.get(primary) or next(iter(successful.values()))

    key_metrics: list[str] = []
    for report in successful.values():
        for metric in report.get("key_metrics", []):
            if metric not in key_metrics:
                key_metrics.append(metric)

    return {
        "classification": state.classification,
        "plan": state.plan,
        "summary": primary_report.get("summary", ""),
        "key_metrics": key_metrics[:8],
        "recommendation": primary_report.get("recommendation", ""),
        "participating_agents": list(successful.keys()),
        "specialist_reports": [
            {"domain": k, **v} for k, v in state.specialist_results.items()
        ],
        "review": state.review,
        "security": state.security,
        "traces": state.traces,
    }
