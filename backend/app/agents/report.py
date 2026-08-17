"""Report Agent.

Takes the Manager's final_output (classification, specialist reports, review,
security, plan, traces) and produces structured, human-readable report content:
  - an executive summary narrative
  - an "agent collaboration" narrative explaining how the specialists, the
    reviewer, and the security agent arrived at the final recommendation
  - a deterministic step-by-step timeline built from plan/traces (no LLM,
    since that data already exists exactly as it happened)
  - per-specialist sections

Both narratives come from a single call to the Consensus Engine (Groq +
NVIDIA Nemotron + Claude queried in parallel, merged with agreement/
contradiction detection - see app/services/consensus.py). Both are
strictly grounded in the analysis JSON - the prompt forbids inventing new
numbers or claims.

The output of this agent is plain JSON, not a file. Rendering to PDF/DOCX is
a separate step (see services/report_render.py), and the frontend lets a
human edit the executive summary before either export happens.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from app.services.consensus import get_consensus_json

# reportlab's default PDF font (Helvetica/WinAnsi) can't render some "smart"
# Unicode punctuation that LLMs like to use (en/em dashes, curly quotes,
# ellipsis) and shows a black box instead. Normalize to plain ASCII once
# here so both the PDF and DOCX exports are clean.
_UNICODE_REPLACEMENTS = {
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2011": "-",  # non-breaking hyphen
    "\u2212": "-",  # minus sign
    "\u2018": "'",  # left single quote
    "\u2019": "'",  # right single quote
    "\u201c": '"',  # left double quote
    "\u201d": '"',  # right double quote
    "\u2026": "...",  # ellipsis
    "\u00a0": " ",  # non-breaking space
    "\u2022": "-",  # bullet
}


def _sanitize(value: Any) -> Any:
    """Recursively clean smart-punctuation characters from strings, lists, and dicts."""
    if isinstance(value, str):
        for bad, good in _UNICODE_REPLACEMENTS.items():
            value = value.replace(bad, good)
        return value
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    return value


def _build_sections(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    sections = []
    for report in analysis.get("specialist_reports", []):
        if "error" in report:
            continue
        sections.append(
            {
                "heading": f"{report.get('domain', 'Specialist')} Analysis",
                "body": report.get("summary", ""),
                "bullets": report.get("key_metrics", []),
                "recommendation": report.get("recommendation", ""),
            }
        )
    return sections


def _build_agent_timeline(analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Deterministic, no-LLM timeline of what each agent actually did.

    Built from plan + specialist_reports + review + security, which are the
    real outputs of the run - not a re-telling, the actual record.
    """
    timeline: list[dict[str, Any]] = []
    classification = analysis.get("classification", {})

    timeline.append(
        {
            "agent": "Manager Agent",
            "role": "Domain detection & planning",
            "detail": (
                f"Classified the dataset as {classification.get('primary_domain', 'General')} "
                f"(confidence {classification.get('confidence', 'n/a')}) and built a "
                f"{len(analysis.get('plan', []))}-step analysis plan."
            ),
        }
    )

    for step in analysis.get("plan", []):
        agent_name = step.get("agent", "Specialist")
        report = next(
            (r for r in analysis.get("specialist_reports", []) if r.get("domain") == agent_name),
            None,
        )
        if report and "error" not in report:
            detail = report.get("summary", "Completed analysis.")
        elif report:
            detail = f"Failed: {report.get('error')}"
        else:
            detail = "Did not complete."
        timeline.append(
            {
                "agent": f"{agent_name} Specialist",
                "role": step.get("task", "analysis"),
                "detail": detail,
            }
        )

    review = analysis.get("review")
    if review:
        timeline.append(
            {
                "agent": "Reviewer Agent",
                "role": "Quality check",
                "detail": (
                    f"Rated overall quality '{review.get('overall_quality', 'n/a')}', "
                    f"{'approved' if review.get('approved') else 'flagged for revision'}."
                    + (f" Issues noted: {', '.join(review.get('issues', []))}." if review.get("issues") else "")
                ),
            }
        )

    security = analysis.get("security")
    if security:
        timeline.append(
            {
                "agent": "Security Agent",
                "role": "Safety & privacy check",
                "detail": (
                    f"Assessed risk as '{security.get('risk_level', 'n/a')}', "
                    f"{'cleared for display' if security.get('safe_to_show') else 'blocked from display'}."
                ),
            }
        )

    timeline.append(
        {
            "agent": "Manager Agent",
            "role": "Final synthesis",
            "detail": "Combined specialist findings into the final recommendation below.",
        }
    )

    return timeline


def _report_prompt(analysis: dict[str, Any], filename: str) -> str:
    return f"""You are the Report Agent for a data analysis platform. You are given the
complete output of a multi-agent analysis run (a Manager agent, one or more domain
specialists, a Reviewer agent, and a Security agent) for the dataset "{filename}".

Write TWO short pieces of text, using ONLY the facts in the JSON below - never invent a
number, trend, or claim that is not already present in this JSON:

1. "executive_summary": 3-5 sentences, plain text, no markdown, summarizing the overall
   finding and recommendation for someone who will only read this paragraph.
2. "agent_collaboration_narrative": 3-5 sentences, plain text, explaining IN PLAIN
   ENGLISH how the different agents worked together to reach this recommendation -
   e.g. what the Manager decided to delegate, what each specialist contributed, and
   what the Reviewer/Security agents checked before this was approved. Write it as an
   explanation for a non-technical reader, not a log dump.

ANALYSIS:
{json.dumps(analysis, indent=2, default=str)}

Respond ONLY with this exact JSON shape (no markdown, no extra text):
{{"executive_summary": "...", "agent_collaboration_narrative": "..."}}"""


def _call_llm(prompt: str) -> dict | None:
    try:
        return get_consensus_json(prompt, temperature=0.7, max_tokens=768)
    except (RuntimeError, json.JSONDecodeError) as e:
        print(f"Consensus engine failed in report agent: {e}")
        return None


def generate_report_content(
    analysis: dict[str, Any],
    filename: str,
    dataset_id: str,
    charts: list[dict] | None = None,
) -> dict[str, Any]:
    """Build the full report content JSON from a Manager final_output payload."""

    narratives = _call_llm(_report_prompt(analysis, filename)) or {}

    executive_summary = narratives.get("executive_summary") or (
        analysis.get("summary") or analysis.get("recommendation") or
        "Analysis completed. See the sections below for specialist findings."
    )
    agent_collaboration_narrative = narratives.get("agent_collaboration_narrative") or (
        "The Manager agent routed this dataset to the specialists listed below, "
        "then the Reviewer and Security agents checked their output before it was "
        "approved for display."
    )

    classification = analysis.get("classification", {})
    primary_domain = classification.get("primary_domain", "General")

    report = {
        "title": "Nexus AI Analysis Report",
        "subtitle": filename,
        "dataset_id": dataset_id,
        "filename": filename,
        "primary_domain": primary_domain,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "executive_summary": executive_summary,
        "agent_collaboration_narrative": agent_collaboration_narrative,
        "agent_timeline": _build_agent_timeline(analysis),
        "key_metrics": analysis.get("key_metrics", []),
        "recommendation": analysis.get("recommendation", ""),
        "sections": _build_sections(analysis),
        "charts": charts or [],
        "participating_agents": analysis.get("participating_agents", []),
        "review": analysis.get("review"),
        "security": analysis.get("security"),
        # Approval state - defaults to pending; a human can edit the executive
        # summary and approve before either export happens.
        "approval_status": "pending",
    }
    return _sanitize(report)