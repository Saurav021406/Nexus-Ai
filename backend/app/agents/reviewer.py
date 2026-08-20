"""Reviewer Agent (Phase 4 roadmap: "Separate Reviewer & Security agents",
split out of the old fused agents/quality.py::quality_check).

Checks specialist outputs for internal consistency and groundedness only.
Security concerns (PII, unsafe advice, prompt injection) are
agents/security.py's job now, not this one's - keeping the two independent
means a security finding can never get diluted/missed inside a combined
"looks fine overall" verdict, and each shows as its own step in live
progress instead of one fused "Quality" row.
"""

import json

from app.agents.state import WorkflowState
from app.services.consensus import get_consensus_json


def review_check(state: WorkflowState) -> dict:
    prompt = f"""You are a strict Reviewer Agent for a data analysis platform.
Check the specialist reports below against the data summary they were built from.

REVIEWER checks:
- Are numbers only taken from the data summary? (no invented figures)
- Are the specialists consistent with each other?
- Are recommendations concrete and useful?
- Any contradictions or vague statements?

DATA SUMMARY (already privacy-filtered):
{state.data_summary}

SPECIALIST REPORTS:
{json.dumps(state.specialist_results, indent=2, default=str)}

Respond ONLY with this exact JSON shape (no markdown, no extra text):
{{
  "overall_quality": "high",
  "issues": [],
  "approved": true,
  "suggested_improvements": []
}}
"""
    return get_consensus_json(prompt, temperature=1, max_tokens=1024)
