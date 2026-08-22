"""Data Scientist Agent (Section 16 of the Phase 4 spec).

Capabilities: statistics, EDA, correlation, distribution/trend analysis,
insight extraction. NOT full AutoML - that's explicitly out of scope for
Phase 4 (see ML Engineer Agent for the boundary). Same data_summary input
as every other agent here; the difference is entirely in what lens the
prompt asks for.
"""

from app.services.consensus import get_consensus_json


def analyze(data_summary: str, task_description: str = "") -> dict:
    focus_block = (
        f"The user's specific request for this analysis is: {task_description}\n"
        "Directly address this request in your summary, key_metrics, and recommendation, "
        "while still only using the exact numbers given below."
        if task_description else ""
    )
    prompt = f"""You are a Data Scientist. Below is a precise statistical summary of a
dataset, computed exactly from the full data (not a guess). Use ONLY these numbers for
any figures you state - never invent or estimate a number that isn't given below.

Focus specifically on STATISTICAL patterns: correlations, distributions, notable
trends, variance, and what the numbers imply as testable hypotheses. Do not write a
business-impact narrative - a Business Analyst agent already covers that; stay
technical and precise.

{focus_block}

{data_summary}

Respond ONLY in this exact JSON format, no extra text:
{{
  "summary": "one paragraph on the statistical patterns found, using exact numbers above",
  "key_metrics": ["statistical finding 1 with an exact number from above", "metric 2", "metric 3"],
  "recommendation": "one concrete next analysis or test worth running"
}}"""

    return get_consensus_json(prompt, temperature=1, max_tokens=2048)
