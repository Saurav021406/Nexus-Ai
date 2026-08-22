"""ML Engineer Agent (Section 16 of the Phase 4 spec).

"Phase 4 only needs the orchestration interface... Do NOT implement the
complete Phase 5 AutoML engine here." This agent reasons about whether the
dataset LOOKS ready for modeling (target variable candidates, feature
readiness, likely modeling approach) - it does not train anything.
Real forecasting already exists and works (services powering /forecast);
this agent is the reasoning layer that would eventually decide *when* to
invoke that, not a replacement for it.
"""

from app.services.consensus import get_consensus_json


def analyze(data_summary: str, task_description: str = "") -> dict:
    focus_block = (
        f"The user's specific request for this analysis is: {task_description}\n"
        "Directly address this request in your summary, key_metrics, and recommendation, "
        "while still only using the exact numbers given below."
        if task_description else ""
    )
    prompt = f"""You are an ML Engineer. Below is a precise statistical summary of a
dataset, computed exactly from the full data (not a guess). Use ONLY these numbers for
any figures you state - never invent or estimate a number that isn't given below.

Focus specifically on MODELING READINESS: which column(s) look like plausible
prediction targets, whether there's enough signal/history for forecasting, what
features look useful, and what modeling approach (e.g. regression, classification,
time-series) would fit. Do NOT actually build or describe training a specific model
in detail - you are assessing readiness and recommending an approach, not modeling.

{focus_block}

{data_summary}

Respond ONLY in this exact JSON format, no extra text:
{{
  "summary": "one paragraph on modeling readiness and a suitable approach, using exact numbers above",
  "key_metrics": ["readiness signal 1 with an exact number from above", "metric 2", "metric 3"],
  "recommendation": "one concrete next step toward a model (e.g. which target/features to use)"
}}"""

    return get_consensus_json(prompt, temperature=1, max_tokens=2048)
