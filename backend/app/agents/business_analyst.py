"""Business Analyst Agent (Section 16 of the Phase 4 spec).

Capabilities: interpret technical results, translate into business meaning,
KPI analysis, recommendations, executive summaries. The spec's own example
of what this agent should NOT do: "Correlation = -0.63." What it SHOULD do:
"Discounts have a strong negative relationship with profit in this dataset,
suggesting discounting may be eroding margins." That's the bar this prompt
is written against.
"""

from app.services.consensus import get_consensus_json


def analyze(data_summary: str, task_description: str = "") -> dict:
    focus_block = (
        f"The user's specific request for this analysis is: {task_description}\n"
        "Directly address this request in your summary, key_metrics, and recommendation, "
        "while still only using the exact numbers given below."
        if task_description else ""
    )
    prompt = f"""You are a Business Analyst. Below is a precise statistical summary of a
dataset, computed exactly from the full data (not a guess). Use ONLY these numbers for
any figures you state - never invent or estimate a number that isn't given below.

Translate the data into BUSINESS meaning. Never state a bare statistic without saying
what it implies for the business. For example, do not write "Correlation = -0.63" -
write "Discounts have a strong negative relationship with profit, suggesting
discounting may be eroding margins." Focus on KPIs, business impact, and what a
decision-maker should actually do about this.

{focus_block}

{data_summary}

Respond ONLY in this exact JSON format, no extra text:
{{
  "summary": "one paragraph explaining the business implications, using exact numbers above",
  "key_metrics": ["business-framed metric 1 with an exact number from above", "metric 2", "metric 3"],
  "recommendation": "one concrete business action a decision-maker should take"
}}"""

    return get_consensus_json(prompt, temperature=1, max_tokens=2048)
