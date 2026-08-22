"""Visualization Agent (Section 16 of the Phase 4 spec).

"Decide which chart is appropriate... Use existing Phase 3 visualization
functionality. Do not rebuild charts from scratch." This agent reasons
about WHICH chart(s) would best represent the data and returns a structured
recommendation - it does not render anything itself. Actual chart
rendering already exists and works (services/visualization.py for the
interactive dashboard, services/report_charts.py for static report
exports); this agent is a reasoning layer on top, not a third
implementation of chart drawing.
"""

from app.services.consensus import get_consensus_json


def analyze(data_summary: str, task_description: str = "") -> dict:
    focus_block = (
        f"The user's specific request for this analysis is: {task_description}\n"
        "Directly address this request in your summary, key_metrics, and recommendation, "
        "while still only using the exact numbers given below."
        if task_description else ""
    )
    prompt = f"""You are a Visualization Agent. Below is a precise statistical summary of
a dataset, computed exactly from the full data (not a guess). Use ONLY these numbers
for any figures you state - never invent or estimate a number that isn't given below.

Recommend which chart(s) would best represent this data and why - e.g. a correlation
heatmap for relationships between numeric fields, a bar chart for top categories, a
line chart for a trend over time. You are recommending chart types, not drawing them.

{focus_block}

{data_summary}

Respond ONLY in this exact JSON format, no extra text:
{{
  "summary": "one paragraph recommending which chart(s) fit this data and why, using exact numbers above",
  "key_metrics": ["chart recommendation 1 (type + what it shows)", "recommendation 2", "recommendation 3"],
  "recommendation": "the single most useful chart to look at first, and what to watch for in it"
}}"""

    return get_consensus_json(prompt, temperature=1, max_tokens=2048)
