"""Data Engineer Agent (Section 16 of the Phase 4 spec).

Capabilities: dataset profiling, data quality/validation focus, cleaning
recommendations. This is the LLM-reasoning layer only - the actual cleaning
operations already exist as working code in services/cleaning.py
(/clean/apply, /clean/quality) and are NOT duplicated here per the spec's
own guidance ("Use Phase 2 services rather than rebuilding them"). This
agent's job is to let the Manager delegate a data-quality-flavored
question to something that answers with an engineering lens, not a
business or statistical one - the same data_summary a Business Analyst or
Data Scientist agent would see gets a different kind of answer here.
"""

from app.services.consensus import get_consensus_json


def analyze(data_summary: str, task_description: str = "") -> dict:
    focus_block = (
        f"The user's specific request for this analysis is: {task_description}\n"
        "Directly address this request in your summary, key_metrics, and recommendation, "
        "while still only using the exact numbers given below."
        if task_description else ""
    )
    prompt = f"""You are a Data Engineer. Below is a precise statistical summary of a
dataset, computed exactly from the full data (not a guess). Use ONLY these numbers for
any figures you state - never invent or estimate a number that isn't given below.

Focus specifically on DATA QUALITY and ENGINEERING concerns: missing values, likely
type issues, outliers, duplicate risk, column naming/consistency problems, and whether
the dataset is in a fit state for downstream analysis. Do not write a general business
summary - a Business Analyst agent already covers that.

{focus_block}

{data_summary}

Respond ONLY in this exact JSON format, no extra text:
{{
  "summary": "one paragraph on the data's engineering quality, using exact numbers above",
  "key_metrics": ["quality metric 1 with an exact number from above", "metric 2", "metric 3"],
  "recommendation": "one concrete data-cleaning or pipeline action to take next"
}}"""

    return get_consensus_json(prompt, temperature=1, max_tokens=2048)
