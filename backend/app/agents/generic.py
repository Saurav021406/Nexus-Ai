from app.services.consensus import get_consensus_json


def analyze(data_summary: str, task_description: str = "") -> dict:
    """Safe fallback when no specialist is a confident fit for a dataset."""
    focus_block = (
        f"The user's specific request for this analysis is: {task_description}\n"
        "Directly address this request in your summary, key_metrics, and recommendation, "
        "while still only using the exact numbers given below."
        if task_description else ""
    )
    prompt = f"""You are a general data analyst. The following is a privacy-safe,
precise statistical summary computed from the full dataset. Use ONLY the supplied
figures for any number you state. Do not infer a business domain that is not
supported by the summary.

{focus_block}

{data_summary}

Respond ONLY in this exact JSON format, no extra text:
{{
  "summary": "one paragraph overview using exact numbers above",
  "key_metrics": ["metric 1 with an exact number from above", "metric 2", "metric 3"],
  "recommendation": "one actionable, domain-neutral next step"
}}"""

    return get_consensus_json(prompt, temperature=1, max_tokens=2048)