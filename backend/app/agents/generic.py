from app.services.consensus import get_consensus_json


def analyze(data_summary: str) -> dict:
    """Safe fallback when no specialist is a confident fit for a dataset."""
    prompt = f"""You are a general data analyst. The following is a privacy-safe,
precise statistical summary computed from the full dataset. Use ONLY the supplied
figures for any number you state. Do not infer a business domain that is not
supported by the summary.

{data_summary}

Respond ONLY in this exact JSON format, no extra text:
{{
  "summary": "one paragraph overview using exact numbers above",
  "key_metrics": ["metric 1 with an exact number from above", "metric 2", "metric 3"],
  "recommendation": "one actionable, domain-neutral next step"
}}"""

    return get_consensus_json(prompt, temperature=1, max_tokens=2048)