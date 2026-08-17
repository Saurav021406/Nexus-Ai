from app.services.consensus import get_consensus_json


def analyze(data_summary: str) -> dict:
    prompt = f"""You are a healthcare operations data analyst. Below is a precise
statistical summary of a healthcare-related dataset, computed exactly from the full
data (not a guess). Use ONLY these numbers for any figures you state - never invent
or estimate a number that isn't given below. Focus on operational/statistical patterns
only - do NOT provide any medical diagnosis or clinical advice about individual patients.

{data_summary}

Respond ONLY in this exact JSON format, no extra text:
{{
  "summary": "one paragraph overview of the operational patterns, using exact numbers above",
  "key_metrics": ["metric 1 with an exact number from above", "metric 2", "metric 3"],
  "recommendation": "one actionable operational suggestion based on the data (not medical advice)"
}}"""

    return get_consensus_json(prompt, temperature=1, max_tokens=2048)
