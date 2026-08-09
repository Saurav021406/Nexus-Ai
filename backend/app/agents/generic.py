import json

import google.generativeai as genai

from app.config import settings

genai.configure(api_key=settings.gemini_api_key)


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

    model = genai.GenerativeModel("gemini-3-flash-preview")
    response = model.generate_content(prompt)
    text = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(text)
