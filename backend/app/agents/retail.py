import json
import google.generativeai as genai
from app.config import settings

genai.configure(api_key=settings.gemini_api_key)


def analyze(data_summary: str) -> dict:
    prompt = f"""You are a retail sales analyst. Below is a precise statistical summary
of a sales dataset, computed exactly from the full data (not a guess). Use ONLY
these numbers for any figures you state - never invent or estimate a number that
isn't given below.

{data_summary}

Respond ONLY in this exact JSON format, no extra text:
{{
  "summary": "one paragraph overview of sales performance, using exact numbers above",
  "key_metrics": ["metric 1 with an exact number from above", "metric 2", "metric 3"],
  "recommendation": "one actionable business suggestion based on the data"
}}"""

    model = genai.GenerativeModel("gemini-3-flash-preview")
    response = model.generate_content(prompt)
    text = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(text)
