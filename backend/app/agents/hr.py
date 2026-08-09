import json
import google.generativeai as genai
from app.config import settings

genai.configure(api_key=settings.gemini_api_key)


def analyze(data_summary: str) -> dict:
    prompt = f"""You are an HR / workforce analytics specialist. Below is a precise
statistical summary of an employee/HR dataset, computed exactly from the full data
(not a guess). Use ONLY these numbers for any figures you state - never invent or
estimate a number that isn't given below.

{data_summary}

Respond ONLY in this exact JSON format, no extra text:
{{
  "summary": "one paragraph overview of the workforce data, using exact numbers above",
  "key_metrics": ["metric 1 with an exact number from above", "metric 2", "metric 3"],
  "recommendation": "one actionable HR/people-management suggestion based on the data"
}}"""

    model = genai.GenerativeModel("gemini-3-flash-preview")
    response = model.generate_content(prompt)
    text = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(text)
