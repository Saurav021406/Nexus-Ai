import json
import google.generativeai as genai

from app.config import settings
from app.agents.state import WorkflowState

genai.configure(api_key=settings.gemini_api_key)


def review_results(state: WorkflowState) -> dict:
    prompt = f"""You are a strict Reviewer Agent.
Review the specialist reports for quality:

- Are numbers only taken from the data summary? (no invented figures)
- Are the specialists consistent with each other?
- Are recommendations concrete and useful?
- Any contradictions or vague statements?

DATA SUMMARY:
{state.data_summary}

SPECIALIST REPORTS:
{json.dumps(state.specialist_results, indent=2, default=str)}

Respond ONLY with this exact JSON (no markdown, no extra text):
{{
  "overall_quality": "high",
  "issues": [],
  "approved": true,
  "suggested_improvements": []
}}
"""

    model = genai.GenerativeModel("gemini-3-flash-preview")
    response = model.generate_content(prompt)
    text = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(text)
