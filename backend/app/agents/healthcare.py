import json
import google.generativeai as genai
from openai import OpenAI
from app.config import settings

genai.configure(api_key=settings.gemini_api_key)

nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=settings.nvidia_api_key,
)


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

    text = None

    try:
        completion = nvidia_client.chat.completions.create(
            model="nvidia/nemotron-3-super-120b-a12b",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            top_p=0.95,
            max_tokens=2048,
        )
        text = completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"NVIDIA failed: {e}")

    if not text:
        model = genai.GenerativeModel("gemini-3-flash-preview")
        response = model.generate_content(prompt)
        text = response.text.strip()

    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)