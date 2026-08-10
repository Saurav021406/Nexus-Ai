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

    text = None

    # PRIMARY: NVIDIA
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

    # FALLBACK: Gemini (only if NVIDIA fails)
    if not text:
        try:
            model = genai.GenerativeModel("gemini-3-flash-preview")
            response = model.generate_content(prompt)
            text = response.text.strip()
        except Exception as e:
            print(f"Gemini also failed: {e}")
            raise

    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)