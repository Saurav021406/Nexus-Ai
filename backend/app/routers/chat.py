"""Natural-language Q&A over an uploaded dataset.

Reuses the same exact, pandas-computed data summary that the analysis agents
use (never raw rows, never guessed numbers), so chat answers stay accurate
and consistent with the Analysis tab.
"""

import json
import google.generativeai as genai
from openai import OpenAI
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.config import settings
from app.deps import get_current_user
from app.services.datasets import build_data_summary, get_dataset_dataframe

router = APIRouter(prefix="/chat", tags=["chat"])

genai.configure(api_key=settings.gemini_api_key)

nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=settings.nvidia_api_key,
)

MAX_HISTORY_MESSAGES = 6  # keep prompts small and cheap


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    dataset_id: str
    question: str = Field(min_length=1, max_length=1000)
    history: list[ChatMessage] = Field(default_factory=list)


@router.post("")
async def chat_with_dataset(payload: ChatRequest, user=Depends(get_current_user)):
    dataframe = get_dataset_dataframe(payload.dataset_id, user.id)
    data_summary = build_data_summary(dataframe)

    recent_history = payload.history[-MAX_HISTORY_MESSAGES:]
    history_text = "\n".join(
        f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" for m in recent_history
    )

    prompt = f"""You are a data analyst answering questions about a dataset. Below is a
precise statistical summary computed exactly from the FULL dataset (not a guess, not a
sample). Use ONLY the numbers given below - never invent or estimate a number that isn't
present in this summary. If the question can't be answered from this summary, say so
clearly instead of guessing.

DATA SUMMARY:
{data_summary}

{"CONVERSATION SO FAR:" if history_text else ""}
{history_text}

USER QUESTION: {payload.question}

Answer in 2-4 short sentences, plain text, no markdown formatting, no JSON."""

    answer = None

    # PRIMARY: NVIDIA
    try:
        completion = nvidia_client.chat.completions.create(
            model="nvidia/nemotron-3-super-120b-a12b",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            top_p=0.95,
            max_tokens=1024,
        )
        answer = completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"NVIDIA failed: {e}")

    # FALLBACK: Gemini
    if not answer:
        try:
            model = genai.GenerativeModel("gemini-3-flash-preview")
            response = model.generate_content(prompt)
            answer = response.text.strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

    return {"answer": answer}