"""Natural-language Q&A over an uploaded dataset.

Reuses the same exact, pandas-computed data summary that the analysis agents
use (never raw rows, never guessed numbers), so chat answers stay accurate
and consistent with the Analysis tab.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.deps import get_current_user
from app.services.consensus import get_consensus
from app.services.datasets import build_data_summary, get_dataset_dataframe

router = APIRouter(prefix="/chat", tags=["chat"])

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
    dataframe = await run_in_threadpool(get_dataset_dataframe, payload.dataset_id, user.id)
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

    try:
        result = await run_in_threadpool(get_consensus, prompt, temperature=1, max_tokens=1024)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

    return {"answer": result.answer, "consensus": result.to_meta_dict()}