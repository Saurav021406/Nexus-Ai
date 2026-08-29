"""Natural-language Q&A over an uploaded dataset.

This is the Router (Step 6 of the RAG design) - it decides which of the
two engines a question belongs to, and is the single place both paths
converge on the same Multi-Agent + Consensus answer generation:

    User Query
        |
   is_document_dataset()?
        |
   +----+-----------------------------+
   |                                  |
 TABULAR                          DOCUMENT
   |                                  |
 Domain Gate (agents/domain_gate)   Hybrid Retrieval (services/retrieval)
   |  reject -> no LLM call           |  fetches CANDIDATE_CHUNKS (15)
   v                                  v
 pandas data_summary                Reranker (services/reranker)
   |                                  |  rescores & keeps TOP_K_CHUNKS (5)
   |                                  v
   |                                Evidence Gate (services/evidence_gate)
   |                                  |  reject -> no LLM call
   |                                  v
   |                              retrieved chunks -> context
   |                                  |
   +----------------+-----------------+
                    v
         get_consensus() - same Hybrid Consensus Engine
         (Groq + NVIDIA + MiniMax) for both paths
                    v
              Final Answer (+ sources, for documents)

Tabular answers reuse the exact, pandas-computed data summary the analysis
agents use (never raw rows, never guessed numbers). Document answers are
grounded only in the actual retrieved excerpts (never outside knowledge),
with which excerpt each part of the answer came from returned as `sources`
so the UI can show citations.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.agents.domain_gate import check_domain_relevance
from app.deps import get_current_user
from app.services.consensus import get_consensus
from app.services.datasets import build_data_summary, get_dataset_dataframe, is_document_dataset
from app.services.evidence_gate import check_evidence
from app.services.reranker import rerank_chunks
from app.services.retrieval import hybrid_search

router = APIRouter(prefix="/chat", tags=["chat"])

MAX_HISTORY_MESSAGES = 6  # keep prompts small and cheap
CANDIDATE_CHUNKS = 15     # wider initial pool from hybrid search
TOP_K_CHUNKS = 5          # final best reranked chunks fed to LLM context


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    dataset_id: str
    question: str = Field(min_length=1, max_length=1000)
    history: list[ChatMessage] = Field(default_factory=list)


def _history_text(history: list[ChatMessage]) -> str:
    recent_history = history[-MAX_HISTORY_MESSAGES:]
    return "\n".join(
        f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}" for m in recent_history
    )


@router.post("")
async def chat_with_dataset(payload: ChatRequest, user=Depends(get_current_user)):
    """The Router: picks the Tabular Engine or the RAG Engine based on
    what kind of dataset this actually is, then hands off to whichever
    engine applies. Both engines end at the same get_consensus() call."""
    document = await run_in_threadpool(is_document_dataset, payload.dataset_id, user.id)

    if document:
        return await _chat_over_document(payload, user.id)
    return await _chat_over_tabular(payload, user.id)


async def _chat_over_tabular(payload: ChatRequest, user_id: str) -> dict:
    dataframe = await run_in_threadpool(get_dataset_dataframe, payload.dataset_id, user_id)
    data_summary = build_data_summary(dataframe)

    # Domain Gate (Step 1): free, instant, no LLM call. Rejects questions
    # that have nothing to do with this dataset before they ever reach an
    # expensive consensus call.
    gate = check_domain_relevance(payload.question, list(dataframe.columns), data_summary)
    if not gate["in_domain"]:
        return {"answer": gate["reason"], "consensus": None, "sources": None, "in_domain": False}

    history_text = _history_text(payload.history)

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
        result = await run_in_threadpool(get_consensus, prompt, temperature=0.1, max_tokens=1024)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

    return {"answer": result.answer, "consensus": result.to_meta_dict(), "sources": None, "in_domain": True}


async def _chat_over_document(payload: ChatRequest, user_id: str) -> dict:
    # 1. Hybrid Retrieval (Step 4): vector + keyword search pool
    candidates = await run_in_threadpool(
        hybrid_search, payload.dataset_id, user_id, payload.question, CANDIDATE_CHUNKS
    )

    # 2. Reranker (Step 7): neural cross-encoder / lexical rescore to Top 5
    chunks = await run_in_threadpool(
        rerank_chunks, payload.question, candidates, TOP_K_CHUNKS
    )

    # 3. Evidence Gate (Step 5): free, instant, no LLM call
    evidence = check_evidence(chunks)
    if not evidence["has_evidence"]:
        return {"answer": evidence["reason"], "consensus": None, "sources": [], "in_domain": True}

    context = "\n\n---\n\n".join(
        f"[Excerpt {i + 1}]\n{c['chunk_text']}" for i, c in enumerate(chunks)
    )
    history_text = _history_text(payload.history)

    prompt = f"""You are answering a question using ONLY the document excerpts below - the
most relevant sections retrieved from the uploaded document for this exact question. Do
NOT use any outside knowledge. If the excerpts don't actually contain the answer, say so
clearly instead of guessing.

DOCUMENT EXCERPTS:
{context}

{"CONVERSATION SO FAR:" if history_text else ""}
{history_text}

USER QUESTION: {payload.question}

Answer in 2-4 short sentences, plain text, no markdown formatting, no JSON. Reference
excerpt numbers like [Excerpt 1] when citing a specific claim."""

    try:
        result = await run_in_threadpool(get_consensus, prompt, temperature=0.1, max_tokens=1024)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")

    sources = [
        {
            "excerpt_number": i + 1,
            "chunk_index": c.get("chunk_index"),
            "preview": c["chunk_text"][:160],
            "score": c.get("rerank_score", c.get("score")),
        }
        for i, c in enumerate(chunks)
    ]

    return {"answer": result.answer, "consensus": result.to_meta_dict(), "sources": sources, "in_domain": True}