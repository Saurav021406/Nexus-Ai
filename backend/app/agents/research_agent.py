"""Research Agent (Section 16 of the Phase 4 spec).

"Provide the architecture/hook for: Knowledge retrieval, Research tasks,
Relevant document/source retrieval. Full RAG belongs to Phase 6."

This is deliberately honest about what it can and can't do right now:
there's no document corpus, no vector store, no retrieval system in this
app yet (that's the RAG subsystem being designed separately - Domain
Router -> ingestion -> chunking -> embeddings -> hybrid retrieval). Until
that exists, this agent answers research-flavored questions using the
model's own general knowledge, and says so explicitly rather than
pretending an answer is grounded in a source it never actually retrieved.

The point of building this now, even without real retrieval behind it, is
the interface itself (Section 16's "hook"): it's already a registered
agent the Manager can select for a research-shaped question, with the
same analyze(data_summary, task_description) signature every other
reasoning-only agent uses. When RAG lands, this file's internals get
replaced with real retrieval - nothing about the registry entry, the
Manager's ability to pick it, or its output shape needs to change.
"""

from app.services.consensus import get_consensus_json


def analyze(data_summary: str, task_description: str = "") -> dict:
    task_description = task_description or "Provide relevant external context for this dataset"

    prompt = f"""You are the Research Agent. You do NOT have access to any external
document corpus, web search, or retrieval system - that capability doesn't exist yet
in this platform. You only have your own general knowledge.

The user's request:
{task_description}

Dataset context (for relevance only - do not treat this as something to search):
{data_summary}

Rules:
- Answer using only your own general/background knowledge - clearly note that this is
  NOT grounded in any retrieved source or document specific to this user.
- If the question genuinely requires a specific document, internal policy, or a source
  you cannot possibly know, say so plainly instead of guessing or inventing a citation.
- Never fabricate a citation, URL, or document reference - you have none to cite.
- Keep it concise and clearly caveated.

Respond ONLY in this exact JSON format, no extra text:
{{
  "summary": "one paragraph answering the request from general knowledge, clearly caveated as not source-grounded",
  "key_metrics": ["relevant general-knowledge point 1", "point 2"],
  "recommendation": "one suggestion, noting if a real source/document would be needed to confirm this"
}}"""

    result = get_consensus_json(prompt, temperature=0.7, max_tokens=1024)
    result["grounded"] = False  # explicit flag: this answer has no retrieved sources behind it
    return result
