"""Hybrid Consensus Engine.

    User Query
        |
   +----+----+
   |    |    |
   v    v    v
 Groq Nemotron MiniMax-M3  <- called in PARALLEL (see get_consensus)
   |    |    |
   +----+----+
        v
 Hybrid Consensus Engine
        |
        +-- agreement_detection()      pairwise similarity between answers
        +-- contradiction_detection()  flags pairs that meaningfully diverge
        +-- confidence_scoring()       0-1 score for how much to trust the result
        +-- answer_ranking()           orders answers, not just picks the loudest
        +-- response_synthesis()       builds the final answer from the ranking
        v
   Final Answer

This is a WEIGHTED consensus engine, not simple majority voting: each
model has a trust weight (see MODEL_WEIGHTS) that factors into both ranking
and confidence, alongside how much the models actually agreed with each
other.

Used everywhere an agent needs a single LLM-authored answer: the domain
specialists, report generation, chat, quality/security review, and chart
suggestion. Only raises if ALL three providers fail - callers always get
*some* answer as long as at least one model responded.

Note: this module is for plain prompt -> text/JSON calls. The separate
Agents-SDK path (agents/model_provider.py) has its own model wiring, since
that layer must return tool-calls/handoffs the SDK understands, which a
merged 3-way answer can't represent - see that file's docstring.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings

GROQ_MODEL_NAME = "openai/gpt-oss-120b"
NVIDIA_MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b"
# Also served via NVIDIA's build.nvidia.com catalog, so this reuses the
# same NVIDIA client/API key as NVIDIA_MODEL_NAME above - just a different
# model string on the same OpenAI-compatible endpoint.
MINIMAX_MODEL_NAME = "minimaxai/minimax-m3"

# Relative trust weights used to rank answers and weigh confidence. Not a
# benchmark score - just this project's judgment on whose answer to lean on
# more heavily when models disagree. Must sum to 1.0.
MODEL_WEIGHTS = {
    "minimax": 0.40,
    "nvidia": 0.35,
    "groq": 0.25,
}

# Below this pairwise similarity, two models' answers are flagged as an
# actual disagreement rather than just different wording of the same thing.
CONTRADICTION_THRESHOLD = 0.35


def _groq_client() -> OpenAI:
    return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=settings.groq_api_key,timeout=20)


def _nvidia_client() -> OpenAI:
    return OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=settings.nvidia_api_key,timeout=20)


def _minimax_client() -> OpenAI:
    return OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=settings.nvidia_api_key,timeout=20)


def _call_groq(prompt: str, temperature: float, max_tokens: int) -> str:
    completion = _groq_client().chat.completions.create(
        model=GROQ_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content.strip()


def _call_nvidia(prompt: str, temperature: float, max_tokens: int) -> str:
    completion = _nvidia_client().chat.completions.create(
        model=NVIDIA_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content.strip()


def _call_minimax(prompt: str, temperature: float, max_tokens: int) -> str:
    completion = _minimax_client().chat.completions.create(
        model=MINIMAX_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=min(temperature, 1.0),
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content.strip()


_PROVIDERS = {
    "groq": _call_groq,
    "nvidia": _call_nvidia,
    "minimax": _call_minimax,
}

# Stable display/tie-break order, independent of which finishes first.
_ORDER = {"minimax": 0, "nvidia": 1, "groq": 2}


@dataclass
class ModelAnswer:
    name: str
    weight: float
    text: str | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.text is not None


@dataclass
class RankedAnswer:
    """An answer plus its composite rank score, produced by answer_ranking()."""

    answer: ModelAnswer
    rank_score: float  # 0-1, blends trust weight + agreement with the others


@dataclass
class ConsensusResult:
    answer: str
    responses: list[ModelAnswer] = field(default_factory=list)
    ranking: list[str] = field(default_factory=list)  # model names, best first
    agreement_score: float = 0.0
    contradictions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    models_used: list[str] = field(default_factory=list)
    synthesis_notes: list[str] = field(default_factory=list)

    def to_meta_dict(self) -> dict:
        """Small JSON-safe summary of the consensus run, suitable for
        attaching to an API response or storing in traces."""
        return {
            "models_used": self.models_used,
            "ranking": self.ranking,
            "agreement_score": self.agreement_score,
            "confidence": self.confidence,
            "contradictions": self.contradictions,
            "synthesis_notes": self.synthesis_notes,
        }


# ---------------------------------------------------------------------------
# Stage 1: parallel calls
# ---------------------------------------------------------------------------

def _query_all_models(prompt: str, temperature: float, max_tokens: int) -> list[ModelAnswer]:
    """Calls Groq, NVIDIA, and MiniMax-M3 at the same time via a thread pool -
    this is the "User Query -> Groq / Nemotron / MiniMax-M3" fan-out in the
    diagram. A model that raises doesn't block the others; it's recorded
    as a failed ModelAnswer instead."""
    answers: list[ModelAnswer] = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        future_to_name = {
            pool.submit(fn, prompt, temperature, max_tokens): name for name, fn in _PROVIDERS.items()
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                text = future.result()
                answers.append(ModelAnswer(name=name, weight=MODEL_WEIGHTS[name], text=text))
            except Exception as e:
                print(f"{name} failed in consensus engine: {e}")
                answers.append(ModelAnswer(name=name, weight=MODEL_WEIGHTS[name], text=None, error=str(e)))
    answers.sort(key=lambda a: _ORDER[a.name])
    return answers


# ---------------------------------------------------------------------------
# Stage 2: agreement detection
# ---------------------------------------------------------------------------

def agreement_detection(ok_answers: list[ModelAnswer]) -> tuple[list[list[float]], float]:
    """Pairwise TF-IDF cosine similarity between every model's answer - a
    lightweight, dependency-light stand-in for semantic similarity, good
    enough to tell "these three basically agree" from "these three said
    different things" without a 4th network call just to compare text.

    Returns (similarity_matrix, overall_agreement_score)."""
    texts = [a.text for a in ok_answers]
    if len(texts) < 2:
        return [[1.0]], 1.0  # nothing to compare against

    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform(texts)
        sim_matrix = cosine_similarity(matrix).tolist()
    except ValueError:
        # e.g. every answer was empty or pure stopwords
        sim_matrix = [[1.0 if a == b else 0.0 for b in texts] for a in texts]

    n = len(texts)
    pairs = [sim_matrix[i][j] for i in range(n) for j in range(i + 1, n)]
    agreement_score = sum(pairs) / len(pairs)
    return sim_matrix, agreement_score


# ---------------------------------------------------------------------------
# Stage 3: contradiction detection
# ---------------------------------------------------------------------------

def contradiction_detection(ok_answers: list[ModelAnswer], sim_matrix: list[list[float]]) -> list[str]:
    """Flags model pairs whose answers fall below CONTRADICTION_THRESHOLD -
    a cheap proxy for "these two are saying different/conflicting things"
    that a caller (or human reviewer) should look at."""
    flags = []
    for i in range(len(ok_answers)):
        for j in range(i + 1, len(ok_answers)):
            score = sim_matrix[i][j]
            if score < CONTRADICTION_THRESHOLD:
                flags.append(
                    f"{ok_answers[i].name} and {ok_answers[j].name} diverge significantly "
                    f"(similarity {score:.2f})"
                )
    return flags


# ---------------------------------------------------------------------------
# Stage 4: answer ranking
# ---------------------------------------------------------------------------

def answer_ranking(ok_answers: list[ModelAnswer], sim_matrix: list[list[float]]) -> list[RankedAnswer]:
    """Ranks answers by a WEIGHTED composite score - this is what makes the
    engine "weighted consensus" instead of simple majority voting. Each
    answer's score blends:
      - its model's static trust weight (MODEL_WEIGHTS), and
      - how closely it agrees with the OTHER answers (an answer that the
        other models corroborate ranks higher than an outlier, even if its
        raw model weight is slightly lower).
    Returns answers sorted best-first."""
    n = len(ok_answers)
    ranked: list[RankedAnswer] = []
    for i, ans in enumerate(ok_answers):
        if n > 1:
            others = [sim_matrix[i][j] for j in range(n) if j != i]
            corroboration = sum(others) / len(others)
        else:
            corroboration = 1.0
        rank_score = (0.6 * ans.weight / max(MODEL_WEIGHTS.values())) + (0.4 * corroboration)
        ranked.append(RankedAnswer(answer=ans, rank_score=rank_score))

    ranked.sort(key=lambda r: r.rank_score, reverse=True)
    return ranked


# ---------------------------------------------------------------------------
# Stage 5: confidence scoring
# ---------------------------------------------------------------------------

def confidence_scoring(
    agreement_score: float,
    ranked: list[RankedAnswer],
    total_providers: int,
    contradictions: list[str],
) -> float:
    """Blends: how much the models agreed, how dominant the top-ranked
    answer is over the rest, and how many of the 3 providers actually
    responded (fewer responses = less cross-checking = lower confidence).
    Penalized further if any contradictions were flagged."""
    top_score = ranked[0].rank_score
    score_total = sum(r.rank_score for r in ranked) or 1.0
    dominance = top_score / score_total
    availability = len(ranked) / total_providers

    confidence = (0.5 * agreement_score) + (0.3 * dominance) + (0.2 * availability)
    if contradictions:
        confidence *= 0.85  # penalize flagged disagreement between models
    return round(confidence, 3)


# ---------------------------------------------------------------------------
# Stage 6: response synthesis
# ---------------------------------------------------------------------------

def response_synthesis(ranked: list[RankedAnswer], contradictions: list[str]) -> tuple[str, list[str]]:
    """Builds the FINAL ANSWER from the ranking, rather than blindly
    returning the raw top pick. The top-ranked answer is the base (it's
    already the one the ranking stage decided was best-supported), but this
    is where disagreement gets surfaced instead of silently discarded: if
    other models meaningfully contradicted the winner, that's recorded as a
    synthesis note attached to the result so callers/UI can show it.

    Deliberately does NOT make a 4th LLM call to blend the texts together -
    that would add real latency/cost to every one of the dozens of calls
    this engine handles per analysis run, for a synthesis whose main job
    (surfacing disagreement) is already handled by the notes below."""
    notes: list[str] = []
    winner = ranked[0].answer

    if len(ranked) > 1:
        runner_up = ranked[1].answer
        was_contradicted = any(winner.name in c and runner_up.name in c for c in contradictions)
        if was_contradicted:
            notes.append(
                f"{winner.name} was selected (highest rank score), but {runner_up.name} "
                f"gave a meaningfully different answer - treat with some caution."
            )
        elif len(ranked) < 3:
            notes.append(f"Only {len(ranked)}/3 models responded ({winner.name} selected).")

    return winner.text, notes


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_consensus(prompt: str, *, temperature: float = 0.7, max_tokens: int = 1024) -> ConsensusResult:
    """Runs the full Hybrid Consensus Engine pipeline:

        query Groq + NVIDIA + MiniMax-M3 in parallel
          -> agreement_detection
          -> contradiction_detection
          -> answer_ranking
          -> confidence_scoring
          -> response_synthesis
          -> Final Answer

    Raises RuntimeError only if all three providers fail."""
    answers = _query_all_models(prompt, temperature, max_tokens)
    ok_answers = [a for a in answers if a.ok]

    if not ok_answers:
        errors = " | ".join(f"{a.name}: {a.error}" for a in answers)
        raise RuntimeError(f"All model providers failed (Groq, NVIDIA, MiniMax-M3). {errors}")

    sim_matrix, agreement_score = agreement_detection(ok_answers)
    contradictions = contradiction_detection(ok_answers, sim_matrix)
    ranked = answer_ranking(ok_answers, sim_matrix)
    confidence = confidence_scoring(agreement_score, ranked, total_providers=len(_PROVIDERS), contradictions=contradictions)
    final_answer, synthesis_notes = response_synthesis(ranked, contradictions)

    return ConsensusResult(
        answer=final_answer,
        responses=answers,
        ranking=[r.answer.name for r in ranked],
        agreement_score=round(agreement_score, 3),
        contradictions=contradictions,
        confidence=confidence,
        models_used=[a.name for a in ok_answers],
        synthesis_notes=synthesis_notes,
    )


def get_consensus_json(prompt: str, *, temperature: float = 0.7, max_tokens: int = 2048) -> dict:
    """Same as get_consensus(), but parses the winning answer as JSON (after
    stripping ```json code fences, since every provider here occasionally
    adds them anyway) and attaches consensus metadata under a "_consensus"
    key so callers can surface it without colliding with the model's own
    JSON fields."""
    result = get_consensus(prompt, temperature=temperature, max_tokens=max_tokens)
    text = result.answer.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        parsed["_consensus"] = result.to_meta_dict()
    return parsed
