"""Consensus Engine.

Replaces this project's old "NVIDIA primary, Gemini fallback" pattern.
Queries Groq, NVIDIA Nemotron, and Claude in PARALLEL for every prompt, then
merges the three answers into one response using:

  - agreement scoring   (pairwise TF-IDF cosine similarity between answers)
  - contradiction flags (model pairs whose answers diverge significantly)
  - weighted selection  (the highest-weighted model that actually responded
                          wins as the final answer - see MODEL_WEIGHTS)
  - a confidence score  (blends agreement, the winner's weight share, and
                          how many of the 3 providers actually responded)

This is used everywhere an agent needs a single LLM-authored answer: the
domain specialists, report generation, chat, quality/security review, and
chart suggestion. It only raises if ALL three providers fail - callers
always get *some* answer as long as at least one model responded.

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

GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
NVIDIA_MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b"
# Routed through OpenRouter (not a direct Anthropic API key) using
# OpenRouter's OpenAI-compatible endpoint, so this uses the same OpenAI()
# client shape as Groq/NVIDIA above.
CLAUDE_MODEL_NAME ="anthropic/claude-sonnet-4.5"

# Relative trust weights used when the three models disagree. Not a
# benchmark score - just this project's judgment on whose answer to lean on
# more heavily. Must sum to 1.0.
MODEL_WEIGHTS = {
    "claude": 0.40,
    "nvidia": 0.35,
    "groq": 0.25,
}

# Below this pairwise similarity, two models' answers are flagged as an
# actual disagreement rather than just different wording of the same thing.
CONTRADICTION_THRESHOLD = 0.35


def _groq_client() -> OpenAI:
    return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=settings.groq_api_key)


def _nvidia_client() -> OpenAI:
    return OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=settings.nvidia_api_key)


def _claude_client() -> OpenAI:
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=settings.openrouter_api_key)


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


def _call_claude(prompt: str, temperature: float, max_tokens: int) -> str:
    completion = _claude_client().chat.completions.create(
        model=CLAUDE_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=min(temperature, 1.0),
        max_tokens=max_tokens,
    )
    return completion.choices[0].message.content.strip()


_PROVIDERS = {
    "groq": _call_groq,
    "nvidia": _call_nvidia,
    "claude": _call_claude,
}

# Stable display/tie-break order, independent of which finishes first.
_ORDER = {"claude": 0, "nvidia": 1, "groq": 2}


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
class ConsensusResult:
    answer: str
    responses: list[ModelAnswer] = field(default_factory=list)
    agreement_score: float = 0.0
    contradictions: list[str] = field(default_factory=list)
    confidence: float = 0.0
    models_used: list[str] = field(default_factory=list)

    def to_meta_dict(self) -> dict:
        """Small JSON-safe summary of the consensus run, suitable for
        attaching to an API response or storing in traces."""
        return {
            "models_used": self.models_used,
            "agreement_score": self.agreement_score,
            "confidence": self.confidence,
            "contradictions": self.contradictions,
        }


def _agreement_matrix(texts: list[str]) -> list[list[float]]:
    """Pairwise cosine similarity over TF-IDF vectors. A lightweight,
    dependency-light stand-in for semantic similarity - good enough to tell
    "these three basically agree" from "these three said different things"
    without a 4th network call just to compare text."""
    if len(texts) < 2:
        return [[1.0]]
    try:
        matrix = TfidfVectorizer(stop_words="english").fit_transform(texts)
        return cosine_similarity(matrix).tolist()
    except ValueError:
        # e.g. every answer was empty or pure stopwords
        return [[1.0 if a == b else 0.0 for b in texts] for a in texts]


def _detect_contradictions(ok_answers: list[ModelAnswer], sim_matrix: list[list[float]]) -> list[str]:
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


def get_consensus(prompt: str, *, temperature: float = 0.7, max_tokens: int = 1024) -> ConsensusResult:
    """Calls Groq, NVIDIA Nemotron, and Claude in parallel and runs the
    consensus engine over their answers. Raises RuntimeError only if all
    three providers fail."""
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
    ok_answers = [a for a in answers if a.ok]

    if not ok_answers:
        errors = " | ".join(f"{a.name}: {a.error}" for a in answers)
        raise RuntimeError(f"All model providers failed (Groq, NVIDIA, Claude). {errors}")

    sim_matrix = _agreement_matrix([a.text for a in ok_answers])
    n = len(ok_answers)
    if n > 1:
        pairs = [sim_matrix[i][j] for i in range(n) for j in range(i + 1, n)]
        agreement_score = sum(pairs) / len(pairs)
    else:
        agreement_score = 1.0  # nothing to compare against

    contradictions = _detect_contradictions(ok_answers, sim_matrix)

    # The highest-weight model that actually answered wins as the final
    # answer. Cheaper and more deterministic than a 4th LLM call to
    # synthesize a merged answer, and keeps this fast since agents in this
    # app call it dozens of times per analysis run.
    winner = max(ok_answers, key=lambda a: a.weight)

    weight_share = winner.weight / sum(a.weight for a in ok_answers)
    availability = n / 3
    confidence = (0.5 * agreement_score) + (0.3 * weight_share) + (0.2 * availability)
    if contradictions:
        confidence *= 0.85  # penalize flagged disagreement between models

    return ConsensusResult(
        answer=winner.text,
        responses=answers,
        agreement_score=round(agreement_score, 3),
        contradictions=contradictions,
        confidence=round(confidence, 3),
        models_used=[a.name for a in ok_answers],
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
