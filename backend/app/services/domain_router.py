"""Deterministic, multi-domain routing for dataset analysis.

The router deliberately uses schema-level signals (the filename and column names),
not arbitrary sample rows.  That keeps routing explainable and avoids exposing row
data to the model before analysis starts.  It can be extended through the signal
configuration below without adding one-off conditions to API routes.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import pandas as pd


MAX_SPECIALISTS = 3
MIN_ROUTABLE_SCORE = 5.0


# Keep this in one configuration layer so a new specialist can be added without
# changing router control flow. Multi-word phrases carry more weight than broad
# words such as "student" or "health".
DOMAIN_SIGNALS: dict[str, dict[str, float]] = {
    "Healthcare": {
        "mental health": 10,
        "anxiety": 9,
        "depression": 9,
        "diagnosis": 9,
        "patient": 8,
        "clinical": 7,
        "symptom": 7,
        "stress": 6,
        "wellbeing": 6,
        "wellness": 6,
        "mental": 5,
        "sleep": 4,
        "health": 3,
        "physical activity": 2,
    },
    "Education": {
        "academic": 8,
        "grade": 8,
        "exam": 8,
        "attendance": 7,
        "course": 7,
        "school": 7,
        "university": 7,
        "learning": 6,
        "study": 5,
        "student": 3,
    },
    "Finance": {
        "transaction": 8,
        "balance": 8,
        "invoice": 7,
        "expense": 7,
        "revenue": 6,
        "profit": 6,
        "budget": 6,
        "payment": 6,
        "account": 5,
        "currency": 5,
    },
    "HR": {
        "employee": 8,
        "payroll": 8,
        "salary": 7,
        "attrition": 7,
        "recruitment": 7,
        "department": 5,
        "workforce": 5,
        "tenure": 5,
        "job role": 5,
    },
    "Retail": {
        "product": 7,
        "customer": 6,
        "order": 6,
        "inventory": 6,
        "quantity": 5,
        "store": 5,
        "sku": 5,
        "discount": 5,
        "sales": 4,
    },
}

TAG_SIGNALS: dict[str, tuple[str, ...]] = {
    "mental health": ("mental health", "mental", "stress", "anxiety", "depression"),
    "wellness": ("wellbeing", "wellness", "sleep", "physical activity"),
    "social media": ("social media", "platform", "daily unlock", "screen time"),
    "students": ("student", "academic", "study"),
    "sales": ("sales", "order", "product", "inventory"),
    "workforce": ("employee", "salary", "payroll", "attrition"),
}


@dataclass(frozen=True)
class DomainCandidate:
    domain: str
    score: float
    matched_signals: list[str]


@dataclass(frozen=True)
class DomainRoute:
    primary_domain: str
    secondary_domains: list[str]
    tags: list[str]
    confidence: float
    reasoning: str
    agent_domains: list[str]
    candidates: list[DomainCandidate]

    def to_dict(self) -> dict:
        return asdict(self)


def _normalise(value: str) -> str:
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    value = value.replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", value.lower()).strip()


def _matched_signals(text: str, signals: dict[str, float]) -> list[str]:
    return [signal for signal in signals if signal in text]


def _format_signal_list(signals: list[str]) -> str:
    return ", ".join(signal.replace("_", " ") for signal in signals[:3])


def classify_dataset(filename: str, df: pd.DataFrame) -> DomainRoute:
    """Return a transparent primary/secondary route from the dataset schema.

    Column names are the strongest evidence. Filename evidence is intentionally
    discounted so a misleading filename cannot outweigh the actual schema.
    """
    source_texts = [(_normalise(str(column)), 1.0) for column in df.columns]
    source_texts.append((_normalise(filename), 0.65))

    candidates: list[DomainCandidate] = []
    for domain, signals in DOMAIN_SIGNALS.items():
        score = 0.0
        matched: list[str] = []
        for text, source_weight in source_texts:
            for signal in _matched_signals(text, signals):
                score += signals[signal] * source_weight
                if signal not in matched:
                    matched.append(signal)
        candidates.append(
            DomainCandidate(
                domain=domain,
                score=round(score, 2),
                matched_signals=matched,
            )
        )

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    top_candidate = candidates[0]

    searchable_text = " ".join(text for text, _ in source_texts)
    tags = [
        tag
        for tag, signals in TAG_SIGNALS.items()
        if any(signal in searchable_text for signal in signals)
    ][:5]

    if top_candidate.score < MIN_ROUTABLE_SCORE:
        return DomainRoute(
            primary_domain="General",
            secondary_domains=[],
            tags=tags or ["unclassified"],
            confidence=0.35,
            reasoning="The schema has no strong specialist-domain signals, so a general analysis will run.",
            agent_domains=["General"],
            candidates=candidates[:3],
        )

    secondary_threshold = max(MIN_ROUTABLE_SCORE, top_candidate.score * 0.35)
    secondary = [
        candidate
        for candidate in candidates[1:]
        if candidate.score >= secondary_threshold
    ][: MAX_SPECIALISTS - 1]
    agent_domains = [top_candidate.domain, *[candidate.domain for candidate in secondary]]

    margin = top_candidate.score - (candidates[1].score if len(candidates) > 1 else 0)
    confidence = min(0.98, 0.55 + min(top_candidate.score, 30) / 100 + min(max(margin, 0), 15) / 100)
    secondary_domains = [candidate.domain for candidate in secondary]

    reasoning = (
        f"{top_candidate.domain} is primary because of {_format_signal_list(top_candidate.matched_signals)} schema signals."
    )
    if secondary:
        secondary_details = "; ".join(
            f"{candidate.domain}: {_format_signal_list(candidate.matched_signals)}"
            for candidate in secondary
        )
        reasoning += f" Also relevant: {secondary_details}."

    return DomainRoute(
        primary_domain=top_candidate.domain,
        secondary_domains=secondary_domains,
        tags=tags,
        confidence=round(confidence, 2),
        reasoning=reasoning,
        agent_domains=agent_domains,
        candidates=candidates[:3],
    )
