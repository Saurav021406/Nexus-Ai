"""Wraps this project's model providers behind the OpenAI Agents SDK's Model
interface.

Stack (Section 5 of the Phase 4 spec: "Use OpenAI Agents SDK as the primary
agent runtime"):
  1. Groq       - primary   (fast, cheap, strong general-purpose model)
  2. NVIDIA     - secondary (this project's original primary - kept as the
                              first fallback rather than dropped, since it's
                              proven reliable in the rest of the codebase)
  3. OpenRouter - tertiary  (replaces the old direct-Gemini fallback; routed
                              to a Gemini model via OpenRouter so quality
                              stays comparable to what this path had before)

All three expose OpenAI-compatible chat completions endpoints, so all three
go through the SDK's own `OpenAIChatCompletionsModel` - no custom low-level
Model implementation needed for any of them, which keeps this low-risk.

Scope note: this file only affects the new Agents-SDK Manager path
(agents/manager_v2.py -> /agent/run). The original NVIDIA-primary/
Gemini-fallback calls hand-rolled in agents/quality.py, agents/report.py,
routers/chat.py, and the 6 domain specialists are UNCHANGED - migrating
those to this same stack is a deliberately separate, later step.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from agents import (
    AgentOutputSchemaBase,
    Handoff,
    Model,
    ModelResponse,
    ModelSettings,
    ModelTracing,
    OpenAIChatCompletionsModel,
    Tool,
    TResponseInputItem,
    set_tracing_disabled,
)
from agents.items import TResponseStreamEvent
from openai import AsyncOpenAI

from app.config import settings

# This project doesn't use real OpenAI models, so the SDK's built-in
# tracing (which phones home to OpenAI's own tracing API) has nothing
# valid to authenticate with and would otherwise fail on every run.
# agents/tracing.py (this codebase's own tracing) is unaffected - it's a
# separate, already-working mechanism that writes to WorkflowState.traces.
set_tracing_disabled(True)

GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
NVIDIA_MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b"
OPENROUTER_MODEL_NAME = "google/gemini-2.0-flash-001"


def _build_groq_model() -> OpenAIChatCompletionsModel:
    client = AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=settings.groq_api_key,
    )
    return OpenAIChatCompletionsModel(model=GROQ_MODEL_NAME, openai_client=client)


def _build_nvidia_model() -> OpenAIChatCompletionsModel:
    client = AsyncOpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=settings.nvidia_api_key,
    )
    return OpenAIChatCompletionsModel(model=NVIDIA_MODEL_NAME, openai_client=client)


def _build_openrouter_model() -> OpenAIChatCompletionsModel:
    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=settings.openrouter_api_key,
    )
    return OpenAIChatCompletionsModel(model=OPENROUTER_MODEL_NAME, openai_client=client)


class FallbackModel(Model):
    """Tries Groq, then NVIDIA, then OpenRouter, in that order. Each tier
    only gets called if every prior tier raised - a slow primary doesn't
    delay things further by also being retried, it just fails fast onto
    the next tier.

    Streaming only uses the primary tier (Groq) - matches this codebase's
    existing scope, since no agent currently streams. A mid-stream Groq
    failure surfaces directly rather than silently switching providers
    partway through a response.
    """

    def __init__(self) -> None:
        self._tiers = [
            ("Groq", _build_groq_model),
            ("NVIDIA", _build_nvidia_model),
            ("OpenRouter", _build_openrouter_model),
        ]

    async def get_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any = None,
    ) -> ModelResponse:
        errors: list[str] = []

        for tier_name, build_model in self._tiers:
            try:
                model = build_model()
                return await model.get_response(
                    system_instructions,
                    input,
                    model_settings,
                    tools,
                    output_schema,
                    handoffs,
                    tracing,
                    previous_response_id=previous_response_id,
                    conversation_id=conversation_id,
                    prompt=prompt,
                )
            except Exception as e:
                errors.append(f"{tier_name}: {e}")

        raise RuntimeError(
            "All model providers failed (Groq -> NVIDIA -> OpenRouter). " + " | ".join(errors)
        )

    def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[TResponseInputItem],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: ModelTracing,
        *,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any = None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        # No fallback for streaming - see class docstring.
        return _build_groq_model().stream_response(
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            tracing,
            previous_response_id=previous_response_id,
            conversation_id=conversation_id,
            prompt=prompt,
        )


def get_nexus_model() -> FallbackModel:
    """Single entry point every new Agents-SDK-based agent should use."""
    return FallbackModel()

