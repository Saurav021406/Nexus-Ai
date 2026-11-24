from types import SimpleNamespace

import pytest

from app.services import consensus


@pytest.mark.parametrize(
    ("provider_call", "expected_model", "expected_base_url", "key_name"),
    [
        (
            consensus._call_nvidia,
            "nvidia/nemotron-3.5-super-vl-120b-a12b",
            "https://integrate.api.nvidia.com/v1",
            "nvidia_api_key",
        ),
        (
            consensus._call_gemma,
            "google/gemma-4-26b-a4b-it:free",
            "https://openrouter.ai/api/v1",
            "openrouter_api_key",
        ),
    ],
)
def test_provider_calls_use_expected_catalog_models(
    monkeypatch, provider_call, expected_model, expected_base_url, key_name
):
    request = {}
    client_config = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            client_config.update(kwargs)
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create_completion)
            )

        @staticmethod
        def create_completion(**kwargs):
            request.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
            )

    monkeypatch.setattr(consensus, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(consensus.settings, key_name, "test-key")

    assert provider_call("prompt", 0.2, 50) == "answer"
    assert request["model"] == expected_model
    assert client_config["base_url"] == expected_base_url
    assert client_config["api_key"] == "test-key"


@pytest.mark.parametrize(
    ("provider_call", "provider_name"),
    [
        (consensus._call_nvidia, "NVIDIA"),
        (consensus._call_gemma, "Gemma"),
    ],
)
def test_provider_calls_report_missing_text_content(
    monkeypatch, provider_call, provider_name
):
    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: SimpleNamespace(
                        choices=[
                            SimpleNamespace(
                                message=SimpleNamespace(content=None),
                                finish_reason="tool_calls",
                            )
                        ]
                    )
                )
            )

    monkeypatch.setattr(consensus, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(consensus.settings, "nvidia_api_key", "test-key")
    monkeypatch.setattr(consensus.settings, "openrouter_api_key", "test-key")
    monkeypatch.setattr(consensus.time, "sleep", lambda _: None)

    with pytest.raises(
        ValueError, match=rf"{provider_name} returned no text content"
    ):
        provider_call("prompt", 0.2, 50)


def test_nvidia_retries_with_larger_budget_when_reasoning_uses_output_limit(monkeypatch):
    token_limits = []

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self.create_completion)
            )

        @staticmethod
        def create_completion(**kwargs):
            token_limits.append(kwargs["max_tokens"])
            content = None if kwargs["max_tokens"] < 512 else "answer"
            finish_reason = "length" if content is None else "stop"
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=content),
                        finish_reason=finish_reason,
                    )
                ]
            )

    monkeypatch.setattr(consensus, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(consensus.settings, "nvidia_api_key", "test-key")

    assert consensus._call_nvidia("prompt", 0.2, 128) == "answer"
    assert token_limits == [128, 512]


def test_gemma_timeout_is_not_retried_in_parallel_consensus(monkeypatch):
    calls = 0

    def gemma_timeout(*args):
        nonlocal calls
        calls += 1
        raise TimeoutError("request timed out")

    monkeypatch.setitem(consensus._PROVIDERS, "groq", lambda *args: "groq answer")
    monkeypatch.setitem(consensus._PROVIDERS, "nvidia", lambda *args: "nvidia answer")
    monkeypatch.setitem(consensus._PROVIDERS, "gemma", gemma_timeout)

    answers = consensus._query_all_models("prompt", 0.2, 50)

    assert calls == 1
    assert [answer.name for answer in answers if answer.ok] == ["nvidia", "groq"]
