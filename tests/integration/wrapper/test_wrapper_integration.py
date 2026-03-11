"""E2E integration tests for the base LLM wrapper (src.wrapper.complete). Zero mocking — real LLM only. Runs with both OpenAI and Anthropic."""
import os

import pytest

from src.wrapper import complete

pytestmark = [
    pytest.mark.integration,
    pytest.mark.wrapper,
]

PROVIDERS = [
    pytest.param("openai", id="openai", marks=pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")),
    pytest.param("anthropic", id="anthropic", marks=pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")),
]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_wrapper_integration_complete_returns_non_empty_string(provider, monkeypatch):
    """complete() with real LLM returns a non-empty assistant reply. Runs for OpenAI and Anthropic."""
    monkeypatch.setenv("LLM_PROVIDER", provider)
    if provider == "anthropic":
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    reply = complete([{"role": "user", "content": "Reply with exactly: OK"}])
    assert isinstance(reply, str)
    assert len(reply.strip()) > 0


@pytest.mark.parametrize("provider", PROVIDERS)
def test_wrapper_integration_complete_system_and_user(provider, monkeypatch):
    """complete() with system + user messages uses both (real LLM). Runs for OpenAI and Anthropic."""
    monkeypatch.setenv("LLM_PROVIDER", provider)
    if provider == "anthropic":
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    reply = complete([
        {"role": "system", "content": "You are a helpful assistant. Answer in one short word."},
        {"role": "user", "content": "What is 1+1? Reply with only the number."},
    ])
    assert isinstance(reply, str)
    assert len(reply.strip()) > 0
    assert "2" in reply or "two" in reply.lower()
