"""E2E integration tests for the base LLM wrapper (src.wrapper.complete). Zero mocking — real LLM only."""
import os

import pytest

from src.wrapper import complete

pytestmark = [
    pytest.mark.integration,
    pytest.mark.wrapper,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"),
        reason="Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run wrapper integration tests",
    ),
]


def test_wrapper_integration_complete_returns_non_empty_string():
    """complete() with real LLM returns a non-empty assistant reply."""
    reply = complete([{"role": "user", "content": "Reply with exactly: OK"}])
    assert isinstance(reply, str)
    assert len(reply.strip()) > 0


def test_wrapper_integration_complete_system_and_user():
    """complete() with system + user messages uses both (real LLM)."""
    reply = complete([
        {"role": "system", "content": "You are a helpful assistant. Answer in one short word."},
        {"role": "user", "content": "What is 1+1? Reply with only the number."},
    ])
    assert isinstance(reply, str)
    assert len(reply.strip()) > 0
    assert "2" in reply or "two" in reply.lower()
