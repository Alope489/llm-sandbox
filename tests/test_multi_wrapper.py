"""Unit tests for src.multi.wrapper (mocked)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.multi.wrapper

@pytest.fixture(autouse=True)
def reset_env(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

def test_complete_with_knowledge_openai_prepends_system_message(monkeypatch):
    def mock_search(q, k):
        return [{"source": "s1", "content": "c1"}, {"source": "s2", "content": "c2"}]
    monkeypatch.setattr("src.multi.wrapper.search", mock_search)
    called_with = []
    def mock_complete(m):
        called_with.append(m)
        return "reply"
    monkeypatch.setattr("src.multi.wrapper.complete", mock_complete)
    messages = [{"role": "user", "content": "hi"}]
    reply = src.multi.wrapper.complete_with_knowledge(messages, "q")
    assert reply == "reply"
    augmented = called_with[0]
    assert augmented[0]["role"] == "system"
    assert "Relevant context:" in augmented[0]["content"]
    assert "[Source: s1] c1" in augmented[0]["content"]
    assert "[Source: s2] c2" in augmented[0]["content"]
    assert augmented[1] == messages[0]

def test_complete_with_knowledge_anthropic_injects_search_result_blocks(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    def mock_search(q, k):
        return [{"source": "s1", "title": "t1", "content": "c1"}]
    monkeypatch.setattr("src.multi.wrapper.search", mock_search)
    called_with = []
    def mock_complete(m):
        called_with.append(m)
        return "reply"
    monkeypatch.setattr("src.multi.wrapper.complete", mock_complete)
    messages = [{"role": "user", "content": "hi"}]
    src.multi.wrapper.complete_with_knowledge(messages, "q")
    augmented = called_with[0]
    user_content = augmented[0]["content"]
    assert isinstance(user_content, list)
    assert len(user_content) == 2
    assert user_content[0]["type"] == "search_result"
    assert user_content[0]["source"] == "s1"
    assert user_content[0]["title"] == "t1"
    assert user_content[0]["content"] == [{"type": "text", "text": "c1"}]
    assert user_content[1] == {"type": "text", "text": "hi"}

def test_complete_with_knowledge_calls_complete_once(monkeypatch):
    call_count = 0
    def mock_complete(m):
        nonlocal call_count
        call_count += 1
        return "reply"
    monkeypatch.setattr("src.multi.wrapper.complete", mock_complete)
    src.multi.wrapper.complete_with_knowledge([], "q")
    assert call_count == 1

def test_complete_with_knowledge_default_provider_uses_openai_format(monkeypatch):
    called_with = []
    def mock_complete(m):
        called_with.append(m)
        return ""
    monkeypatch.setattr("src.multi.wrapper.complete", mock_complete)
    src.multi.wrapper.complete_with_knowledge([{"role": "user", "content": "hi"}], "q")
    augmented = called_with[0]
    assert augmented[0]["role"] == "system"

@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="No OpenAI API key set; set OPENAI_API_KEY to run integration test",
)
def test_integration_complete_with_knowledge_openai():
    from src.multi.knowledge_base import clear, index, store_size
    clear()
    index(["The secret number is 42."])
    assert store_size() > 0
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the secret number?"}]
    response = src.multi.wrapper.complete_with_knowledge(messages, "secret number")
    assert isinstance(response, str)
    assert "42" in response

@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="No Anthropic API key set; set ANTHROPIC_API_KEY to run integration test",
)
def test_integration_complete_with_knowledge_anthropic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    from src.multi.knowledge_base import clear, index, store_size
    clear()
    index(["The secret number is 42."])
    assert store_size() > 0
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "What is the secret number?"}]
    response = src.multi.wrapper.complete_with_knowledge(messages, "secret number")
    assert isinstance(response, str)
    assert "42" in response