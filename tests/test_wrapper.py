"""Tests for src.wrapper complete() with mocked providers."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


def test_complete_default_provider_calls_openai(monkeypatch):
    from src import wrapper
    monkeypatch.setattr(wrapper, "_complete_openai", lambda msgs: "openai-reply")
    monkeypatch.setattr(wrapper, "_complete_anthropic", lambda msgs: "anthropic-reply")
    assert wrapper.complete([{"role": "user", "content": "Hi"}]) == "openai-reply"


def test_complete_anthropic_provider_calls_anthropic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    from src import wrapper
    monkeypatch.setattr(wrapper, "_complete_openai", lambda msgs: "openai-reply")
    monkeypatch.setattr(wrapper, "_complete_anthropic", lambda msgs: "anthropic-reply")
    assert wrapper.complete([{"role": "user", "content": "Hi"}]) == "anthropic-reply"


def test_complete_openai_provider_calls_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    from src import wrapper
    monkeypatch.setattr(wrapper, "_complete_openai", lambda msgs: "openai-reply")
    monkeypatch.setattr(wrapper, "_complete_anthropic", lambda msgs: "anthropic-reply")
    assert wrapper.complete([{"role": "user", "content": "Hello"}]) == "openai-reply"
