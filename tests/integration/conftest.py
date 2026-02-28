"""Shared config for integration tests: path, env, and skip conditions."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# Skip integration tests when no LLM API key is set (linear and multi both need at least one).
def _has_llm_key():
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))


def _has_openai_key():
    return bool(os.environ.get("OPENAI_API_KEY"))


def _has_anthropic_key():
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


requires_llm = pytest.mark.skipif(
    not _has_llm_key(),
    reason="Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run integration tests",
)
requires_openai = pytest.mark.skipif(
    not _has_openai_key(),
    reason="Set OPENAI_API_KEY to run this integration test",
)
requires_anthropic = pytest.mark.skipif(
    not _has_anthropic_key(),
    reason="Set ANTHROPIC_API_KEY to run this integration test",
)
