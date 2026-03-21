"""Shared assertion helpers for telemetry tests.

Purpose:
    Provides reusable, rigorous assertion utilities for verifying llm_call
    telemetry records emitted by the LLM pipeline. Separates test utility
    functions from pytest configuration in tests/conftest.py.

Dependencies:
    typing (stdlib only) — no production src/ imports.

Pillar compliance:
    - Pillar 2: Single source of truth for the server-latency invariant;
      ensures consistent, rigorous enforcement across all OpenAI call paths.
    - Pillar 4: DRY — future OpenAI paths add one call rather than duplicating
      two inline assertions.
    - Pillar 7: Failure messages name the actual value found and point at the
      root cause, making CI failures self-diagnosing.
"""
from typing import Any


def assert_openai_server_latency(rec: dict[str, Any]) -> None:
    """Assert that an llm_call record carries a valid OpenAI server latency value.

    Enforces the research invariant: every OpenAI LLM call must populate
    provider_server_latency_ms from the openai-processing-ms response header as a
    strictly positive plain integer.  This is required for
    is_provider_server_latency_complete to be True on pipeline_outcome_and_stats
    records for pure-OpenAI pipeline runs.

    Uses type() is int rather than isinstance(..., int) to explicitly reject bool,
    which is a Python subclass of int and would otherwise pass silently.

    Args:
        rec: A llm_call telemetry record dict as stored in CallContext.records.

    Raises:
        AssertionError: If rec is not a dict, if provider_server_latency_ms is
            absent, None, non-integer (including bool), or <= 0.

    Examples:
        assert_openai_server_latency({"provider_server_latency_ms": 312})  # passes
        assert_openai_server_latency({"provider_server_latency_ms": None}) # fails
        assert_openai_server_latency({"provider_server_latency_ms": True}) # fails
    """
    assert isinstance(rec, dict), (
        f"expected a llm_call record dict, got {type(rec).__name__}: {rec!r}"
    )
    assert "provider_server_latency_ms" in rec, (
        f"provider_server_latency_ms key missing from record "
        f"(agent={rec.get('agent', '?')}, span={rec.get('span', '?')})"
    )
    psm = rec["provider_server_latency_ms"]
    assert type(psm) is int, (
        f"provider_server_latency_ms must be a plain int, got "
        f"{type(psm).__name__}={psm!r} — openai-processing-ms header was not "
        f"extracted or was set to the wrong type "
        f"(agent={rec.get('agent', '?')}, span={rec.get('span', '?')})"
    )
    assert psm > 0, (
        f"provider_server_latency_ms must be > 0, got {psm} — "
        f"openai-processing-ms header returned zero or negative "
        f"(agent={rec.get('agent', '?')}, span={rec.get('span', '?')})"
    )


__all__ = ["assert_openai_server_latency"]
