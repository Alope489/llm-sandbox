"""Unit tests for the AtomWrapper integration surface."""

import pytest

from src.multi.materials.atom_wrapper import AtomWrapper


def test_wrapper_raises_runtime_error_when_anthropic_key_missing(monkeypatch):
    """AtomWrapper fails fast with clear guidance when API key is missing."""
    # Prevent this unit test from loading local machine dotenv files.
    monkeypatch.setattr(AtomWrapper, "_load_environment", lambda self: None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Missing ANTHROPIC_API_KEY"):
        AtomWrapper()


def test_wrapper_uses_sonnet_fallback_model_when_env_model_missing(monkeypatch):
    """AtomWrapper defaults to Sonnet 4.6 model when env model is unset."""
    # Keep this test deterministic by bypassing filesystem-based dotenv loading.
    monkeypatch.setattr(AtomWrapper, "_load_environment", lambda self: None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-test-key")
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)

    observed = {}

    class FakeDriver:
        """Capture constructor args without running subprocess logic."""

        def __init__(self, model, temp):
            observed["model"] = model
            observed["temp"] = temp

        def run_materials_sim(self, prompt):
            return {"output": prompt, "latency_seconds": 0.0}

    monkeypatch.setattr("src.multi.materials.atom_wrapper.AtomAgentDriver", FakeDriver)

    wrapper = AtomWrapper()

    assert wrapper.model == "claude-sonnet-4-6"
    assert observed["model"] == "claude-sonnet-4-6"
    assert observed["temp"] == 0.1
