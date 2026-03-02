"""Unit tests for the coordinator agent (routing decisions, fallbacks)."""
from src import coordinator


def test_route_prompt_uses_llm_decision(monkeypatch):
    raw = '{"agent": "simulation", "mode": "structured", "params": {"initial_cooling_rate_K_per_min": 20}}'

    def fake_classify(prompt: str) -> str:
        return raw

    monkeypatch.setattr(coordinator, "_classify_with_llm", fake_classify)
    prompt = "Optimize the nickel-based superalloy heat treatment."
    decision = coordinator.route_prompt(prompt)
    assert decision["agent"] == "simulation"
    assert decision["mode"] == "structured"
    assert decision["params"]["initial_cooling_rate_K_per_min"] == 20


def test_route_prompt_invalid_json_falls_back_to_kb(monkeypatch):
    def fake_classify(prompt: str) -> str:
        return "not json at all"

    monkeypatch.setattr(coordinator, "_classify_with_llm", fake_classify)
    prompt = "What is the capital of France?"
    decision = coordinator.route_prompt(prompt)
    assert decision["agent"] == "kb"
    assert decision["mode"] == "pass_through"
    assert decision["params"]["query"] == prompt


def test_route_prompt_missing_fields_defaults_and_fills_params(monkeypatch):
    raw = '{"agent": "processor", "params": {}}'

    def fake_classify(prompt: str) -> str:
        return raw

    monkeypatch.setattr(coordinator, "_classify_with_llm", fake_classify)
    prompt = "Simulate a nickel superalloy at high temperature."
    decision = coordinator.route_prompt(prompt)
    assert decision["agent"] == "processor"
    assert decision["mode"] == "pass_through"
    assert decision["params"]["input_text"] == prompt

