"""Unit tests for the executor (dispatch to simulation, KB, processor)."""
import os
from unittest.mock import MagicMock, patch

from src import executor


def test_execute_simulation_routes_to_simulation_agent(monkeypatch):
    calls = {}

    class FakeSimulationAgent:
        def __init__(self, provider=None, duration_hours=4.0, max_iterations=10):
            calls["init"] = {
                "provider": provider,
                "duration_hours": duration_hours,
                "max_iterations": max_iterations,
            }

        def run_and_report(self, initial_cooling_rate_K_per_min=15.0, ctx=None):
            calls["run"] = {"initial_cooling_rate_K_per_min": initial_cooling_rate_K_per_min}
            return ["history"], "output"

    monkeypatch.setattr(executor, "SimulationAgent", FakeSimulationAgent)
    monkeypatch.setenv("CURRENT_SIMULATION_MODE", "mock_sim_mode")
    decision = {
        "agent": "simulation",
        "mode": "structured",
        "params": {
            "initial_cooling_rate_K_per_min": 18.5,
            "duration_hours": 3.0,
            "max_iterations": 5,
        },
    }
    result = executor.execute(decision, original_prompt="ignored")
    assert result["agent"] == "simulation"
    assert result["mode"] == "structured"
    assert result["result"]["history"] == ["history"]
    assert result["result"]["output"] == "output"
    assert calls["init"]["duration_hours"] == 3.0
    assert calls["init"]["max_iterations"] == 5
    assert calls["run"]["initial_cooling_rate_K_per_min"] == 18.5


def test_execute_kb_routes_to_kb_agent(monkeypatch):
    seen = {}

    def fake_ask(query: str, ctx=None) -> str:
        seen["query"] = query
        return "ANSWER:" + query

    monkeypatch.setattr(executor.kb_agent, "ask", fake_ask)
    decision = {
        "agent": "kb",
        "mode": "pass_through",
        "params": {"query": "What is the capital of Germany?"},
    }
    result = executor.execute(decision, original_prompt=None)
    assert result["agent"] == "kb"
    assert result["mode"] == "pass_through"
    assert result["result"].startswith("ANSWER:")
    assert seen["query"] == "What is the capital of Germany?"


def test_execute_processor_routes_to_linear_orchestrator(monkeypatch):
    seen = {}

    def fake_run(input_text: str, tasks=None):
        seen["input_text"] = input_text
        seen["tasks"] = tasks
        return {"summary": "ok", "extraction": {}, "processing": {}}

    monkeypatch.setattr(executor.linear, "run", fake_run)
    decision = {
        "agent": "processor",
        "mode": "structured",
        "params": {"input_text": "Simulate nickel", "tasks": ["schema_validation"]},
    }
    result = executor.execute(decision, original_prompt=None)
    assert result["agent"] == "processor"
    assert result["mode"] == "structured"
    assert result["result"]["summary"] == "ok"
    assert seen["input_text"] == "Simulate nickel"
    assert seen["tasks"] == ["schema_validation"]


def test_execute_unknown_agent_falls_back_to_kb(monkeypatch):
    seen = {}

    def fake_ask(query: str, ctx=None) -> str:
        seen["query"] = query
        return "fallback:" + query

    monkeypatch.setattr(executor.kb_agent, "ask", fake_ask)
    decision = {
        "agent": "unknown",
        "mode": "structured",
        "params": {},
    }
    prompt = "Tell me a fun fact."
    result = executor.execute(decision, original_prompt=prompt)
    assert result["agent"] == "kb"
    assert result["mode"] == "structured"
    assert result["result"].startswith("fallback:")
    assert seen["query"] == prompt


# ---------------------------------------------------------------------------
# CURRENT_SIMULATION_MODE dispatch tests
# ---------------------------------------------------------------------------

class _FakeAgentForMode:
    """Minimal SimulationAgent stand-in that records which method was called."""

    def __init__(self, **_kwargs):
        self.run_and_report_called = False
        self.prefetch_called = False

    def run_and_report(self, initial_cooling_rate_K_per_min=15.0, ctx=None):
        self.run_and_report_called = True
        return ([], "output")

    def perform_real_simulation(self, original_prompt, ctx=None):
        self.prefetch_called = True
        self.received_prompt = original_prompt
        return ["prefetch summary"]


def test_execute_simulation_mock_sim_mode(monkeypatch):
    """CURRENT_SIMULATION_MODE=mock_sim_mode → run_and_report called, perform_real_simulation not called.

    Pre-conditions:
        CURRENT_SIMULATION_MODE env var is explicitly set to 'mock_sim_mode'.
    Post-conditions:
        - result contains 'history' and 'output' keys.
        - perform_real_simulation is never invoked.
    """
    captured = {}

    class TrackingAgent(_FakeAgentForMode):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured["agent"] = self

    monkeypatch.setattr(executor, "SimulationAgent", TrackingAgent)
    monkeypatch.setenv("CURRENT_SIMULATION_MODE", "mock_sim_mode")

    result = executor.execute({"agent": "simulation", "mode": "structured", "params": {}})
    assert result["agent"] == "simulation"
    assert "result" in result
    assert "history" in result["result"]
    assert "output" in result["result"]
    assert captured["agent"].run_and_report_called
    assert not captured["agent"].prefetch_called


def test_execute_simulation_real_sim_mode(monkeypatch):
    """CURRENT_SIMULATION_MODE=real_sim_mode → perform_real_simulation called, run_and_report not called.

    Pre-conditions:
        CURRENT_SIMULATION_MODE env var is explicitly set to 'real_sim_mode'.
    Post-conditions:
        - result contains 'list_of_sim_results' key holding a list[str].
        - run_and_report is never invoked.
        - original_prompt is forwarded to perform_real_simulation.
    """
    captured = {}

    class TrackingAgent(_FakeAgentForMode):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured["agent"] = self

    monkeypatch.setattr(executor, "SimulationAgent", TrackingAgent)
    monkeypatch.setenv("CURRENT_SIMULATION_MODE", "real_sim_mode")

    result = executor.execute(
        {"agent": "simulation", "mode": "structured", "params": {}},
        original_prompt="the-test-prompt",
    )
    assert result["agent"] == "simulation"
    assert "result" in result
    assert "list_of_sim_results" in result["result"]
    assert result["result"]["list_of_sim_results"] == ["prefetch summary"]
    assert captured["agent"].prefetch_called
    assert captured["agent"].received_prompt == "the-test-prompt"
    assert not captured["agent"].run_and_report_called


def test_execute_simulation_default_mode_is_mock(monkeypatch):
    """When CURRENT_SIMULATION_MODE is absent, the default is mock_sim_mode.

    Pre-conditions:
        CURRENT_SIMULATION_MODE is not set in the environment.
    Post-conditions:
        - run_and_report is called (optimization loop path).
        - perform_real_simulation is not called.
    """
    captured = {}

    class TrackingAgent(_FakeAgentForMode):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured["agent"] = self

    monkeypatch.setattr(executor, "SimulationAgent", TrackingAgent)
    monkeypatch.delenv("CURRENT_SIMULATION_MODE", raising=False)

    result = executor.execute({"agent": "simulation", "mode": "structured", "params": {}})
    assert "result" in result
    assert "history" in result["result"]
    assert captured["agent"].run_and_report_called
    assert not captured["agent"].prefetch_called


def test_execute_simulation_invalid_sim_mode(monkeypatch):
    """CURRENT_SIMULATION_MODE set to an unrecognised value returns an error dict.

    Pre-conditions:
        CURRENT_SIMULATION_MODE is set to 'garbage'.
    Post-conditions:
        - The returned dict contains an 'error' key.
        - The error type is 'ValueError'.
        - Neither run_and_report nor perform_real_simulation is called.
    """
    captured = {}

    class TrackingAgent(_FakeAgentForMode):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            captured["agent"] = self

    monkeypatch.setattr(executor, "SimulationAgent", TrackingAgent)
    monkeypatch.setenv("CURRENT_SIMULATION_MODE", "garbage")

    result = executor.execute({"agent": "simulation", "mode": "structured", "params": {}})
    assert "error" in result
    assert result["error"]["type"] == "ValueError"
    assert "CURRENT_SIMULATION_MODE" in result["error"]["message"]
    if "agent" in captured:
        assert not captured["agent"].run_and_report_called
        assert not captured["agent"].prefetch_called

