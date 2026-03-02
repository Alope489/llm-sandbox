"""Unit tests for the executor (dispatch to simulation, KB, processor)."""
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

        def run_and_report(self, initial_cooling_rate_K_per_min=15.0):
            calls["run"] = {"initial_cooling_rate_K_per_min": initial_cooling_rate_K_per_min}
            return ["history"], "output"

    monkeypatch.setattr(executor, "SimulationAgent", FakeSimulationAgent)
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

    def fake_ask(query: str) -> str:
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

    def fake_ask(query: str) -> str:
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

