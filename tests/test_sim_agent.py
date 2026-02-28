"""Tests for simulation agent and format_simulation_output (chat-friendly output).

All agent tests are end-to-end: they use the real LLM (no mocks). Skip when no API key.
Format tests exercise the formatter only (no agent/LLM).
"""
import os

import pytest

from src.multi.sim.agent import (
    SimulationAgent,
    format_simulation_output,
    HistoryEntry,
)

_skip_no_llm = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"),
    reason="Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run agent e2e tests",
)


def test_format_simulation_output_empty():
    assert format_simulation_output([]) == "Simulation run: no iterations."


def test_format_simulation_output_one_step():
    history: list[HistoryEntry] = [(15.0, 420.5, True)]
    out = format_simulation_output(history)
    assert "Iteration 1:" in out
    assert "15" in out
    assert "420.50" in out
    assert "success=True" in out
    assert "Best (successful):" in out
    assert "420.50" in out


def test_format_simulation_output_with_step_lines():
    history: list[HistoryEntry] = [(15.0, 420.5, True), (20.0, 380.0, False)]
    step_lines = [
        "  Iteration 1: cooling_rate=15.0 K/min -> yield_strength=420.50 MPa, success=True",
        "  Iteration 2: cooling_rate=20.0 K/min -> yield_strength=380.00 MPa, success=False",
    ]
    out = format_simulation_output(history, step_lines=step_lines)
    assert step_lines[0] in out
    assert step_lines[1] in out
    assert "Best (successful): cooling_rate=15.0" in out


def test_format_simulation_output_no_successful_runs():
    history: list[HistoryEntry] = [(50.0, 300.0, False), (60.0, 290.0, False)]
    out = format_simulation_output(history)
    assert "No successful runs" in out


@_skip_no_llm
@pytest.mark.parametrize("max_iterations", [1, 2])
def test_run_optimization_loop_calls_on_step(max_iterations):
    """E2E: on_step callback is invoked each iteration with (iteration, rate, y_MPa, success)."""
    seen = []

    def record(iteration: int, rate: float, y_MPa: float, success: bool):
        seen.append((iteration, rate, y_MPa, success))

    agent = SimulationAgent(max_iterations=max_iterations)
    agent.run_optimization_loop(initial_cooling_rate_K_per_min=15.0, on_step=record)
    assert len(seen) == max_iterations
    assert seen[0][0] == 1
    assert isinstance(seen[0][1], (int, float)) and seen[0][1] >= 0
    assert isinstance(seen[0][2], (int, float))
    assert isinstance(seen[0][3], bool)


@_skip_no_llm
def test_run_and_report_returns_history_and_output_string():
    """E2E: run_and_report returns correct shape and output format."""
    agent = SimulationAgent(max_iterations=2)
    history, output = agent.run_and_report(initial_cooling_rate_K_per_min=15.0)
    assert len(history) == 2
    for rate, y_MPa, success in history:
        assert isinstance(rate, (int, float)) and rate >= 0
        assert isinstance(y_MPa, (int, float))
        assert isinstance(success, bool)
    assert "Simulation run:" in output
    assert "Iteration 1:" in output
    assert "Iteration 2:" in output
    assert "Best (successful):" in output or "No successful runs" in output


@_skip_no_llm
def test_simulation_agent_integration_real_llm():
    """E2E: full run with real LLM suggestions, simulation, and formatted output."""
    agent = SimulationAgent(max_iterations=3)
    history, output = agent.run_and_report(initial_cooling_rate_K_per_min=15.0)
    assert len(history) == 3
    for rate, y_MPa, success in history:
        assert isinstance(rate, (int, float)) and rate >= 0
        assert isinstance(y_MPa, (int, float))
        assert isinstance(success, bool)
    assert "Simulation run:" in output
    assert "Iteration 1:" in output
    assert "Iteration 3:" in output
    assert "Best (successful):" in output or "No successful runs" in output
