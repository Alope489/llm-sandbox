"""Tests for simulation agent and format_simulation_output (chat-friendly output).

All agent tests are end-to-end: they use the real LLM (no mocks). Skip when no API key.
Format tests exercise the formatter only (no agent/LLM).
Unit tests for Option A pre-computation are mocked.
"""
import os
from unittest.mock import MagicMock, patch

import pytest

from src.multi.sim.agent import (
    SimulationAgent,
    SYSTEM_PROMPT,
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


# ---------------------------------------------------------------------------
# Option A pre-computation unit tests (all mocked)
# ---------------------------------------------------------------------------

@patch("src.multi.sim.agent.SimulationAgent.run_simulation", return_value=(420.0, True))
@patch("src.wrapper.complete_with_tools", return_value="Ni C11=247 GPa")
def test_prefetch_called_when_use_tools_true(mock_cwt, mock_sim):
    """run_optimization_loop with use_tools=True calls complete_with_tools once before the loop."""
    with patch("src.multi.sim.agent.SimulationAgent._call_openai", return_value="15.0"):
        agent = SimulationAgent(max_iterations=1)
        agent.run_optimization_loop(use_tools=True)
    mock_cwt.assert_called_once()


@patch("src.multi.sim.agent.SimulationAgent.run_simulation", return_value=(420.0, True))
@patch("src.wrapper.complete_with_tools")
def test_prefetch_not_called_when_use_tools_false(mock_cwt, mock_sim):
    """Default use_tools=False means complete_with_tools is never invoked — backward compatible."""
    with patch("src.multi.sim.agent.SimulationAgent._call_openai", return_value="15.0"):
        agent = SimulationAgent(max_iterations=1)
        agent.run_optimization_loop(use_tools=False)
    mock_cwt.assert_not_called()


def test_tool_context_injected_into_system_prompt():
    """_system_prompt() returns SYSTEM_PROMPT unchanged when _tool_context is empty,
    and appends the context string when set."""
    agent = SimulationAgent()
    assert agent._system_prompt() == SYSTEM_PROMPT

    agent._tool_context = "Ni C11=247 GPa, C44=122 GPa"
    result = agent._system_prompt()
    assert SYSTEM_PROMPT in result
    assert "Ni C11=247 GPa" in result
    assert result != SYSTEM_PROMPT


@patch("src.wrapper.complete_with_tools", return_value="C11=247 GPa for Ni")
def test_ask_with_tools_returns_string(mock_cwt):
    """ask_with_tools() returns the string produced by complete_with_tools."""
    agent = SimulationAgent()
    result = agent.ask_with_tools("What are elastic constants of Ni?")
    assert isinstance(result, str)
    assert result == "C11=247 GPa for Ni"
    mock_cwt.assert_called_once()


@patch("src.multi.sim.agent.SimulationAgent.run_optimization_loop")
def test_run_and_report_passes_use_tools(mock_loop):
    """run_and_report forwards use_tools=True to run_optimization_loop."""
    mock_loop.return_value = []
    agent = SimulationAgent()
    agent.run_and_report(use_tools=True)
    _, kwargs = mock_loop.call_args
    assert kwargs.get("use_tools") is True
