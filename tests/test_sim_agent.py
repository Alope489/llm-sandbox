"""Tests for simulation agent and format_simulation_output (chat-friendly output).

All agent tests are end-to-end: they use the real LLM (no mocks). Skip when no API key.
Format tests exercise the formatter only (no agent/LLM).
Unit tests for Option A pre-computation are mocked.
"""
import os
from unittest.mock import MagicMock, patch

import pytest

from tests.telemetry_helpers import assert_openai_server_latency

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
@patch("src.wrapper.complete_with_tools")
def test_prefetch_called_when_use_tools_true(mock_cwt, mock_sim):
    """run_optimization_loop never calls complete_with_tools regardless of use_tools.

    The pre-computation phase was removed from the loop body; use_tools is now
    a no-op deprecated parameter.  The call that previously triggered
    perform_real_simulation must now come from outside (via executor dispatch).
    """
    with patch("src.multi.sim.agent.SimulationAgent._call_openai", return_value="15.0"):
        agent = SimulationAgent(max_iterations=1)
        agent.run_optimization_loop(use_tools=True)
    mock_cwt.assert_not_called()


@patch("src.multi.sim.agent.SimulationAgent.run_simulation", return_value=(420.0, True))
@patch("src.wrapper.complete_with_tools")
def test_prefetch_not_called_when_use_tools_false(mock_cwt, mock_sim):
    """use_tools=False: complete_with_tools is never invoked — backward compatible."""
    with patch("src.multi.sim.agent.SimulationAgent._call_openai", return_value="15.0"):
        agent = SimulationAgent(max_iterations=1)
        agent.run_optimization_loop(use_tools=False)
    mock_cwt.assert_not_called()


@patch("src.multi.sim.agent.SimulationAgent.run_simulation", return_value=(420.0, True))
@patch("src.multi.sim.agent.SimulationAgent.perform_real_simulation")
def test_run_optimization_loop_never_calls_prefetch(mock_prefetch, mock_sim):
    """perform_real_simulation is never invoked by run_optimization_loop regardless of use_tools.

    Pre-conditions:
        - run_optimization_loop is called with both use_tools=True and use_tools=False.
    Post-conditions:
        - perform_real_simulation is never called by the loop in either case.
    """
    with patch("src.multi.sim.agent.SimulationAgent._call_openai", return_value="15.0"):
        agent = SimulationAgent(max_iterations=1)
        agent.run_optimization_loop(use_tools=True)
        agent.run_optimization_loop(use_tools=False)
    mock_prefetch.assert_not_called()


def test_system_prompt_returns_base_system_prompt():
    """_system_prompt() always returns the base SYSTEM_PROMPT constant unchanged.

    Since the _current_sim_results instance attribute was removed in favour of a
    local variable returned from perform_real_simulation, _system_prompt() no
    longer injects simulation results and always returns the module-level constant.

    Pre-conditions:
        SimulationAgent is constructed with default arguments.
    Post-conditions:
        - _system_prompt() returns exactly SYSTEM_PROMPT.
    """
    agent = SimulationAgent()
    assert agent._system_prompt() == SYSTEM_PROMPT


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


# ---------------------------------------------------------------------------
# Telemetry ctx attribution tests (real API — skip when key absent)
# ---------------------------------------------------------------------------

import os as _os
import dataclasses as _dc

_skip_no_llm = pytest.mark.skipif(
    not _os.environ.get("OPENAI_API_KEY") and not _os.environ.get("ANTHROPIC_API_KEY"),
    reason="Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run ctx attribution tests",
)


@_skip_no_llm
def test_run_optimization_loop_ctx_attribution():
    """ctx records from a 1-iteration loop carry the correct pipeline and run_id."""
    from src.llm_pipeline_telemetry import CallContext

    ctx = CallContext(pipeline="multi_agent")
    agent = SimulationAgent(max_iterations=1)
    agent.run_optimization_loop(initial_cooling_rate_K_per_min=15.0, ctx=ctx)

    llm_records = [r for r in ctx.records if r.get("record_type") == "llm_call"]
    assert len(llm_records) >= 1
    for rec in llm_records:
        assert rec["pipeline"] == "multi_agent"
        assert rec["run_id"] == ctx.run_id
        assert rec["agent"] == "sim_agent"
        assert rec["input_tokens"] > 0
        assert rec["output_tokens"] > 0
        assert rec["client_elapsed_ms"] > 0
        assert "call_start_ts" in rec
        assert "call_end_ts" in rec
        assert_openai_server_latency(rec)  # openai-processing-ms must be a positive int


def test_predefined_sim_calls_is_2d_tuple():
    """_PREDEFINED_SIM_CALLS is a tuple[tuple[str, str], ...] — the outer container
    and every inner element must be immutable tuples with exactly two str values.

    Pre-conditions:
        _PREDEFINED_SIM_CALLS is imported from src.multi.sim.agent.
    Post-conditions:
        - isinstance(_PREDEFINED_SIM_CALLS, tuple) is True.
        - Every element is a tuple of length 2.
        - Both values in each inner tuple are str.
        - The constant has exactly 6 entries.
    """
    from src.multi.sim.agent import _PREDEFINED_SIM_CALLS

    assert isinstance(_PREDEFINED_SIM_CALLS, tuple), "outer container must be a tuple"
    assert len(_PREDEFINED_SIM_CALLS) == 6
    for entry in _PREDEFINED_SIM_CALLS:
        assert isinstance(entry, tuple), f"inner entry {entry!r} must be a tuple"
        assert len(entry) == 2, f"inner entry {entry!r} must have exactly 2 elements"
        composition, supercell_size = entry
        assert isinstance(composition, str), f"composition {composition!r} must be str"
        assert isinstance(supercell_size, str), f"supercell_size {supercell_size!r} must be str"


def test_perform_real_simulation_raises_for_non_openai():
    """perform_real_simulation raises RuntimeError when provider is not openai.

    Pre-conditions:
        SimulationAgent is constructed with provider="anthropic".
    Post-conditions:
        - RuntimeError is raised before any LLM call is made.
        - The error message references "openai".
    """
    agent = SimulationAgent(provider="anthropic")
    with pytest.raises(RuntimeError, match="openai"):
        agent.perform_real_simulation("some prompt")


# ---------------------------------------------------------------------------
# Schema tests: _build_tool_message_for_sim_param_api_request_openAI
# Pure function — no mocking needed.
# ---------------------------------------------------------------------------

import json as _json

from src.multi.sim.agent import _PREDEFINED_SIM_CALLS as _PSC

# (prompt_length, expected_n)  — covers all 6 remainder classes plus len=0 boundary
_SCHEMA_CASES = [
    (0,  1),   # boundary: empty string, 0 % 6 + 1 = 1
    (6,  1),   # remainder 0, non-zero multiple: 6 % 6 + 1 = 1
    (7,  2),   # remainder 1
    (8,  3),   # remainder 2
    (9,  4),   # remainder 3
    (10, 5),   # remainder 4
    (5,  6),   # remainder 5
]


@pytest.mark.parametrize("prompt_length,expected_n", _SCHEMA_CASES)
def test_build_tool_message_schema(prompt_length: int, expected_n: int) -> None:
    """_build_tool_message_for_sim_param_api_request_openAI produces correct schema for each remainder class.

    Verifies that the returned (messages, tools) pair encodes exactly ``expected_n``
    as both minItems and maxItems, names the tool correctly, and embeds the full
    ``_PREDEFINED_SIM_CALLS`` list in the user message.  Pure function — zero mocking.

    Args:
        prompt_length: Length of the synthetic prompt ``"x" * prompt_length``.
        expected_n: Expected ``len(prompt) % 6 + 1`` for this case.

    Pre-conditions:
        ``SimulationAgent`` is constructed with ``provider="openai"``.
    Post-conditions:
        - ``messages[0]["role"] == "system"`` and content mentions ``expected_n``.
        - ``messages[1]["content"]`` contains ``json.dumps(_PREDEFINED_SIM_CALLS, indent=2)``.
        - ``tools[0]["name"] == "select_first_pairs"`` and ``tools[0]["strict"] is True``.
        - ``minItems == maxItems == expected_n``.

    Complexity:
        O(1) — no I/O.
    """
    agent = SimulationAgent(provider="openai")
    prompt = "x" * prompt_length
    messages, tools = agent._build_tool_message_for_sim_param_api_request_openAI(prompt)

    assert messages[0]["role"] == "system"
    assert str(expected_n) in messages[0]["content"]

    expected_json_fragment = _json.dumps(_PSC, indent=2)
    assert expected_json_fragment in messages[1]["content"]

    tool = tools[0]
    assert tool["name"] == "select_first_pairs"
    assert tool["strict"] is True

    pairs_schema = tool["parameters"]["properties"]["selected_pairs"]
    assert pairs_schema["minItems"] == expected_n
    assert pairs_schema["maxItems"] == expected_n


# ---------------------------------------------------------------------------
# Mock-based unit tests: _get_elastic_constants_params_from_LLM
# 3a: error-path (no function_call)
# 3b: happy-path parsing (mocked function_call with known arguments)
# Note: 3c (provider guard) is already covered by
#       test_perform_real_simulation_raises_for_non_openai above.
# ---------------------------------------------------------------------------


def _make_mock_openai_client(output_items: list) -> MagicMock:
    """Build a MagicMock that mimics get_openai_client() for Responses API calls.

    Args:
        output_items: List of mock items to place in response.output.

    Returns:
        A MagicMock whose ``.responses.create()`` returns a response with
        ``.output`` set to ``output_items``.
    """
    mock_response = MagicMock()
    mock_response.output = output_items
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response
    return mock_client


def test_get_elastic_constants_params_no_function_call_raises() -> None:
    """_get_elastic_constants_params_from_LLM raises RuntimeError when API returns no function_call.

    This path is unreachable via the real API under tool_choice='required'; mocking is
    the only way to exercise the ``tool_call is None`` branch.

    Pre-conditions:
        ``get_openai_client`` is patched to return a response whose ``.output``
        contains only a message item (``type == "message"``), not a function_call.
    Post-conditions:
        - ``RuntimeError`` is raised.
        - The message contains ``"no function_call tool invocation"``.

    Complexity:
        O(1) — no real I/O.
    """
    message_item = MagicMock()
    message_item.type = "message"

    with patch("src.multi.sim.agent.get_openai_client", return_value=_make_mock_openai_client([message_item])):
        agent = SimulationAgent(provider="openai")
        with pytest.raises(RuntimeError) as exc_info:
            agent._get_elastic_constants_params_from_LLM("any prompt")
        assert "no function_call tool invocation" in str(exc_info.value)


def test_get_elastic_constants_params_happy_path_parsing() -> None:
    """_get_elastic_constants_params_from_LLM correctly parses a mocked function_call response.

    Isolates the json.loads(tool_call.arguments) + tuple-comprehension path so
    regressions in parsing are caught without a real API call.

    Pre-conditions:
        ``get_openai_client`` is patched to return a response whose ``.output``
        contains one function_call item with
        ``arguments = json.dumps({"selected_pairs": [["Al", "3"], ["Cu", "3"]]})``.
    Post-conditions:
        - Return value equals ``(("Al", "3"), ("Cu", "3"))``.
        - Each element of the outer tuple is itself a tuple of two strings.

    Complexity:
        O(1) — no real I/O.
    """
    function_call_item = MagicMock()
    function_call_item.type = "function_call"
    function_call_item.arguments = _json.dumps(
        {"selected_pairs": [["Al", "3"], ["Cu", "3"]]}
    )

    with patch("src.multi.sim.agent.get_openai_client", return_value=_make_mock_openai_client([function_call_item])):
        agent = SimulationAgent(provider="openai")
        result = agent._get_elastic_constants_params_from_LLM("x" * 7)  # len=7 → n=2

    assert result == (("Al", "3"), ("Cu", "3"))
    for pair in result:
        assert isinstance(pair, tuple)
        assert len(pair) == 2
        assert all(isinstance(v, str) for v in pair)
