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


def _make_mock_openai_client(output_items: list, *, processing_ms: str | None = "250") -> MagicMock:
    """Build a MagicMock that mimics get_openai_client() for Responses API calls via with_raw_response.

    The production code calls ``client.with_raw_response.responses.create(...)``
    which returns a raw response object.  Callers then call ``raw.parse()`` to get
    the parsed response and ``raw.headers.get(...)`` to read HTTP headers.

    Args:
        output_items: List of mock items to place in ``parsed_response.output``.
        processing_ms: Value returned by ``raw.headers.get("openai-processing-ms")``.
            Defaults to ``"250"``. Pass ``None`` to simulate an absent header.

    Returns:
        A MagicMock whose ``.with_raw_response.responses.create()`` returns a raw
        object whose ``.parse()`` yields a response with ``.output`` set to
        ``output_items`` and ``.headers.get(...)`` returning ``processing_ms``.
    """
    parsed_response = MagicMock()
    parsed_response.output = output_items
    parsed_response.usage = MagicMock()
    parsed_response.usage.input_tokens = 10
    parsed_response.usage.output_tokens = 5

    mock_raw = MagicMock()
    mock_raw.parse.return_value = parsed_response
    mock_raw.headers.get.return_value = processing_ms

    mock_client = MagicMock()
    mock_client.with_raw_response.responses.create.return_value = mock_raw
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
        assert "API contract violation" in str(exc_info.value)


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


# ---------------------------------------------------------------------------
# Group A: _get_elastic_constants_params_from_LLM — the llm_call record
# ---------------------------------------------------------------------------


def test_get_elastic_constants_params_llm_call_record() -> None:
    """_get_elastic_constants_params_from_LLM appends a correctly-labelled llm_call record.

    Pre-conditions:
        get_openai_client is patched; header "openai-processing-ms" returns "500";
        usage has known token counts; a valid function_call item is returned.
    Post-conditions:
        - Exactly one llm_call record in ctx.records.
        - provider_server_latency_ms == 500.
        - client_elapsed_ms >= 0.
        - call_start_ts and call_end_ts are present.
        - agent == "sim_agent", span == "real_sim_param_select", status == "ok".

    Complexity:
        O(1) — no real I/O.
    """
    from src.llm_pipeline_telemetry import CallContext

    function_call_item = MagicMock()
    function_call_item.type = "function_call"
    function_call_item.arguments = _json.dumps({"selected_pairs": [["Al", "3"]]})

    mock_client = _make_mock_openai_client([function_call_item], processing_ms="500")
    mock_client.with_raw_response.responses.create.return_value.parse.return_value.usage.input_tokens = 20
    mock_client.with_raw_response.responses.create.return_value.parse.return_value.usage.output_tokens = 8

    ctx = CallContext(pipeline="test_pipeline")
    with patch("src.multi.sim.agent.get_openai_client", return_value=mock_client):
        agent = SimulationAgent(provider="openai")
        agent._get_elastic_constants_params_from_LLM("x" * 7, ctx=ctx)

    llm_records = [r for r in ctx.records if r.get("record_type") == "llm_call"]
    assert len(llm_records) == 1
    rec = llm_records[0]
    assert rec["provider_server_latency_ms"] == 500
    assert rec["client_elapsed_ms"] >= 0
    assert "call_start_ts" in rec
    assert "call_end_ts" in rec
    assert rec["agent"] == "sim_agent"
    assert rec["span"] == "real_sim_param_select"
    assert rec["status"] == "ok"


def test_get_elastic_constants_params_llm_call_no_server_latency() -> None:
    """_get_elastic_constants_params_from_LLM sets provider_server_latency_ms=None when header absent.

    Pre-conditions:
        get_openai_client is patched; header "openai-processing-ms" returns None.
    Post-conditions:
        - provider_server_latency_ms is None in the llm_call record.

    Complexity:
        O(1) — no real I/O.
    """
    from src.llm_pipeline_telemetry import CallContext

    function_call_item = MagicMock()
    function_call_item.type = "function_call"
    function_call_item.arguments = _json.dumps({"selected_pairs": [["Al", "3"]]})

    mock_client = _make_mock_openai_client([function_call_item], processing_ms=None)
    ctx = CallContext(pipeline="test_pipeline")
    with patch("src.multi.sim.agent.get_openai_client", return_value=mock_client):
        agent = SimulationAgent(provider="openai")
        agent._get_elastic_constants_params_from_LLM("x" * 7, ctx=ctx)

    llm_records = [r for r in ctx.records if r.get("record_type") == "llm_call"]
    assert len(llm_records) == 1
    assert llm_records[0]["provider_server_latency_ms"] is None


def test_get_elastic_constants_params_llm_call_error_path() -> None:
    """_get_elastic_constants_params_from_LLM emits status='error' record when API raises.

    Pre-conditions:
        get_openai_client is patched; with_raw_response.responses.create raises RuntimeError.
    Post-conditions:
        - Exactly one llm_call record with status='error', input_tokens=0, output_tokens=0.
        - client_elapsed_ms >= 0.
        - The original exception propagates.

    Complexity:
        O(1) — no real I/O.
    """
    from src.llm_pipeline_telemetry import CallContext

    mock_client = MagicMock()
    mock_client.with_raw_response.responses.create.side_effect = RuntimeError("network failure")

    ctx = CallContext(pipeline="test_pipeline")
    with patch("src.multi.sim.agent.get_openai_client", return_value=mock_client):
        agent = SimulationAgent(provider="openai")
        with pytest.raises(RuntimeError, match="network failure"):
            agent._get_elastic_constants_params_from_LLM("any prompt", ctx=ctx)

    llm_records = [r for r in ctx.records if r.get("record_type") == "llm_call"]
    assert len(llm_records) == 1
    rec = llm_records[0]
    assert rec["status"] == "error"
    assert rec["input_tokens"] == 0
    assert rec["output_tokens"] == 0
    assert rec["client_elapsed_ms"] >= 0


def test_get_elastic_constants_params_llm_call_real_api() -> None:
    """_get_elastic_constants_params_from_LLM emits a complete llm_call record against the real API.

    Requires OPENAI_API_KEY to be set; fails loudly without it.  This is the only test
    that verifies the openai-processing-ms header is received from the real OpenAI API.

    Pre-conditions:
        OPENAI_API_KEY is set in the environment.
    Post-conditions:
        - Exactly one llm_call record with status='ok'.
        - assert_openai_server_latency(rec) passes.

    Complexity:
        O(1) calls — one live HTTP request.
    """
    assert os.environ.get("OPENAI_API_KEY"), (
        "OPENAI_API_KEY must be set to run this test — "
        "it verifies the openai-processing-ms header is received from the real API"
    )
    from src.llm_pipeline_telemetry import CallContext

    ctx = CallContext(pipeline="test_real_api")
    agent = SimulationAgent(provider="openai")
    agent._get_elastic_constants_params_from_LLM("Compute elastic constants for Al", ctx=ctx)

    llm_records = [r for r in ctx.records if r.get("record_type") == "llm_call"]
    assert len(llm_records) == 1
    rec = llm_records[0]
    assert rec["status"] == "ok"
    assert_openai_server_latency(rec)


def test_get_elastic_constants_params_ctx_none() -> None:
    """_get_elastic_constants_params_from_LLM with ctx=None produces no records and no crash.

    Pre-conditions:
        get_openai_client is patched; ctx is None.
    Post-conditions:
        - No exception raised.
        - Return value equals the expected tuple of pairs.

    Complexity:
        O(1) — no real I/O.
    """
    function_call_item = MagicMock()
    function_call_item.type = "function_call"
    function_call_item.arguments = _json.dumps({"selected_pairs": [["Al", "3"]]})

    mock_client = _make_mock_openai_client([function_call_item])
    with patch("src.multi.sim.agent.get_openai_client", return_value=mock_client):
        agent = SimulationAgent(provider="openai")
        result = agent._get_elastic_constants_params_from_LLM("x" * 7, ctx=None)

    assert result == (("Al", "3"),)


# ---------------------------------------------------------------------------
# Group B: perform_real_simulation — the tool_execution records
# ---------------------------------------------------------------------------


def test_perform_real_simulation_ctx_attribution() -> None:
    """perform_real_simulation appends correctly-labelled tool_execution records.

    Pre-conditions:
        _get_elastic_constants_params_from_LLM is patched to return 2 pairs.
        compute_elastic_constants_tool is patched to return a successful result dict.
    Post-conditions:
        - Exactly 2 tool_execution records with spans "real_sim_docker_1" / "real_sim_docker_2".
        - tool_execution_ms >= 0 on both.
        - tool_internal_runtime_ms == pytest.approx(500.0) on both.
        - call_start_ts and call_end_ts present on both.
        - status == "ok" on both.

    Complexity:
        O(1) — no real I/O.
    """
    from src.llm_pipeline_telemetry import CallContext

    ctx = CallContext(pipeline="test_pipeline")
    mock_tool_result = {"status": "ok", "runtime_seconds": 0.5, "C11": 100.0, "C12": 60.0, "C44": 30.0}

    with (
        patch(
            "src.multi.sim.agent.SimulationAgent._get_elastic_constants_params_from_LLM",
            return_value=(("Al", "3"), ("Cu", "3")),
        ),
        patch(
            "src.tools.elastic_constants_lammps.host_wrapper.compute_elastic_constants_tool",
            return_value=mock_tool_result,
        ),
    ):
        agent = SimulationAgent(provider="openai")
        agent.perform_real_simulation("any prompt", ctx=ctx)

    tool_records = [r for r in ctx.records if r.get("record_type") == "tool_execution"]
    assert len(tool_records) == 2
    assert tool_records[0]["span"] == "real_sim_docker_1"
    assert tool_records[1]["span"] == "real_sim_docker_2"
    for rec in tool_records:
        assert rec["tool_execution_ms"] >= 0
        assert rec["tool_internal_runtime_ms"] == pytest.approx(500.0)
        assert "call_start_ts" in rec
        assert "call_end_ts" in rec
        assert rec["status"] == "ok"


def test_perform_real_simulation_tool_execution_no_runtime_seconds() -> None:
    """perform_real_simulation omits tool_internal_runtime_ms when runtime_seconds absent.

    Pre-conditions:
        compute_elastic_constants_tool returns {"status": "ok"} with no runtime_seconds.
    Post-conditions:
        - tool_internal_runtime_ms is not present in the tool_execution record.

    Complexity:
        O(1) — no real I/O.
    """
    from src.llm_pipeline_telemetry import CallContext

    ctx = CallContext(pipeline="test_pipeline")
    with (
        patch(
            "src.multi.sim.agent.SimulationAgent._get_elastic_constants_params_from_LLM",
            return_value=(("Al", "3"),),
        ),
        patch(
            "src.tools.elastic_constants_lammps.host_wrapper.compute_elastic_constants_tool",
            return_value={"status": "ok"},
        ),
    ):
        agent = SimulationAgent(provider="openai")
        agent.perform_real_simulation("x", ctx=ctx)

    tool_records = [r for r in ctx.records if r.get("record_type") == "tool_execution"]
    assert len(tool_records) == 1
    assert "tool_internal_runtime_ms" not in tool_records[0]


def test_perform_real_simulation_tool_execution_error_path() -> None:
    """perform_real_simulation records status='error' when tool returns error dict.

    Pre-conditions:
        compute_elastic_constants_tool returns {"status": "error"}.
    Post-conditions:
        - tool_execution record has status == "error".
        - tool_internal_runtime_ms is not present.

    Complexity:
        O(1) — no real I/O.
    """
    from src.llm_pipeline_telemetry import CallContext

    ctx = CallContext(pipeline="test_pipeline")
    with (
        patch(
            "src.multi.sim.agent.SimulationAgent._get_elastic_constants_params_from_LLM",
            return_value=(("Al", "3"),),
        ),
        patch(
            "src.tools.elastic_constants_lammps.host_wrapper.compute_elastic_constants_tool",
            return_value={"status": "error"},
        ),
    ):
        agent = SimulationAgent(provider="openai")
        agent.perform_real_simulation("x", ctx=ctx)

    tool_records = [r for r in ctx.records if r.get("record_type") == "tool_execution"]
    assert len(tool_records) == 1
    assert tool_records[0]["status"] == "error"
    assert "tool_internal_runtime_ms" not in tool_records[0]


def test_perform_real_simulation_ctx_none() -> None:
    """perform_real_simulation with ctx=None produces no records and returns correct results.

    Pre-conditions:
        ctx is None; compute_elastic_constants_tool is patched.
    Post-conditions:
        - No exception raised.
        - Return value is a list of JSON strings.

    Complexity:
        O(1) — no real I/O.
    """
    mock_tool_result = {"status": "ok", "C11": 100.0}
    with (
        patch(
            "src.multi.sim.agent.SimulationAgent._get_elastic_constants_params_from_LLM",
            return_value=(("Al", "3"),),
        ),
        patch(
            "src.tools.elastic_constants_lammps.host_wrapper.compute_elastic_constants_tool",
            return_value=mock_tool_result,
        ),
    ):
        agent = SimulationAgent(provider="openai")
        result = agent.perform_real_simulation("x", ctx=None)

    assert isinstance(result, list)
    assert len(result) == 1
    parsed = _json.loads(result[0])
    assert parsed["status"] == "ok"


# ---------------------------------------------------------------------------
# Group C: get_llm_suggestion retry span labeling
# ---------------------------------------------------------------------------


def _make_mock_openai_chat_client(responses: list[str]) -> MagicMock:
    """Build a MagicMock that mimics get_openai_client() for Chat Completions via with_raw_response.

    The production ``_call_openai`` calls
    ``client.with_raw_response.chat.completions.create(...)`` and then reads
    ``.parse().usage``, ``.parse().choices[0].message.content``, and
    ``.headers.get("openai-processing-ms")``.

    Args:
        responses: Sequence of content strings to return on successive calls.
            Each call to ``create`` consumes the next entry.

    Returns:
        A MagicMock whose ``with_raw_response.chat.completions.create`` returns
        a raw mock that produces the appropriate parsed shape.
    """
    call_index = {"n": 0}

    def _create_side_effect(*args, **kwargs):
        idx = call_index["n"]
        call_index["n"] += 1
        content = responses[idx] if idx < len(responses) else responses[-1]

        mock_message = MagicMock()
        mock_message.content = content
        mock_message.refusal = None

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 3

        mock_parsed = MagicMock()
        mock_parsed.usage = mock_usage
        mock_parsed.choices = [mock_choice]

        mock_raw = MagicMock()
        mock_raw.parse.return_value = mock_parsed
        mock_raw.headers.get.return_value = "100"
        return mock_raw

    mock_client = MagicMock()
    mock_client.with_raw_response.chat.completions.create.side_effect = _create_side_effect
    return mock_client


def test_get_llm_suggestion_retry_span_labels() -> None:
    """get_llm_suggestion retry receives a unique span with _retry_ suffix.

    Patches get_openai_client at the HTTP level so the real _call_openai and
    log_llm_call run, writing actual records to ctx.records.

    Pre-conditions:
        get_openai_client is patched; first response is "not-a-number" (triggers
        retry), second is "15.0".
        A real CallContext is passed.
    Post-conditions:
        - Exactly 2 llm_call records.
        - First record span has no "_retry_" substring.
        - Second record span equals first_span + "_retry_1".

    Complexity:
        O(1) — no real I/O.
    """
    from src.llm_pipeline_telemetry import CallContext

    ctx = CallContext(pipeline="test_pipeline")
    mock_client = _make_mock_openai_chat_client(["not-a-number", "15.0"])

    with (
        patch("src.multi.sim.agent.get_openai_client", return_value=mock_client),
        patch("src.multi.sim.agent.SimulationAgent.run_simulation", return_value=(420.0, True)),
    ):
        agent = SimulationAgent(provider="openai", max_iterations=1)
        agent.run_optimization_loop(ctx=ctx)

    llm_records = [r for r in ctx.records if r.get("record_type") == "llm_call"]
    assert len(llm_records) == 2
    first_span = llm_records[0]["span"]
    assert "_retry_" not in first_span
    assert llm_records[1]["span"] == f"{first_span}_retry_1"


def test_get_llm_suggestion_no_retry_no_suffix() -> None:
    """get_llm_suggestion with no retry produces a single record with no _retry_ in span.

    Patches get_openai_client at the HTTP level so the real _call_openai and
    log_llm_call run, writing actual records to ctx.records.

    Pre-conditions:
        get_openai_client is patched; first response is "15.0" (no retry needed).
        A real CallContext is passed.
    Post-conditions:
        - Exactly 1 llm_call record.
        - Span does not contain "_retry_".

    Complexity:
        O(1) — no real I/O.
    """
    from src.llm_pipeline_telemetry import CallContext

    ctx = CallContext(pipeline="test_pipeline")
    mock_client = _make_mock_openai_chat_client(["15.0"])

    with (
        patch("src.multi.sim.agent.get_openai_client", return_value=mock_client),
        patch("src.multi.sim.agent.SimulationAgent.run_simulation", return_value=(420.0, True)),
    ):
        agent = SimulationAgent(provider="openai", max_iterations=1)
        agent.run_optimization_loop(ctx=ctx)

    llm_records = [r for r in ctx.records if r.get("record_type") == "llm_call"]
    assert len(llm_records) == 1
    assert "_retry_" not in llm_records[0]["span"]
