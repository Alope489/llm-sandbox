"""Tests for src.wrapper complete() and complete_with_tools() with mocked providers."""
import dataclasses as _dc
import json
import os
import os as _os
import sys
from unittest.mock import MagicMock, patch, patch as patch_obj

import pytest

from tests.telemetry_helpers import assert_openai_server_latency

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


def test_complete_default_provider_calls_openai(monkeypatch):
    from src import wrapper
    monkeypatch.setattr(wrapper, "_complete_openai", lambda msgs, ctx=None, max_tokens=None: "openai-reply")
    monkeypatch.setattr(wrapper, "_complete_anthropic", lambda msgs, ctx=None: "anthropic-reply")
    assert wrapper.complete([{"role": "user", "content": "Hi"}]) == "openai-reply"


def test_complete_anthropic_provider_calls_anthropic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    from src import wrapper
    monkeypatch.setattr(wrapper, "_complete_openai", lambda msgs, ctx=None, max_tokens=None: "openai-reply")
    monkeypatch.setattr(wrapper, "_complete_anthropic", lambda msgs, ctx=None: "anthropic-reply")
    assert wrapper.complete([{"role": "user", "content": "Hi"}]) == "anthropic-reply"


def test_complete_openai_provider_calls_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    from src import wrapper
    monkeypatch.setattr(wrapper, "_complete_openai", lambda msgs, ctx=None, max_tokens=None: "openai-reply")
    monkeypatch.setattr(wrapper, "_complete_anthropic", lambda msgs, ctx=None: "anthropic-reply")
    assert wrapper.complete([{"role": "user", "content": "Hello"}]) == "openai-reply"


# ---------------------------------------------------------------------------
# Helpers for complete_with_tools tests
# ---------------------------------------------------------------------------

def _openai_text_response(text: str):
    """Build a minimal mock OpenAI response with a plain text reply."""
    msg = MagicMock()
    msg.content = text
    msg.tool_calls = []
    msg.refusal = None
    msg.model_dump.return_value = {"role": "assistant", "content": text}
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    return resp


def _openai_tool_call_response(tool_name: str, args: dict, tc_id: str = "tc_1"):
    """Build a mock OpenAI response that requests one tool call."""
    tc = MagicMock()
    tc.id = tc_id
    tc.function.name = tool_name
    tc.function.arguments = json.dumps(args)
    msg = MagicMock()
    msg.content = None
    msg.tool_calls = [tc]
    msg.refusal = None
    msg.model_dump.return_value = {"role": "assistant", "content": None, "tool_calls": []}
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage.prompt_tokens = 20
    resp.usage.completion_tokens = 10
    return resp


def _anthropic_text_response(text: str):
    """Build a mock Anthropic response with a plain text block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    resp = MagicMock()
    resp.content = [block]
    resp.usage.input_tokens = 10
    resp.usage.output_tokens = 5
    return resp


def _anthropic_tool_call_response(tool_name: str, args: dict, tu_id: str = "tu_1"):
    """Build a mock Anthropic response that requests one tool_use block."""
    block = MagicMock()
    block.type = "tool_use"
    block.id = tu_id
    block.name = tool_name
    block.input = args
    resp = MagicMock()
    resp.content = [block]
    resp.usage.input_tokens = 20
    resp.usage.output_tokens = 10
    return resp


_TOOL_RESULT = {"composition": "Ni", "C11": 247.0, "C12": 153.0, "C44": 122.0, "status": "ok"}


# ---------------------------------------------------------------------------
# complete_with_tools tests
# ---------------------------------------------------------------------------

@patch("src.tool_registry.get_openai_schemas", return_value=[])
@patch("src.tool_registry.get_anthropic_schemas", return_value=[])
def test_complete_with_tools_no_call_path(mock_ant_schemas, mock_oai_schemas):
    """LLM returns plain text with no tool calls — registry.call never invoked."""
    from src.wrapper import complete_with_tools
    import src.llm_pipeline_telemetry as telem

    def _wrap_openai(resp):
        raw = MagicMock()
        raw.parse.return_value = resp
        raw.headers = {"openai-processing-ms": "42"}
        return raw

    mock_client = MagicMock()
    mock_client.with_raw_response.chat.completions.create.return_value = _wrap_openai(
        _openai_text_response("Direct answer")
    )
    with patch("src.wrapper.get_openai_client", return_value=mock_client):
        with patch("src.tool_registry.call") as mock_call:
            result = complete_with_tools(
                [{"role": "user", "content": "Hello"}], provider="openai"
            )
    assert result == "Direct answer"
    mock_call.assert_not_called()


@patch("src.tool_registry.get_openai_schemas", return_value=[])
def test_complete_with_tools_openai_single_loop(mock_schemas):
    """OpenAI: one tool call followed by a text response — registry.call invoked once."""
    from src.wrapper import complete_with_tools
    import src.llm_pipeline_telemetry as telem

    def _wrap_openai(resp):
        raw = MagicMock()
        raw.parse.return_value = resp
        raw.headers = {"openai-processing-ms": "55"}
        return raw

    call_seq = [
        _wrap_openai(_openai_tool_call_response("compute_elastic_constants_tool", {"composition": "Ni"})),
        _wrap_openai(_openai_text_response("C11=247 GPa")),
    ]

    mock_client = MagicMock()
    mock_client.with_raw_response.chat.completions.create.side_effect = call_seq
    with patch("src.wrapper.get_openai_client", return_value=mock_client):
        with patch("src.tool_registry.call", return_value=_TOOL_RESULT) as mock_call:
            result = complete_with_tools(
                [{"role": "user", "content": "Elastic constants of Ni?"}], provider="openai"
            )

    mock_call.assert_called_once()
    assert mock_call.call_args[0][0] == "compute_elastic_constants_tool"
    assert result == "C11=247 GPa"


@patch("src.tool_registry.get_anthropic_schemas", return_value=[])
def test_complete_with_tools_anthropic_single_loop(mock_schemas):
    """Anthropic: one tool_use block followed by a text response — registry.call invoked once."""
    from src.wrapper import complete_with_tools
    import src.llm_pipeline_telemetry as telem

    # Build raw-response wrappers that .parse() returns the underlying response
    def _wrap(resp):
        raw = MagicMock()
        raw.parse.return_value = resp
        raw.headers = {}
        return raw

    call_seq = [
        _wrap(_anthropic_tool_call_response("compute_elastic_constants_tool", {"composition": "Ni"})),
        _wrap(_anthropic_text_response("C11=247 GPa")),
    ]

    mock_client = MagicMock()
    mock_client.with_raw_response.messages.create.side_effect = call_seq
    with patch("src.wrapper.get_anthropic_client", return_value=mock_client):
        with patch("src.tool_registry.call", return_value=_TOOL_RESULT) as mock_call:
            result = complete_with_tools(
                [{"role": "user", "content": "Elastic constants of Ni?"}], provider="anthropic"
            )

    mock_call.assert_called_once()
    assert result == "C11=247 GPa"


@patch("src.tool_registry.get_openai_schemas", return_value=[])
def test_complete_with_tools_max_tool_calls_guard_openai(mock_schemas):
    """OpenAI: loop exits after MAX_TOOL_CALLS even if LLM keeps requesting tools."""
    from src import wrapper
    from src.wrapper import complete_with_tools
    import src.llm_pipeline_telemetry as telem

    def _wrap_openai(resp):
        raw = MagicMock()
        raw.parse.return_value = resp
        raw.headers = {"openai-processing-ms": "30"}
        return raw

    mock_client = MagicMock()
    mock_client.with_raw_response.chat.completions.create.return_value = _wrap_openai(
        _openai_tool_call_response("compute_elastic_constants_tool", {"composition": "Al"})
    )
    with patch("src.wrapper.get_openai_client", return_value=mock_client):
        with patch("src.tool_registry.call", return_value=_TOOL_RESULT) as mock_call:
            result = complete_with_tools(
                [{"role": "user", "content": "Keep calling tools"}], provider="openai"
            )

    assert mock_call.call_count == wrapper.MAX_TOOL_CALLS
    assert result == ""


@patch("src.tool_registry.get_anthropic_schemas", return_value=[])
def test_complete_with_tools_max_tool_calls_guard_anthropic(mock_schemas):
    """Anthropic: loop exits after MAX_TOOL_CALLS even if LLM keeps requesting tools."""
    from src import wrapper
    from src.wrapper import complete_with_tools
    import src.llm_pipeline_telemetry as telem

    def _wrap(resp):
        raw = MagicMock()
        raw.parse.return_value = resp
        raw.headers = {}
        return raw

    mock_client = MagicMock()
    mock_client.with_raw_response.messages.create.return_value = (
        _wrap(_anthropic_tool_call_response("compute_elastic_constants_tool", {"composition": "Al"}))
    )
    with patch("src.wrapper.get_anthropic_client", return_value=mock_client):
        with patch("src.tool_registry.call", return_value=_TOOL_RESULT) as mock_call:
            result = complete_with_tools(
                [{"role": "user", "content": "Keep calling tools"}], provider="anthropic"
            )

    assert mock_call.call_count == wrapper.MAX_TOOL_CALLS
    assert result == ""


# ---------------------------------------------------------------------------
# Telemetry attribution tests (real API calls — skip when key absent)
# ---------------------------------------------------------------------------

_skip_no_openai = pytest.mark.skipif(
    not _os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping real-API telemetry tests",
)
_skip_no_anthropic = pytest.mark.skipif(
    not _os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; skipping real-API telemetry tests",
)


@_skip_no_openai
def test_complete_openai_with_ctx_emits_llm_call_record():
    """Real OpenAI call: _complete_openai with ctx appends a well-formed record."""
    from src.wrapper import _complete_openai
    from src.llm_pipeline_telemetry import CallContext

    ctx = CallContext(pipeline="test")
    snap = _dc.replace(ctx, agent="wrapper", span="test_complete")
    result = _complete_openai([{"role": "user", "content": "Reply with the single word: hello"}], ctx=snap)
    assert isinstance(result, str) and len(result) > 0

    assert len(ctx.records) == 1
    rec = ctx.records[0]
    assert rec["record_type"] == "llm_call"
    assert rec["input_tokens"] > 0
    assert rec["output_tokens"] > 0
    assert rec["client_elapsed_ms"] > 0
    assert rec["status"] == "ok"
    assert "call_start_ts" in rec
    assert "call_end_ts" in rec
    assert_openai_server_latency(rec)  # openai-processing-ms must be a positive int


@_skip_no_openai
def test_complete_openai_ctx_none_no_op_returns_string():
    """_complete_openai with ctx=None must return a non-empty string without crashing."""
    from src.wrapper import _complete_openai

    result = _complete_openai([{"role": "user", "content": "Reply with the single word: hi"}], ctx=None)
    assert isinstance(result, str) and len(result) > 0


@_skip_no_anthropic
def test_complete_anthropic_with_ctx_emits_llm_call_record():
    """Real Anthropic call: _complete_anthropic with ctx appends a well-formed record."""
    from src.wrapper import _complete_anthropic
    from src.llm_pipeline_telemetry import CallContext

    ctx = CallContext(pipeline="test")
    snap = _dc.replace(ctx, agent="wrapper", span="test_complete_anthropic")
    result = _complete_anthropic([{"role": "user", "content": "Reply with the single word: hello"}], ctx=snap)
    assert isinstance(result, str) and len(result) > 0

    assert len(ctx.records) == 1
    rec = ctx.records[0]
    assert rec["record_type"] == "llm_call"
    assert rec["input_tokens"] > 0
    assert rec["output_tokens"] > 0
    assert rec["client_elapsed_ms"] > 0
    assert rec["status"] == "ok"
    assert rec["provider_server_latency_ms"] is None
    assert "call_start_ts" in rec
    assert "call_end_ts" in rec
