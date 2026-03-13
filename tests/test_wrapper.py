"""Tests for src.wrapper complete() and complete_with_tools() with mocked providers."""
import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


def test_complete_default_provider_calls_openai(monkeypatch):
    from src import wrapper
    monkeypatch.setattr(wrapper, "_complete_openai", lambda msgs: "openai-reply")
    monkeypatch.setattr(wrapper, "_complete_anthropic", lambda msgs: "anthropic-reply")
    assert wrapper.complete([{"role": "user", "content": "Hi"}]) == "openai-reply"


def test_complete_anthropic_provider_calls_anthropic(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    from src import wrapper
    monkeypatch.setattr(wrapper, "_complete_openai", lambda msgs: "openai-reply")
    monkeypatch.setattr(wrapper, "_complete_anthropic", lambda msgs: "anthropic-reply")
    assert wrapper.complete([{"role": "user", "content": "Hi"}]) == "anthropic-reply"


def test_complete_openai_provider_calls_openai(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    from src import wrapper
    monkeypatch.setattr(wrapper, "_complete_openai", lambda msgs: "openai-reply")
    monkeypatch.setattr(wrapper, "_complete_anthropic", lambda msgs: "anthropic-reply")
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

    with patch("openai.OpenAI") as mock_client_cls:
        mock_client_cls.return_value.chat.completions.create.return_value = (
            _openai_text_response("Direct answer")
        )
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

    call_seq = [
        _openai_tool_call_response("compute_elastic_constants_tool", {"composition": "Ni"}),
        _openai_text_response("C11=247 GPa"),
    ]

    with patch("openai.OpenAI") as mock_client_cls:
        mock_client_cls.return_value.chat.completions.create.side_effect = call_seq
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

    call_seq = [
        _anthropic_tool_call_response("compute_elastic_constants_tool", {"composition": "Ni"}),
        _anthropic_text_response("C11=247 GPa"),
    ]

    with patch("anthropic.Anthropic") as mock_client_cls:
        mock_client_cls.return_value.messages.create.side_effect = call_seq
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

    with patch("openai.OpenAI") as mock_client_cls:
        mock_client_cls.return_value.chat.completions.create.return_value = (
            _openai_tool_call_response("compute_elastic_constants_tool", {"composition": "Al"})
        )
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

    with patch("anthropic.Anthropic") as mock_client_cls:
        mock_client_cls.return_value.messages.create.return_value = (
            _anthropic_tool_call_response("compute_elastic_constants_tool", {"composition": "Al"})
        )
        with patch("src.tool_registry.call", return_value=_TOOL_RESULT) as mock_call:
            result = complete_with_tools(
                [{"role": "user", "content": "Keep calling tools"}], provider="anthropic"
            )

    assert mock_call.call_count == wrapper.MAX_TOOL_CALLS
    assert result == ""
