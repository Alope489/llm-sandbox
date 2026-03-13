"""Level 2 integration tests for complete_with_tools.

These tests make REAL LLM API calls but mock tool_registry.call so Docker
is never needed. Their purpose is to verify that the tool schema descriptions
are compelling enough for the live LLM to autonomously emit a tool call when
asked a material science question — something unit tests cannot prove.

If mock_call.assert_called_once() fails, the tool description needs improving.

Runtime: ~3 seconds per test. Cost: ~$0.001 per test.
Requires: OPENAI_API_KEY and/or ANTHROPIC_API_KEY in .env.
"""
import os
from unittest.mock import patch

import pytest
from dotenv import load_dotenv

load_dotenv()

_skip_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run OpenAI integration tests",
)
_skip_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="Set ANTHROPIC_API_KEY to run Anthropic integration tests",
)

_MOCK_RESULT = {
    "composition": "Ni",
    "C11": 247.0,
    "C12": 153.0,
    "C44": 122.0,
    "runtime_seconds": 55.1,
    "status": "ok",
}


@_skip_openai
@patch("src.tool_registry.call", return_value=_MOCK_RESULT)
def test_openai_llm_calls_elastic_tool_for_material_query(mock_call):
    """Real OpenAI API: LLM must autonomously invoke the elastic constants tool.

    Sends a material science query with registered tool schemas. Asserts that
    the live model emits a tool_call for compute_elastic_constants_tool rather
    than answering from memory, and that the final reply is a non-empty string.
    """
    from src.wrapper import complete_with_tools

    result = complete_with_tools(
        [{"role": "user", "content":
          "What are the elastic constants of Nickel? Use the available tools."}],
        provider="openai",
    )

    mock_call.assert_called_once()
    name, kwargs = mock_call.call_args[0]
    assert name == "compute_elastic_constants_tool"
    assert kwargs.get("composition") == "Ni"
    assert isinstance(result, str) and len(result) > 0


@_skip_anthropic
@patch.dict(os.environ, {"ANTHROPIC_MODEL": "claude-sonnet-4-6"})
@patch("src.tool_registry.call", return_value=_MOCK_RESULT)
def test_anthropic_llm_calls_elastic_tool_for_material_query(mock_call):
    """Real Anthropic API: LLM must autonomously invoke the elastic constants tool.

    Sends a material science query with registered tool schemas. Asserts that
    the live model emits a tool_use block for compute_elastic_constants_tool
    rather than answering from memory, and that the final reply is non-empty.
    """
    from src.wrapper import complete_with_tools

    result = complete_with_tools(
        [{"role": "user", "content":
          "What are the elastic constants of Nickel? Use the available tools."}],
        provider="anthropic",
    )

    mock_call.assert_called_once()
    name, kwargs = mock_call.call_args[0]
    assert name == "compute_elastic_constants_tool"
    assert kwargs.get("composition") == "Ni"
    assert isinstance(result, str) and len(result) > 0
