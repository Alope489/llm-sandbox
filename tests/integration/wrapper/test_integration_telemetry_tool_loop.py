"""End-to-end telemetry integration tests for the wrapper tool-calling loop.

ZERO MOCKING: uses the real OpenAI API + Docker/LAMMPS to exercise the
complete_with_tools() path and verify that:
- One llm_call record and one tool_execution record are produced per iteration.
- All records carry the correct pipeline run_id.
- tool_execution records have tool_execution_ms > 0.
- call_start_ts / call_end_ts are present on every record.
- provider_server_latency_ms is a positive int on every llm_call record.

Skip when OPENAI_API_KEY is not set or when Docker is unavailable.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests.telemetry_helpers import assert_openai_server_latency

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

pytestmark = [pytest.mark.integration, pytest.mark.wrapper]


def _docker_available() -> bool:
    """Return True if Docker daemon is accessible."""
    try:
        result = subprocess.run(
            ["docker", "info"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


_skip_no_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping tool-loop telemetry integration tests",
)
_skip_no_docker = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker not available; skipping tool-loop telemetry integration tests",
)


@_skip_no_openai
@_skip_no_docker
def test_tool_loop_emits_llm_call_and_tool_execution_records():
    """complete_with_tools with a real LLM+Docker call emits paired telemetry records."""
    import dataclasses
    from src.llm_pipeline_telemetry import CallContext
    from src.wrapper import complete_with_tools

    ctx = CallContext(pipeline="multi_agent")
    snap = dataclasses.replace(ctx, agent="sim_agent", span="tool_query")

    result = complete_with_tools(
        [{"role": "user", "content": "Compute elastic constants for pure Nickel (Ni)."}],
        provider="openai",
        ctx=snap,
    )
    assert isinstance(result, str) and len(result) > 0

    llm_records = [r for r in ctx.records if r.get("record_type") == "llm_call"]
    tool_records = [r for r in ctx.records if r.get("record_type") == "tool_execution"]

    assert len(llm_records) >= 1, "Expected at least one llm_call record"
    assert len(tool_records) >= 1, "Expected at least one tool_execution record"

    for rec in llm_records:
        assert rec["pipeline"] == "multi_agent"
        assert rec["run_id"] == ctx.run_id
        assert rec["client_elapsed_ms"] > 0
        assert "call_start_ts" in rec
        assert "call_end_ts" in rec
        assert_openai_server_latency(rec)  # openai-processing-ms must be a positive int

    for rec in tool_records:
        assert rec["pipeline"] == "multi_agent"
        assert rec["run_id"] == ctx.run_id
        assert rec["tool_execution_ms"] > 0
        assert "call_start_ts" in rec
        assert "call_end_ts" in rec


@_skip_no_openai
@_skip_no_docker
def test_tool_execution_record_has_tool_name():
    """tool_execution record must carry the tool_name field."""
    import dataclasses
    from src.llm_pipeline_telemetry import CallContext
    from src.wrapper import complete_with_tools

    ctx = CallContext(pipeline="multi_agent")
    snap = dataclasses.replace(ctx, agent="sim_agent", span="tool_query")

    complete_with_tools(
        [{"role": "user", "content": "Compute elastic constants for pure Nickel (Ni)."}],
        provider="openai",
        ctx=snap,
    )

    tool_records = [r for r in ctx.records if r.get("record_type") == "tool_execution"]
    assert len(tool_records) >= 1
    assert all("tool_name" in r for r in tool_records)
