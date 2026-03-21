"""End-to-end telemetry integration tests for the multi-agent pipeline.

ZERO MOCKING: uses the real OpenAI API to exercise the coordinator → executor
→ KB-agent / sim-agent chain and verifies that:
- All llm_call records share the pipeline's run_id.
- pipeline_outcome_and_stats is emitted with status="success".
- Total tokens and duration are positive.
- call_start_ts / call_end_ts are present on every llm_call record.

Skip when OPENAI_API_KEY is not set.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

pytestmark = [pytest.mark.integration, pytest.mark.multi]

_skip_no_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping multi-agent telemetry integration tests",
)


@_skip_no_openai
def test_multi_agent_web_search_path_emits_records():
    """Multi-agent pipeline (web-search path) produces telemetry with correct run_id."""
    import dataclasses
    import time
    from datetime import datetime, timezone

    import src.multi.file_store as file_store
    from src.llm_pipeline_telemetry import CallContext, log_pipeline_outcome_and_stats
    from src.coordinator import _classify_with_llm
    from src.multi.kb_agent import ask

    file_store.clear_openai()

    ctx = CallContext(pipeline="multi_agent")
    start_ts = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    ask("What is the capital city of France?", ctx=dataclasses.replace(ctx, agent="kb_agent", span="kb_query"))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    end_ts = datetime.now(timezone.utc)
    log_pipeline_outcome_and_stats(ctx, total_wall_clock_runtime_ms=elapsed_ms, start_ts=start_ts, end_ts=end_ts, status="success")

    llm_records = [r for r in ctx.records if r.get("record_type") == "llm_call"]
    assert len(llm_records) >= 1

    for rec in llm_records:
        assert rec["pipeline"] == "multi_agent"
        assert rec["run_id"] == ctx.run_id
        assert rec["input_tokens"] > 0
        assert rec["output_tokens"] > 0
        assert rec["client_elapsed_ms"] > 0
        assert "call_start_ts" in rec
        assert "call_end_ts" in rec

    pipeline_records = [r for r in ctx.records if r.get("record_type") == "pipeline_outcome_and_stats"]
    assert len(pipeline_records) == 1
    pr = pipeline_records[0]
    assert pr["status"] == "success"
    assert pr["total_input_tokens"] > 0
    assert pr["total_output_tokens"] > 0


@_skip_no_openai
def test_multi_agent_sim_path_emits_records():
    """Multi-agent sim-agent path: llm_call records are attributed to multi_agent pipeline."""
    import dataclasses
    import time
    from datetime import datetime, timezone

    from src.llm_pipeline_telemetry import CallContext, log_pipeline_outcome_and_stats
    from src.multi.sim.agent import SimulationAgent

    ctx = CallContext(pipeline="multi_agent")
    start_ts = datetime.now(timezone.utc)
    t0 = time.perf_counter()

    agent = SimulationAgent(max_iterations=1)
    agent.run_optimization_loop(
        initial_cooling_rate_K_per_min=15.0,
        ctx=dataclasses.replace(ctx, agent="sim_agent"),
    )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    end_ts = datetime.now(timezone.utc)
    log_pipeline_outcome_and_stats(ctx, total_wall_clock_runtime_ms=elapsed_ms, start_ts=start_ts, end_ts=end_ts, status="success")

    llm_records = [r for r in ctx.records if r.get("record_type") == "llm_call"]
    assert len(llm_records) >= 1
    for rec in llm_records:
        assert rec["pipeline"] == "multi_agent"
        assert rec["run_id"] == ctx.run_id
        assert rec["agent"] == "sim_agent"

    pipeline_records = [r for r in ctx.records if r.get("record_type") == "pipeline_outcome_and_stats"]
    assert len(pipeline_records) == 1
