"""Unit tests for src.llm_pipeline_telemetry — pure Python, no network calls.

Covers:
- CallContext construction, run_id uniqueness, and shared-records invariant.
- run_id consistency across dataclasses.replace() snapshots.
- Singleton accessor identity (get_openai_client / get_anthropic_client).
- log_llm_call: field shape, throughput arithmetic (parametrized), status
  propagation, provider_server_latency_ms fallback logic, field isolation.
- log_tool_execution: field shape, optional tool_internal_runtime_ms, field
  isolation (no token fields).
- log_pipeline_outcome_and_stats: token aggregation arithmetic, total_wall_clock_runtime_ms
  accuracy, start_ts/end_ts presence, is_provider_server_latency_complete three
  coverage cases, zero-accumulated-records edge case, error-path fields.
- JsonFormatter: output is valid JSON with expected keys.
- _configure_logger: no duplicate handlers on repeated calls.

Pillar compliance:
    Pillar 2: ≥85% branch coverage; parametrized arithmetic; edge cases.
    Pillar 3: Google-style docstrings, contract assertions.
    Pillar 6: O(1) / O(n) complexity notes in tested functions.
    Pillar 7: No uncaught exceptions; assertion-driven invariant checks.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import time
from datetime import datetime, timezone

import pytest

import src.llm_pipeline_telemetry as telem
from src.llm_pipeline_telemetry import (
    CallContext,
    JsonFormatter,
    _configure_logger,
    log_llm_call,
    log_pipeline_outcome_and_stats,
    log_tool_execution,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(pipeline: str = "linear") -> CallContext:
    """Create a fresh CallContext for test isolation."""
    return CallContext(pipeline=pipeline)


def _ts() -> str:
    """Return a current UTC ISO timestamp string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# CallContext
# ---------------------------------------------------------------------------

class TestCallContext:
    """Tests for CallContext dataclass invariants."""

    def test_run_id_is_32_hex_chars(self):
        ctx = _make_ctx()
        assert len(ctx.run_id) == 32
        int(ctx.run_id, 16)  # raises ValueError if not valid hex

    def test_run_id_unique_per_instance(self):
        assert _make_ctx().run_id != _make_ctx().run_id

    def test_records_starts_empty(self):
        ctx = _make_ctx()
        assert ctx.records == []

    def test_replace_shares_records_list(self):
        """dataclasses.replace() must share the same records list object."""
        ctx = _make_ctx()
        snap = dataclasses.replace(ctx, agent="extractor", span="extract")
        assert snap.records is ctx.records

    def test_replace_shares_run_id(self):
        """All snapshots from one root ctx carry the same run_id."""
        ctx = _make_ctx()
        snaps = [
            dataclasses.replace(ctx, agent=f"agent_{i}", span=f"span_{i}")
            for i in range(5)
        ]
        for s in snaps:
            assert s.run_id == ctx.run_id

    def test_run_id_consistency_across_multiple_log_calls(self):
        """All llm_call records accumulated via snapshots share one run_id."""
        ctx = _make_ctx(pipeline="multi_agent")
        ts = _ts()
        for i in range(3):
            snap = dataclasses.replace(ctx, agent=f"agent_{i}", span=f"span_{i}")
            log_llm_call(
                snap,
                model="gpt-4o-mini",
                input_tokens=10,
                output_tokens=5,
                provider_server_latency_ms=None,
                client_elapsed_ms=100.0,
                call_start_ts=ts,
                call_end_ts=ts,
            )
        llm_records = [r for r in ctx.records if r["record_type"] == "llm_call"]
        assert len(llm_records) == 3
        run_ids = {r["run_id"] for r in llm_records}
        assert run_ids == {ctx.run_id}


# ---------------------------------------------------------------------------
# Singleton accessors
# ---------------------------------------------------------------------------

import os as _os

_skip_no_openai = pytest.mark.skipif(
    not _os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set; skipping singleton identity test",
)
_skip_no_anthropic = pytest.mark.skipif(
    not _os.environ.get("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set; skipping singleton identity test",
)


class TestSingletonAccessors:
    """Tests for get_openai_client / get_anthropic_client singleton identity."""

    @_skip_no_openai
    def test_get_openai_client_returns_same_instance(self):
        c1 = telem.get_openai_client()
        c2 = telem.get_openai_client()
        assert c1 is c2

    @_skip_no_anthropic
    def test_get_anthropic_client_returns_same_instance(self):
        c1 = telem.get_anthropic_client()
        c2 = telem.get_anthropic_client()
        assert c1 is c2


# ---------------------------------------------------------------------------
# log_llm_call — field shape & status
# ---------------------------------------------------------------------------

class TestLogLlmCallShape:
    """Tests for log_llm_call record structure."""

    def setup_method(self):
        self.ctx = _make_ctx()
        self.ts = _ts()

    def _call(self, **overrides):
        defaults = dict(
            model="gpt-4o-mini",
            input_tokens=100,
            output_tokens=20,
            provider_server_latency_ms=500,
            client_elapsed_ms=800.0,
            call_start_ts=self.ts,
            call_end_ts=self.ts,
            status="ok",
        )
        defaults.update(overrides)
        log_llm_call(self.ctx, **defaults)
        return self.ctx.records[-1]

    def test_record_type(self):
        assert self._call()["record_type"] == "llm_call"

    def test_pipeline_label(self):
        assert self._call()["pipeline"] == "linear"

    def test_run_id_present(self):
        r = self._call()
        assert r["run_id"] == self.ctx.run_id

    def test_input_output_tokens_separate(self):
        r = self._call(input_tokens=77, output_tokens=33)
        assert r["input_tokens"] == 77
        assert r["output_tokens"] == 33
        assert "total_tokens" not in r

    def test_call_start_end_ts_present(self):
        r = self._call()
        assert "call_start_ts" in r
        assert "call_end_ts" in r

    def test_status_ok(self):
        assert self._call(status="ok")["status"] == "ok"

    def test_status_error(self):
        assert self._call(status="error")["status"] == "error"

    def test_status_partial(self):
        assert self._call(status="partial")["status"] == "partial"

    def test_no_tool_name_field(self):
        """llm_call records must not carry tool_name (field isolation)."""
        r = self._call()
        assert "tool_name" not in r

    def test_appends_exactly_one_record(self):
        before = len(self.ctx.records)
        self._call()
        assert len(self.ctx.records) == before + 1


# ---------------------------------------------------------------------------
# log_llm_call — throughput arithmetic (parametrized)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "output_tokens,provider_ms,client_ms,expected_throughput",
    [
        # server latency wins when positive
        (10, 500, 800, 20.0),    # 10 / 0.5 s = 20 tok/s
        (20, 1000, 1200, 20.0),  # 20 / 1.0 s = 20 tok/s
        # fall back to client_ms when provider_ms is None
        (15, None, 500, 30.0),   # 15 / 0.5 s = 30 tok/s
        # fall back to client_ms when provider_ms is 0
        (10, 0, 500, 20.0),      # 10 / 0.5 s = 20 tok/s
        # both zero → throughput is 0.0, no ZeroDivisionError
        (5, 0, 0, 0.0),
    ],
)
def test_log_llm_call_throughput_arithmetic(
    output_tokens, provider_ms, client_ms, expected_throughput
):
    """Throughput is computed correctly for all denominator selection branches."""
    ctx = _make_ctx()
    ts = _ts()
    log_llm_call(
        ctx,
        model="gpt-4o-mini",
        input_tokens=50,
        output_tokens=output_tokens,
        provider_server_latency_ms=provider_ms,
        client_elapsed_ms=float(client_ms),
        call_start_ts=ts,
        call_end_ts=ts,
    )
    assert ctx.records[-1]["throughput_output_tokens_per_sec"] == pytest.approx(
        expected_throughput, rel=1e-3
    )


# ---------------------------------------------------------------------------
# log_tool_execution — field shape & isolation
# ---------------------------------------------------------------------------

class TestLogToolExecution:
    """Tests for log_tool_execution record structure."""

    def setup_method(self):
        self.ctx = _make_ctx(pipeline="multi_agent")
        self.ts = _ts()

    def _call(self, **overrides):
        defaults = dict(
            tool_name="compute_elastic_constants_tool",
            tool_execution_ms=58400.0,
            status="ok",
            call_start_ts=self.ts,
            call_end_ts=self.ts,
        )
        defaults.update(overrides)
        log_tool_execution(self.ctx, **defaults)
        return self.ctx.records[-1]

    def test_record_type(self):
        assert self._call()["record_type"] == "tool_execution"

    def test_tool_name_present(self):
        assert self._call()["tool_name"] == "compute_elastic_constants_tool"

    def test_status_ok(self):
        assert self._call(status="ok")["status"] == "ok"

    def test_status_error(self):
        assert self._call(status="error")["status"] == "error"

    def test_tool_internal_runtime_ms_included_when_provided(self):
        r = self._call(tool_internal_runtime_ms=52100.0)
        assert r["tool_internal_runtime_ms"] == pytest.approx(52100.0)

    def test_tool_internal_runtime_ms_absent_when_none(self):
        r = self._call()
        assert "tool_internal_runtime_ms" not in r

    def test_no_input_tokens_field(self):
        """tool_execution records must not carry token fields (field isolation)."""
        r = self._call()
        assert "input_tokens" not in r
        assert "output_tokens" not in r

    def test_call_start_end_ts_present(self):
        r = self._call()
        assert "call_start_ts" in r
        assert "call_end_ts" in r

    def test_appends_exactly_one_record(self):
        before = len(self.ctx.records)
        self._call()
        assert len(self.ctx.records) == before + 1


# ---------------------------------------------------------------------------
# log_pipeline_outcome_and_stats
# ---------------------------------------------------------------------------

class TestLogPipelineOutcomeAndStats:
    """Tests for log_pipeline_outcome_and_stats aggregation and field contracts."""

    def _populate_llm_records(
        self, ctx: CallContext, count: int, provider_ms: int | None
    ):
        ts = _ts()
        for i in range(count):
            snap = dataclasses.replace(ctx, agent="a", span=f"s_{i}")
            log_llm_call(
                snap,
                model="gpt-4o-mini",
                input_tokens=10,
                output_tokens=5,
                provider_server_latency_ms=provider_ms,
                client_elapsed_ms=100.0,
                call_start_ts=ts,
                call_end_ts=ts,
            )

    def test_record_type(self):
        ctx = _make_ctx()
        self._populate_llm_records(ctx, 2, 500)
        now = datetime.now(timezone.utc)
        log_pipeline_outcome_and_stats(ctx, total_wall_clock_runtime_ms=1200.0, start_ts=now, end_ts=now, status="success")
        r = ctx.records[-1]
        assert r["record_type"] == "pipeline_outcome_and_stats"

    def test_token_aggregation_arithmetic(self):
        """total_input/output_tokens are exact sums across llm_call records."""
        ctx = _make_ctx()
        ts = _ts()
        for inp, out in [(10, 5), (20, 8), (15, 12)]:
            snap = dataclasses.replace(ctx, agent="a", span="s")
            log_llm_call(snap, model="gpt-4o-mini", input_tokens=inp, output_tokens=out,
                         provider_server_latency_ms=None, client_elapsed_ms=100.0,
                         call_start_ts=ts, call_end_ts=ts)
        now = datetime.now(timezone.utc)
        log_pipeline_outcome_and_stats(ctx, total_wall_clock_runtime_ms=500.0, start_ts=now, end_ts=now, status="success")
        r = ctx.records[-1]
        assert r["total_input_tokens"] == 45
        assert r["total_output_tokens"] == 25
        assert r["llm_call_count"] == 3

    def test_total_wall_clock_runtime_ms_accuracy(self):
        """total_wall_clock_runtime_ms must be at least 15ms when pipeline sleeps 20ms."""
        ctx = _make_ctx()
        t0 = time.perf_counter()
        time.sleep(0.020)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        now = datetime.now(timezone.utc)
        log_pipeline_outcome_and_stats(ctx, total_wall_clock_runtime_ms=elapsed_ms, start_ts=now, end_ts=now, status="success")
        r = ctx.records[-1]
        assert r["total_wall_clock_runtime_ms"] >= 15.0

    def test_start_ts_and_end_ts_present(self):
        ctx = _make_ctx()
        start = datetime.now(timezone.utc)
        end = datetime.now(timezone.utc)
        log_pipeline_outcome_and_stats(ctx, total_wall_clock_runtime_ms=100.0, start_ts=start, end_ts=end, status="success")
        r = ctx.records[-1]
        assert "start_ts" in r
        assert "end_ts" in r
        assert r["start_ts"] == start.isoformat()
        assert r["end_ts"] == end.isoformat()

    # -- server-latency coverage cases --

    def test_all_calls_have_server_latency_complete_true(self):
        """is_provider_server_latency_complete=True when every call has valid server ms."""
        ctx = _make_ctx()
        self._populate_llm_records(ctx, 3, 500)
        now = datetime.now(timezone.utc)
        log_pipeline_outcome_and_stats(ctx, total_wall_clock_runtime_ms=300.0, start_ts=now, end_ts=now, status="success")
        r = ctx.records[-1]
        assert r["is_provider_server_latency_complete"] is True
        assert r["calls_with_provider_server_latency"] == 3
        assert r["total_provider_server_latency_ms"] == 1500

    def test_no_calls_have_server_latency_complete_false(self):
        """is_provider_server_latency_complete=False when no call has server ms."""
        ctx = _make_ctx()
        self._populate_llm_records(ctx, 3, None)
        now = datetime.now(timezone.utc)
        log_pipeline_outcome_and_stats(ctx, total_wall_clock_runtime_ms=300.0, start_ts=now, end_ts=now, status="success")
        r = ctx.records[-1]
        assert r["is_provider_server_latency_complete"] is False
        assert r["calls_with_provider_server_latency"] == 0
        assert r["total_provider_server_latency_ms"] is None

    def test_partial_calls_have_server_latency_complete_false(self):
        """is_provider_server_latency_complete=False when only some calls have server ms."""
        ctx = _make_ctx()
        ts = _ts()
        snap1 = dataclasses.replace(ctx, agent="a", span="s1")
        log_llm_call(snap1, model="gpt-4o-mini", input_tokens=10, output_tokens=5,
                     provider_server_latency_ms=500, client_elapsed_ms=100.0,
                     call_start_ts=ts, call_end_ts=ts)
        snap2 = dataclasses.replace(ctx, agent="a", span="s2")
        log_llm_call(snap2, model="gpt-4o-mini", input_tokens=10, output_tokens=5,
                     provider_server_latency_ms=None, client_elapsed_ms=100.0,
                     call_start_ts=ts, call_end_ts=ts)
        now = datetime.now(timezone.utc)
        log_pipeline_outcome_and_stats(ctx, total_wall_clock_runtime_ms=200.0, start_ts=now, end_ts=now, status="success")
        r = ctx.records[-1]
        assert r["is_provider_server_latency_complete"] is False
        assert r["calls_with_provider_server_latency"] == 1
        assert r["total_provider_server_latency_ms"] is None

    def test_zero_accumulated_records_edge_case(self):
        """Zero llm_call records: counts are 0, complete=False, total_server=None."""
        ctx = _make_ctx()
        now = datetime.now(timezone.utc)
        log_pipeline_outcome_and_stats(ctx, total_wall_clock_runtime_ms=10.0, start_ts=now, end_ts=now, status="success")
        r = ctx.records[-1]
        assert r["llm_call_count"] == 0
        assert r["total_input_tokens"] == 0
        assert r["total_output_tokens"] == 0
        assert r["is_provider_server_latency_complete"] is False
        assert r["total_provider_server_latency_ms"] is None

    def test_error_status_sets_is_partial_data_true(self):
        ctx = _make_ctx()
        now = datetime.now(timezone.utc)
        log_pipeline_outcome_and_stats(ctx, total_wall_clock_runtime_ms=0.0, start_ts=now, end_ts=now,
                                       status="error", exception_type="ValueError",
                                       exception_message="something broke")
        r = ctx.records[-1]
        assert r["status"] == "error"
        assert r["is_partial_data"] is True
        assert r["exception_type"] == "ValueError"
        assert r["exception_message"] == "something broke"

    def test_success_status_no_exception_fields(self):
        ctx = _make_ctx()
        now = datetime.now(timezone.utc)
        log_pipeline_outcome_and_stats(ctx, total_wall_clock_runtime_ms=100.0, start_ts=now, end_ts=now, status="success")
        r = ctx.records[-1]
        assert r["is_partial_data"] is False
        assert "exception_type" not in r
        assert "exception_message" not in r


# ---------------------------------------------------------------------------
# JsonFormatter
# ---------------------------------------------------------------------------

class TestJsonFormatter:
    """Tests for JsonFormatter structured JSON output."""

    def _capture(self, payload) -> dict:
        import io
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter())
        logger = logging.getLogger(f"test.json.{id(self)}")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.info(payload)
        line = stream.getvalue().strip()
        return json.loads(line)

    def test_output_is_valid_json(self):
        out = self._capture({"foo": "bar"})
        assert isinstance(out, dict)

    def test_expected_top_level_keys(self):
        out = self._capture({"foo": "bar"})
        assert set(out.keys()) >= {"level", "logger", "timestamp", "data"}

    def test_data_key_contains_payload(self):
        out = self._capture({"record_type": "llm_call", "model": "gpt-4o-mini"})
        assert out["data"]["record_type"] == "llm_call"


# ---------------------------------------------------------------------------
# _configure_logger
# ---------------------------------------------------------------------------

class TestConfigureLogger:
    """Tests for _configure_logger deduplication guard."""

    def test_repeated_calls_do_not_add_duplicate_handlers(self):
        _configure_logger()
        _configure_logger()
        logger = logging.getLogger("llm.telemetry")
        assert len(logger.handlers) == 1
