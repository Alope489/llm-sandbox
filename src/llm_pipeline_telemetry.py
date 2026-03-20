"""Structured telemetry for LLM pipeline calls.

Provides:
- ``CallContext`` dataclass — created once per pipeline run and propagated via
  ``dataclasses.replace()`` snapshots to every call site.
- ``get_openai_client`` / ``get_anthropic_client`` — process-level singleton
  accessors so the httpx TLS connection pool is shared across all call sites.
  All instrumented files import and call these instead of instantiating clients
  locally, ensuring ``client_elapsed_ms`` reflects only the HTTP round-trip
  after the first warm call.
- ``log_llm_call`` — emits a ``record_type="llm_call"`` JSON record.
- ``log_tool_execution`` — emits a ``record_type="tool_execution"`` JSON record.
- ``log_pipeline_outcome_and_stats`` — emits a
  ``record_type="pipeline_outcome_and_stats"`` aggregate record. Always called
  by entry points — both on success and on error — so every pipeline run
  produces exactly one summary record, enabling failure-rate analysis without
  survivorship bias.
- ``JsonFormatter`` / ``_configure_logger`` — ships structured JSON output on
  the ``llm.telemetry`` logger out of the box.

Dependencies:
    - Standard library only (``dataclasses``, ``uuid``, ``logging``, ``json``,
      ``time``, ``datetime``).
    - ``openai`` and ``anthropic`` SDKs (lazy-imported inside singleton
      accessors to avoid import-time errors when keys are absent in unit tests).

Pillar compliance:
    - Pillar 1: Precisely captures all required metrics per plan specification.
    - Pillar 3: Google-style docstrings with full contracts on every public item.
    - Pillar 4: Single-responsibility, DRY singleton pattern, no hardcoding.
    - Pillar 5: No secrets stored; reads API keys from environment via SDK.
    - Pillar 6: O(n) over ctx.records in log_pipeline_outcome_and_stats (n =
      call count per run, typically < 100; negligible).
    - Pillar 7: list.append() is GIL-atomic in CPython; throughput guards
      prevent ZeroDivisionError; exception fields are optional and never leak
      into success records.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from anthropic import Anthropic as _AnthropicT
    from openai import OpenAI as _OpenAIT

# ---------------------------------------------------------------------------
# Singleton SDK client registry
# ---------------------------------------------------------------------------

_openai_client: Optional["_OpenAIT"] = None
_anthropic_client: Optional["_AnthropicT"] = None


def get_openai_client() -> "_OpenAIT":
    """Return the process-level OpenAI singleton, creating it on first call.

    The client is created once and reused for all subsequent calls in the
    process, so the underlying ``httpx.Client`` TLS connection pool is shared.
    After the first warm call, ``client_elapsed_ms`` in call sites reflects
    only the HTTP request/response round-trip with no TCP/TLS handshake noise.

    Returns:
        The process-level ``openai.OpenAI`` singleton instance.

    Raises:
        openai.AuthenticationError: If ``OPENAI_API_KEY`` is absent or invalid
            (raised on first API call, not here).

    Examples:
        >>> client = get_openai_client()
        >>> client is get_openai_client()
        True

    Complexity:
        O(1) — dictionary lookup after first call.
    """
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI()
    return _openai_client


def get_anthropic_client() -> "_AnthropicT":
    """Return the process-level Anthropic singleton, creating it on first call.

    Same connection-pooling rationale as ``get_openai_client``.

    Returns:
        The process-level ``anthropic.Anthropic`` singleton instance.

    Raises:
        anthropic.AuthenticationError: If ``ANTHROPIC_API_KEY`` is absent or
            invalid (raised on first API call, not here).

    Examples:
        >>> client = get_anthropic_client()
        >>> client is get_anthropic_client()
        True

    Complexity:
        O(1) — dictionary lookup after first call.
    """
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic
        _anthropic_client = Anthropic()
    return _anthropic_client


# ---------------------------------------------------------------------------
# CallContext dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class CallContext:
    """Immutable-label context propagated through a single pipeline run.

    Created once at the pipeline entry point (``orchestrator.run`` or
    ``coordinator.run``).  Labels are never mutated on the shared instance;
    call sites use ``dataclasses.replace(ctx, agent=..., span=...)`` to produce
    per-call snapshots.  The ``records`` list is intentionally shared across
    all snapshots (shallow-copied by ``dataclasses.replace``), so every
    ``log_*`` call from any snapshot appends to the one list the caller holds.

    Attributes:
        pipeline: Pipeline name — ``"linear"`` or ``"multi_agent"``.
        run_id: 32-char UUID hex string (128-bit); unique per
            ``orchestrator.run`` / ``coordinator.run`` invocation.
        agent: Active agent label, e.g. ``"extractor"``, ``"sim_agent"``.
        span: Active span label, e.g. ``"extract"``, ``"sim_iter_3"``.
        iteration: Iteration counter (used by the simulation loop).
        records: Accumulates one record dict per ``log_*`` call during the run.

    Invariants:
        - ``run_id`` is a 32-character hexadecimal string.
        - ``records`` is always a list (never None).

    Examples:
        >>> ctx = CallContext(pipeline="linear")
        >>> len(ctx.run_id) == 32
        True
        >>> snap = dataclasses.replace(ctx, agent="extractor", span="extract")
        >>> snap.records is ctx.records  # shared list
        True
    """

    pipeline: str
    run_id: str = dataclasses.field(
        default_factory=lambda: uuid.uuid4().hex
    )
    agent: str = ""
    span: str = ""
    iteration: int = 0
    records: list = dataclasses.field(default_factory=list)


# ---------------------------------------------------------------------------
# Logging infrastructure
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """Format each ``LogRecord`` as a single-line JSON string.

    Serialises the record's ``msg`` (expected to already be a dict or
    JSON-serialisable object) along with ``levelname``, ``name``, and
    ``asctime`` into a compact JSON line.  Non-serialisable objects fall back
    to their ``repr``.

    Examples:
        >>> import logging, io
        >>> handler = logging.StreamHandler(io.StringIO())
        >>> handler.setFormatter(JsonFormatter())
        >>> logger = logging.getLogger("test.json")
        >>> logger.addHandler(handler)
        >>> logger.info({"key": "value"})

    Complexity:
        O(k) where k is the number of fields in the record dict.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        """Serialize *record* to a single-line JSON string.

        Args:
            record: The ``logging.LogRecord`` to format.

        Returns:
            A single-line JSON string with ``level``, ``logger``,
            ``timestamp``, and ``data`` keys.

        Postconditions:
            - The returned string is valid JSON parseable with ``json.loads``.
            - No trailing newline is included.
        """
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "timestamp": self.formatTime(record, self.datefmt),
            "data": record.msg,
        }
        return json.dumps(payload, default=repr)


def _configure_logger() -> logging.Logger:
    """Attach a ``JsonFormatter`` to the ``llm.telemetry`` logger once.

    Guards against duplicate handlers with ``if not logger.handlers`` so
    calling this function multiple times (e.g. during test collection) does
    not produce duplicate log lines.

    Returns:
        The configured ``llm.telemetry`` logger instance.

    Postconditions:
        - ``logging.getLogger("llm.telemetry")`` has exactly one handler.
        - The handler uses ``JsonFormatter``.
        - Log level is ``DEBUG`` so all records are emitted.

    Complexity:
        O(1).
    """
    logger = logging.getLogger("llm.telemetry")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger


_logger = _configure_logger()


# ---------------------------------------------------------------------------
# log_llm_call
# ---------------------------------------------------------------------------

def log_llm_call(
    ctx: CallContext,
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    provider_server_latency_ms: Optional[int],
    client_elapsed_ms: float,
    call_start_ts: str,
    call_end_ts: str,
    status: str = "ok",
) -> None:
    """Append an ``llm_call`` record to *ctx* and emit it to the telemetry logger.

    ``throughput_output_tokens_per_sec`` uses ``provider_server_latency_ms``
    as the denominator when it is a **positive non-zero integer** (i.e. the
    provider returned a valid server-timing value); otherwise falls back to
    ``client_elapsed_ms``.  When both are zero or absent the throughput is
    ``0.0`` — no ``ZeroDivisionError`` is raised.

    Args:
        ctx: The per-call ``CallContext`` snapshot (produced by
            ``dataclasses.replace``).  Its ``pipeline``, ``run_id``,
            ``agent``, ``span``, and ``iteration`` fields label the record.
        model: The model identifier string (e.g. ``"gpt-4o-mini"``).
        input_tokens: Number of prompt/input tokens billed for this call.
        output_tokens: Number of completion/output tokens billed for this call.
        provider_server_latency_ms: Server-side processing time in ms from the
            provider HTTP header (e.g. ``openai-processing-ms``).  Pass
            ``None`` when the header is absent, zero, or the provider does not
            expose this metric (e.g. Anthropic).
        client_elapsed_ms: Wall-clock duration of the HTTP call as measured by
            ``time.perf_counter()`` on the client side, in milliseconds.
        call_start_ts: UTC ISO 8601 string captured immediately before the
            API call via ``datetime.now(timezone.utc).isoformat()``.
        call_end_ts: UTC ISO 8601 string captured immediately after the API
            call (or in the ``except`` block on failure).
        status: ``"ok"``, ``"error"``, or ``"partial"`` (tool-loop cap hit).
            Defaults to ``"ok"``.

    Returns:
        None

    Postconditions:
        - Exactly one dict is appended to ``ctx.records``.
        - The dict has ``record_type="llm_call"``.
        - ``input_tokens`` and ``output_tokens`` are stored separately.
        - ``throughput_output_tokens_per_sec`` is a finite non-negative float.
        - ``total_tokens`` is intentionally absent from the record.
        - The record is emitted at ``DEBUG`` level to ``llm.telemetry``.

    Examples:
        >>> ctx = CallContext(pipeline="linear")
        >>> log_llm_call(ctx, model="gpt-4o-mini", input_tokens=100,
        ...     output_tokens=10, provider_server_latency_ms=500,
        ...     client_elapsed_ms=800, call_start_ts="2026-01-01T00:00:00+00:00",
        ...     call_end_ts="2026-01-01T00:00:00.8+00:00")
        >>> ctx.records[0]["throughput_output_tokens_per_sec"]
        20.0

    Complexity:
        O(1).
    """
    denom_ms: float
    if provider_server_latency_ms is not None and provider_server_latency_ms > 0:
        denom_ms = float(provider_server_latency_ms)
    else:
        denom_ms = float(client_elapsed_ms)

    if denom_ms > 0:
        throughput = (output_tokens / denom_ms) * 1000.0
    else:
        throughput = 0.0

    record = {
        "record_type": "llm_call",
        "pipeline": ctx.pipeline,
        "run_id": ctx.run_id,
        "agent": ctx.agent,
        "span": ctx.span,
        "iteration": ctx.iteration,
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "provider_server_latency_ms": provider_server_latency_ms,
        "client_elapsed_ms": round(client_elapsed_ms, 3),
        "throughput_output_tokens_per_sec": round(throughput, 3),
        "call_start_ts": call_start_ts,
        "call_end_ts": call_end_ts,
        "status": status,
    }
    ctx.records.append(record)
    _logger.debug(record)


# ---------------------------------------------------------------------------
# log_tool_execution
# ---------------------------------------------------------------------------

def log_tool_execution(
    ctx: CallContext,
    *,
    tool_name: str,
    tool_execution_ms: float,
    status: str,
    call_start_ts: str,
    call_end_ts: str,
    tool_internal_runtime_ms: Optional[float] = None,
) -> None:
    """Append a ``tool_execution`` record to *ctx* and emit it to the logger.

    Args:
        ctx: The per-call ``CallContext`` snapshot.
        tool_name: Name of the tool executed (e.g.
            ``"compute_elastic_constants_tool"``).
        tool_execution_ms: Wall-clock duration of the ``registry.call(...)``
            invocation in milliseconds (includes container spin-up time).
        status: ``"ok"`` or ``"error"`` — taken from the ``"status"`` key in
            the tool result dict.
        call_start_ts: UTC ISO 8601 string captured immediately before
            ``registry.call``.
        call_end_ts: UTC ISO 8601 string captured immediately after
            ``registry.call``.
        tool_internal_runtime_ms: Optional time in ms reported by the tool's
            own internal framework (e.g. ``result["runtime_seconds"] * 1000``
            for LAMMPS).  Pass ``None`` when absent or when
            ``status == "error"``.  The field is omitted from the record when
            ``None``.

    Returns:
        None

    Postconditions:
        - Exactly one dict is appended to ``ctx.records``.
        - The dict has ``record_type="tool_execution"``.
        - No token fields (``input_tokens``, ``output_tokens``) are present.
        - ``tool_internal_runtime_ms`` is included only when not ``None``.
        - The record is emitted at ``DEBUG`` level to ``llm.telemetry``.

    Examples:
        >>> ctx = CallContext(pipeline="multi_agent")
        >>> log_tool_execution(ctx, tool_name="compute_elastic_constants_tool",
        ...     tool_execution_ms=58400, status="ok",
        ...     call_start_ts="2026-01-01T00:00:00+00:00",
        ...     call_end_ts="2026-01-01T00:00:58.4+00:00",
        ...     tool_internal_runtime_ms=52100)
        >>> ctx.records[0]["record_type"]
        'tool_execution'

    Complexity:
        O(1).
    """
    record: dict = {
        "record_type": "tool_execution",
        "pipeline": ctx.pipeline,
        "run_id": ctx.run_id,
        "agent": ctx.agent,
        "span": ctx.span,
        "tool_name": tool_name,
        "tool_execution_ms": round(tool_execution_ms, 3),
        "call_start_ts": call_start_ts,
        "call_end_ts": call_end_ts,
        "status": status,
    }
    if tool_internal_runtime_ms is not None:
        record["tool_internal_runtime_ms"] = round(tool_internal_runtime_ms, 3)
    ctx.records.append(record)
    _logger.debug(record)


# ---------------------------------------------------------------------------
# log_pipeline_outcome_and_stats
# ---------------------------------------------------------------------------

def log_pipeline_outcome_and_stats(
    ctx: CallContext,
    *,
    total_wall_clock_runtime_ms: float,
    start_ts: datetime,
    end_ts: datetime,
    status: str,
    exception_type: Optional[str] = None,
    exception_message: Optional[str] = None,
) -> None:
    """Emit a ``pipeline_outcome_and_stats`` aggregate record for the run.

    Called **always** by ``orchestrator.run`` and ``coordinator.run`` — both
    when the run succeeds and when it fails — so every run produces exactly one
    summary record.

    Aggregates ``total_input_tokens``, ``total_output_tokens``,
    ``llm_call_count``, and three server-latency coverage fields by iterating
    ``ctx.records`` for records where ``record_type == "llm_call"``.

    Provider-server-latency coverage fields:

    * ``calls_with_provider_server_latency`` — count of ``llm_call`` records
      where ``provider_server_latency_ms`` is a positive non-null integer.
    * ``is_provider_server_latency_complete`` — ``True`` only when every
      ``llm_call`` record has valid server-timing data.  Use this as the
      analysis filter: exclude runs where it is ``False`` from server-side
      throughput / latency calculations.
    * ``total_provider_server_latency_ms`` — sum of all
      ``provider_server_latency_ms`` values, **only** when
      ``is_provider_server_latency_complete`` is ``True``.  Set to ``None``
      when coverage is partial or zero to prevent accidental partial sums.

    Args:
        ctx: The pipeline-level ``CallContext`` (the original, not a snapshot).
        total_wall_clock_runtime_ms: End-to-end wall-clock duration of the run in ms,
            measured with ``time.perf_counter()``.
        start_ts: UTC-aware ``datetime`` captured before the first stage.
        end_ts: UTC-aware ``datetime`` captured after the last stage (or in the
            ``except`` block on failure).
        status: ``"success"`` or ``"error"``.
        exception_type: ``type(exc).__name__`` of the exception that caused the
            run to fail.  Omitted from the record when ``None``.
        exception_message: ``str(exc)`` of the exception.  Omitted when
            ``None``.

    Returns:
        None

    Postconditions:
        - Exactly one dict is appended to ``ctx.records``.
        - The dict has ``record_type="pipeline_outcome_and_stats"``.
        - ``is_partial_data`` is ``True`` iff ``status == "error"``.
        - ``exception_type`` / ``exception_message`` appear only when provided.
        - The record is emitted at ``DEBUG`` level to ``llm.telemetry``.

    Examples:
        >>> import dataclasses, time
        >>> ctx = CallContext(pipeline="linear")
        >>> log_pipeline_outcome_and_stats(
        ...     ctx,
        ...     total_wall_clock_runtime_ms=1200.0,
        ...     start_ts=datetime.now(timezone.utc),
        ...     end_ts=datetime.now(timezone.utc),
        ...     status="success",
        ... )
        >>> ctx.records[-1]["record_type"]
        'pipeline_outcome_and_stats'

    Complexity:
        O(n) where n = len(ctx.records) — single pass to aggregate token sums.
    """
    llm_records = [r for r in ctx.records if r.get("record_type") == "llm_call"]
    llm_call_count = len(llm_records)
    total_input_tokens = sum(r.get("input_tokens", 0) for r in llm_records)
    total_output_tokens = sum(r.get("output_tokens", 0) for r in llm_records)

    calls_with_server_latency = sum(
        1 for r in llm_records
        if isinstance(r.get("provider_server_latency_ms"), int)
        and r["provider_server_latency_ms"] > 0
    )
    is_complete = (
        llm_call_count > 0 and calls_with_server_latency == llm_call_count
    )
    total_server_ms: Optional[int] = (
        sum(r["provider_server_latency_ms"] for r in llm_records)
        if is_complete
        else None
    )

    record: dict = {
        "record_type": "pipeline_outcome_and_stats",
        "pipeline": ctx.pipeline,
        "run_id": ctx.run_id,
        "status": status,
        "is_partial_data": status == "error",
        "total_wall_clock_runtime_ms": round(total_wall_clock_runtime_ms, 3),
        "start_ts": start_ts.isoformat() if isinstance(start_ts, datetime) else start_ts,
        "end_ts": end_ts.isoformat() if isinstance(end_ts, datetime) else end_ts,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "llm_call_count": llm_call_count,
        "calls_with_provider_server_latency": calls_with_server_latency,
        "is_provider_server_latency_complete": is_complete,
        "total_provider_server_latency_ms": total_server_ms,
    }
    if exception_type is not None:
        record["exception_type"] = exception_type
    if exception_message is not None:
        record["exception_message"] = exception_message

    ctx.records.append(record)
    _logger.debug(record)
