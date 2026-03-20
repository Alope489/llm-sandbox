"""Orchestrates the linear pipeline: extract → process (one or more tasks) → reasoning summary.

Telemetry instrumentation:
    ``run`` creates a ``CallContext(pipeline="linear")`` at entry and wraps all
    pipeline stages in a ``try/except`` block.  ``log_pipeline_outcome_and_stats``
    is **always** called — with ``status="success"`` when all stages complete or
    ``status="error"`` (plus exception details) when any stage raises.  This
    ensures every linear pipeline run emits exactly one summary record so
    failure rates can be computed without survivorship bias.

Dependencies:
    dataclasses, time, datetime, src.linear.{extractor,processor,reasoning},
    src.llm_pipeline_telemetry.

Pillar compliance:
    - Pillar 1: Functional parity with pre-telemetry code.
    - Pillar 7: Always-emit pattern in try/except; re-raises on error.
"""
from __future__ import annotations

import dataclasses
import time
from datetime import datetime, timezone

from src.linear.extractor import extract
from src.linear.processor import TASKS, process
from src.linear.reasoning import summarize
from src.llm_pipeline_telemetry import CallContext, log_pipeline_outcome_and_stats


def run(input_text: str, tasks: list[str] | None = None) -> dict:
    """Run the full pipeline: extract from input, run processor tasks, produce human-readable summary.

    Always emits a ``pipeline_outcome_and_stats`` record — on both success and
    failure.  On failure the exception is re-raised after the record is emitted.

    Args:
        input_text: Raw task description (e.g. material/simulation prompt).
        tasks: Processor task names to run.  If ``None``, all ``TASKS`` are run.

    Returns:
        ``{"summary": str, "extraction": dict, "processing": dict}`` on success.

    Raises:
        Any exception raised by ``extract``, ``process``, or ``summarize`` is
        re-raised after the error record is emitted.

    Postconditions:
        - Exactly one ``pipeline_outcome_and_stats`` record is appended to the
          internal ``CallContext.records``.  Callers that need access to
          telemetry records should instrument at the caller level or inject a
          ``CallContext`` via future refactoring; the current API does not
          expose ``ctx`` to callers.

    Complexity:
        O(|tasks|) — one LLM call per task plus one for extract and one for summarize.
    """
    ctx = CallContext(pipeline="linear")
    start_ts = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    try:
        extraction = extract(
            input_text,
            ctx=dataclasses.replace(ctx, agent="extractor", span="extract"),
        )
        task_list = tasks if tasks is not None else list(TASKS)
        processing = {
            t: process(
                extraction,
                t,
                ctx=dataclasses.replace(ctx, agent="processor", span=f"process.{t}"),
            )
            for t in task_list
        }
        summary = summarize(
            input_text,
            extraction,
            processing,
            ctx=dataclasses.replace(ctx, agent="reasoning", span="summarize"),
        )
        log_pipeline_outcome_and_stats(
            ctx,
            total_wall_clock_runtime_ms=(time.perf_counter() - t0) * 1000,
            start_ts=start_ts,
            end_ts=datetime.now(timezone.utc),
            status="success",
        )
        return {"summary": summary, "extraction": extraction, "processing": processing}
    except Exception as exc:
        log_pipeline_outcome_and_stats(
            ctx,
            total_wall_clock_runtime_ms=(time.perf_counter() - t0) * 1000,
            start_ts=start_ts,
            end_ts=datetime.now(timezone.utc),
            status="error",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
        )
        raise
