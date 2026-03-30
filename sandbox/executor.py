"""Sandbox executor for coordinator-managed runs."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from sandbox import extractor, kb_agent, processor, sim_agent

_KB_STATE: dict[str, str | None] = {"ready_scale": None}


def _safe_float(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _safe_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _executor_workers(config: dict[str, Any]) -> int:
    if str(config.get("worker_mode", "max")).strip().lower() == "fixed":
        if isinstance(config.get("max_workers"), int) and config["max_workers"] > 0:
            return config["max_workers"]
        raise ValueError("config.max_workers must be a positive integer when worker_mode=fixed")
    return min(32, (os.cpu_count() or 1) + 4)


def _ensure_kb(scale: str) -> dict[str, Any] | None:
    if _KB_STATE["ready_scale"] == scale:
        return None
    _KB_STATE["ready_scale"] = scale
    return kb_agent.preload(scale=scale, emit_artifacts=False)


def _processor_pipeline(payload: str, config: dict[str, Any]) -> tuple[dict[str, Any], float, int, int]:
    tasks = [task for task in config.get("processor_tasks", list(processor.TASKS)) if task in processor.TASKS]
    extraction_output = extractor.extract(payload, emit_artifacts=False)
    task_outputs = [processor.process(extraction_output["extraction"], task, emit_artifacts=False) for task in tasks]
    return (
        {
            "extraction": extraction_output["extraction"],
            "extractor_output": extraction_output,
            "processor_outputs": task_outputs,
        },
        _safe_float(extraction_output.get("elapsed_ms")) + sum(_safe_float(item.get("elapsed_ms")) for item in task_outputs),
        _safe_int(extraction_output.get("input_tokens")) + sum(_safe_int(item.get("input_tokens")) for item in task_outputs),
        _safe_int(extraction_output.get("output_tokens")) + sum(_safe_int(item.get("output_tokens")) for item in task_outputs),
    )


def execute_item(input_id: str, payload: str, agent: str, config: dict[str, Any]) -> dict[str, Any]:
    """Execute one routed job and return a normalized envelope."""
    t0 = time.perf_counter()
    try:
        if agent == "extractor":
            result = extractor.extract(payload, emit_artifacts=False)
            return {
                "input_id": input_id,
                "payload": payload,
                "agent": agent,
                "status": "success",
                "elapsed_ms": _safe_float(result.get("elapsed_ms")),
                "input_tokens": _safe_int(result.get("input_tokens")),
                "output_tokens": _safe_int(result.get("output_tokens")),
                "result": result,
            }
        if agent == "processor":
            result, elapsed_ms, input_tokens, output_tokens = _processor_pipeline(payload, config)
            return {
                "input_id": input_id,
                "payload": payload,
                "agent": agent,
                "status": "success",
                "elapsed_ms": elapsed_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "result": result,
            }
        if agent == "kb":
            preload_output = _ensure_kb(str(config.get("kb_scale", "small")).strip().lower())
            result = kb_agent.ask(payload, emit_artifacts=False)
            return {
                "input_id": input_id,
                "payload": payload,
                "agent": agent,
                "status": "success",
                "elapsed_ms": _safe_float(result.get("elapsed_ms")) + _safe_float((preload_output or {}).get("elapsed_ms")),
                "input_tokens": _safe_int(result.get("input_tokens")) + _safe_int((preload_output or {}).get("input_tokens")),
                "output_tokens": _safe_int(result.get("output_tokens")) + _safe_int((preload_output or {}).get("output_tokens")),
                "result": {"kb_output": result, "preload_output": preload_output},
            }
        if str(config.get("sim_mode", "regular")).strip().lower() == "real":
            result = sim_agent.run_real_simulation(payload, emit_artifacts=False)
        else:
            result = sim_agent.run_regular_simulation(
                initial_cooling_rate_K_per_min=float(config.get("sim_regular", {}).get("initial_cooling_rate_K_per_min", 15.0)),
                max_iterations=int(config.get("sim_regular", {}).get("max_iterations", 3)),
                duration_hours=float(config.get("sim_regular", {}).get("duration_hours", 4.0)),
                emit_artifacts=False,
            )
        return {
            "input_id": input_id,
            "payload": payload,
            "agent": "simulation",
            "status": "success",
            "elapsed_ms": _safe_float(result.get("elapsed_ms")),
            "input_tokens": _safe_int(result.get("input_tokens")),
            "output_tokens": _safe_int(result.get("output_tokens")),
            "result": result,
        }
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return {
            "input_id": input_id,
            "payload": payload,
            "agent": agent,
            "status": "error",
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 3),
            "input_tokens": 0,
            "output_tokens": 0,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }


def execute_jobs(jobs: list[dict[str, str]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute routed jobs in linear or parallel mode.

    In parallel mode this returns completion order.
    """
    if str(config.get("execution_mode", "linear")).strip().lower() == "linear" or len(jobs) < 2:
        return [execute_item(job["input_id"], job["payload"], job["agent"], config) for job in jobs]
    with ThreadPoolExecutor(max_workers=_executor_workers(config)) as pool:
        return [
            future.result()
            for future in as_completed(
                [pool.submit(execute_item, job["input_id"], job["payload"], job["agent"], config) for job in jobs]
            )
        ]
