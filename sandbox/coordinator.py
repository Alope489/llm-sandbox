"""Sandbox coordinator with queue-based linear/parallel orchestration."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from sandbox import reasoning
from sandbox.executor import execute_jobs

_DEFAULT_CONFIG_PATH = os.path.join("sandbox", "config.json")
_DEFAULT_QUEUE_PATH = os.path.join("sandbox", "prompts", "coordinator_queue.txt")
_ALLOWED_AGENTS = ("extractor", "processor", "kb", "simulation")


def load_config(path: str = _DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    defaults = {
        "execution_mode": "parallel",
        "worker_mode": "max",
        "max_workers": 4,
        "kb_scale": "small",
        "sim_mode": "regular",
        "sim_regular": {"initial_cooling_rate_K_per_min": 15.0, "max_iterations": 3, "duration_hours": 4.0},
        "processor_tasks": [
            "schema_validation",
            "constraint_verification",
            "feature_extraction",
            "normalization",
            "risk_ranking",
        ],
    }
    if not os.path.exists(path):
        return defaults
    with open(path, "r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return {
        **defaults,
        **({key: value for key, value in loaded.items() if key in defaults} if isinstance(loaded, dict) else {}),
        "sim_regular": {
            **defaults["sim_regular"],
            **((loaded or {}).get("sim_regular", {}) if isinstance(loaded, dict) else {}),
        },
    }


def _chunks(text: str) -> list[str]:
    if "\n---\n" in text:
        return [chunk.strip() for chunk in text.split("\n---\n") if chunk.strip()]
    return [line.strip() for line in text.splitlines() if line.strip()]


def _parse_queue_record(raw: str, index: int) -> dict[str, str]:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("Queue records must be JSON objects")
    if not isinstance(data.get("payload"), str) or not data["payload"].strip():
        raise ValueError("Queue record must include non-empty string payload")
    if not isinstance(data.get("input_id"), str) or not data["input_id"].strip():
        raise ValueError("Queue record must include non-empty string input_id")
    return {
        "input_id": data["input_id"].strip(),
        "payload": data["payload"].strip(),
        "source_hint": str(data.get("source_hint", "")).strip(),
        "route_hint": str(data.get("route_hint", "")).strip(),
        "position": str(index),
    }


def load_queue(path: str = _DEFAULT_QUEUE_PATH) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [_parse_queue_record(raw, index) for index, raw in enumerate(_chunks(handle.read()), start=1)]


def route(payload: str, source_hint: str = "", route_hint: str = "") -> str:
    if route_hint.strip().lower() in _ALLOWED_AGENTS:
        return route_hint.strip().lower()
    if "extractor" in source_hint.strip().lower():
        return "extractor"
    if "processor" in source_hint.strip().lower():
        return "processor"
    if "kb" in source_hint.strip().lower() or "query" in source_hint.strip().lower():
        return "kb"
    text = payload.strip().lower()
    if any(word in text for word in ("simulate", "simulation", "supercell", "elastic constant", "cooling rate")):
        return "simulation"
    if any(word in text for word in ("schema", "constraint", "normalize", "risk ranking", "feature extraction", "processor")):
        return "processor"
    if any(word in text for word in ("extract", "structured data", "json schema")):
        return "extractor"
    return "kb"


def _reasoning_inputs(run_item: dict[str, Any]) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    if run_item.get("agent") == "extractor" and run_item.get("status") == "success":
        return (
            run_item["payload"],
            run_item.get("result", {}).get("extraction", {}),
            [{"producer_file": "extractor.py", **run_item.get("result", {})}],
        )
    if run_item.get("agent") == "processor" and run_item.get("status") == "success":
        return (
            run_item["payload"],
            run_item.get("result", {}).get("extraction", {}),
            [{"producer_file": "processor.py", **run_item.get("result", {})}],
        )
    if run_item.get("status") == "success":
        return run_item["payload"], {}, [{"producer_file": f"{run_item.get('agent')}.py", **run_item.get("result", {})}]
    return run_item["payload"], {}, [{"producer_file": "coordinator.py", "error": run_item.get("error", {}), "agent": run_item.get("agent")}]


def run(queue_path: str = _DEFAULT_QUEUE_PATH, config_path: str = _DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Run queue orchestration and feed executor outputs to reasoning sequentially."""
    config = load_config(config_path)
    jobs = [
        {"input_id": item["input_id"], "payload": item["payload"], "agent": route(item["payload"], item["source_hint"], item["route_hint"])}
        for item in load_queue(queue_path)
    ]
    t0 = time.perf_counter()
    per_input_runs = execute_jobs(jobs, config)
    reasoning_runs = []
    for run_item in per_input_runs:
        original_input, extraction, agent_outputs = _reasoning_inputs(run_item)
        reasoning_output = reasoning.summarize(
            original_input=original_input,
            extraction=extraction,
            agent_outputs=agent_outputs,
            emit_artifacts=False,
        )
        reasoning_runs.append(
            {
                "input_id": run_item["input_id"],
                "agent": run_item["agent"],
                "summary": reasoning_output.get("summary", ""),
                "elapsed_ms": reasoning_output.get("elapsed_ms", 0),
                "input_tokens": reasoning_output.get("input_tokens", 0),
                "output_tokens": reasoning_output.get("output_tokens", 0),
            }
        )
    return {
        "config": config,
        "execution_mode": config["execution_mode"],
        "completion_order_input_ids": [item["input_id"] for item in per_input_runs],
        "routed_jobs": jobs,
        "per_input_runs": per_input_runs,
        "reasoning_runs": reasoning_runs,
        "final_reasoning_response": (reasoning_runs[-1]["summary"] if reasoning_runs else ""),
        "aggregate": {
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 3),
            "input_tokens": sum(int(item.get("input_tokens", 0)) for item in per_input_runs) + sum(int(item.get("input_tokens", 0)) for item in reasoning_runs),
            "output_tokens": sum(int(item.get("output_tokens", 0)) for item in per_input_runs) + sum(int(item.get("output_tokens", 0)) for item in reasoning_runs),
            "count": len(per_input_runs),
        },
    }
