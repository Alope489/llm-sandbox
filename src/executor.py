"""Executor: run the selected downstream agent based on a coordinator decision.

This module receives a decision dict of the form:
    {"agent": "simulation" | "kb" | "processor",
     "mode": "pass_through" | "structured",
     "params": {...}}
and executes the appropriate downstream agent.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

from src.multi.sim.agent import SimulationAgent
import src.multi.kb_agent as kb_agent
from src.linear import orchestrator as linear


ALLOWED_AGENTS = ("simulation", "kb", "processor")
AgentRunner = Callable[[Dict[str, Any], Optional[str]], Any]


def _validate_runtime_environment(agent: str, params: Dict[str, Any]) -> None:
    provider = params.get("provider")
    if isinstance(provider, str) and provider.strip():
        provider = provider.strip().lower()
    else:
        provider = (os.environ.get("LLM_PROVIDER", "openai") or "openai").strip().lower()

    if provider not in ("openai", "anthropic"):
        raise RuntimeError("Invalid LLM_PROVIDER/provider. Expected 'openai' or 'anthropic'.")

    if agent in ALLOWED_AGENTS:
        if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Missing required environment variable: OPENAI_API_KEY")
        if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Missing required environment variable: ANTHROPIC_API_KEY")


def _coerce_positive_float(value: Any, *, name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive number")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive number")
    if result <= 0:
        raise ValueError(f"{name} must be a positive number")
    return result


def _coerce_positive_int(value: Any, *, name: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        result = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a positive integer")
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _sanitize_simulation_params(params: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}

    provider = params.get("provider")
    if isinstance(provider, str):
        provider = provider.strip().lower()
        if provider in ("openai", "anthropic"):
            sanitized["provider"] = provider

    try:
        duration_hours = _coerce_positive_float(params.get("duration_hours"), name="duration_hours")
    except ValueError:
        duration_hours = None
    if duration_hours is not None:
        sanitized["duration_hours"] = duration_hours

    try:
        max_iterations = _coerce_positive_int(params.get("max_iterations"), name="max_iterations")
    except ValueError:
        max_iterations = None
    if max_iterations is not None:
        sanitized["max_iterations"] = max_iterations

    try:
        initial_rate = _coerce_positive_float(
            params.get("initial_cooling_rate_K_per_min"),
            name="initial_cooling_rate_K_per_min",
        )
    except ValueError:
        initial_rate = None
    if initial_rate is not None:
        sanitized["initial_cooling_rate_K_per_min"] = initial_rate

    return sanitized


def _apply_mode(agent: str, mode: str, params: Dict[str, Any], original_prompt: Optional[str]) -> Dict[str, Any]:
    if mode == "structured":
        return params
    if agent == "kb":
        return {"query": original_prompt or params.get("query", "")}
    if agent == "processor":
        return {"input_text": original_prompt or params.get("input_text", "")}
    if agent == "simulation":
        return {}
    return params


def _execute_simulation(params: Dict[str, Any], original_prompt: Optional[str] = None) -> Dict[str, Any]:
    del original_prompt
    safe_params = _sanitize_simulation_params(params)
    provider = safe_params.get("provider")
    duration_hours = safe_params.get("duration_hours")
    max_iterations = safe_params.get("max_iterations")

    if duration_hours is None and max_iterations is None and provider is None:
        agent = SimulationAgent()
    else:
        agent = SimulationAgent(
            provider=provider,
            duration_hours=duration_hours if duration_hours is not None else 4.0,
            max_iterations=max_iterations if max_iterations is not None else 10,
        )

    initial_rate = safe_params.get("initial_cooling_rate_K_per_min")
    if initial_rate is None:
        history, output = agent.run_and_report()
    else:
        history, output = agent.run_and_report(initial_cooling_rate_K_per_min=initial_rate)
    return {"history": history, "output": output}


def _execute_kb(params: Dict[str, Any], original_prompt: Optional[str]) -> str:
    query = params.get("query") or original_prompt or ""
    return kb_agent.ask(query)


def _execute_processor(params: Dict[str, Any], original_prompt: Optional[str]) -> Dict[str, Any]:
    input_text = params.get("input_text") or original_prompt or ""
    tasks = params.get("tasks")
    single_task = params.get("task")
    if single_task is not None:
        tasks = single_task
    if tasks is None:
        task_list = None
    elif isinstance(tasks, str):
        task_list = [tasks]
    elif isinstance(tasks, (list, tuple)):
        task_list = [t for t in tasks if isinstance(t, str)] or None
    else:
        task_list = None
    return linear.run(input_text, tasks=task_list)


AGENT_REGISTRY: Dict[str, AgentRunner] = {
    "simulation": _execute_simulation,
    "kb": _execute_kb,
    "processor": _execute_processor,
}


def execute(decision: Dict[str, Any], original_prompt: Optional[str] = None) -> Dict[str, Any]:
    agent = decision.get("agent", "kb")
    mode = decision.get("mode", "pass_through")
    params = decision.get("params") or {}
    if not isinstance(params, dict):
        params = {}

    if agent not in AGENT_REGISTRY:
        agent = "kb"

    try:
        effective_params = _apply_mode(agent, mode, params, original_prompt)
        _validate_runtime_environment(agent, effective_params)
        result = AGENT_REGISTRY[agent](effective_params, original_prompt)
        return {"agent": agent, "mode": mode, "result": result}
    except Exception as exc:
        return {
            "agent": agent,
            "mode": mode,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
