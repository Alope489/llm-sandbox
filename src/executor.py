"""Executor: run the selected downstream agent based on a coordinator decision.

This module receives a decision dict of the form:
    {"agent": "simulation" | "kb" | "processor",
     "mode": "pass_through" | "structured",
     "params": {...}}
and executes the appropriate existing agent:
    - simulation: src.multi.sim.agent.SimulationAgent
    - kb       : src.multi.kb_agent.ask
    - processor: src.linear.orchestrator.run
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.multi.sim.agent import SimulationAgent
import src.multi.kb_agent as kb_agent
import src.linear.orchestrator as linear


def _execute_simulation(params: Dict[str, Any]) -> Dict[str, Any]:
    provider = params.get("provider")
    duration_hours = params.get("duration_hours")
    max_iterations = params.get("max_iterations")
    if duration_hours is None and max_iterations is None and provider is None:
        agent = SimulationAgent()
    else:
        agent = SimulationAgent(
            provider=provider,
            duration_hours=duration_hours if duration_hours is not None else 4.0,
            max_iterations=max_iterations if max_iterations is not None else 10,
        )
    initial_rate = params.get("initial_cooling_rate_K_per_min")
    if initial_rate is None:
        history, output = agent.run_and_report()
    else:
        history, output = agent.run_and_report(
            initial_cooling_rate_K_per_min=float(initial_rate)
        )
    return {"history": history, "output": output}


def _execute_kb(params: Dict[str, Any], original_prompt: Optional[str]) -> str:
    query = params.get("query") or original_prompt or ""
    return kb_agent.ask(query)


def _execute_processor(params: Dict[str, Any], original_prompt: Optional[str]) -> Dict[str, Any]:
    input_text = params.get("input_text") or original_prompt or ""
    tasks = params.get("tasks")
    return linear.run(input_text, tasks=tasks)


def execute(decision: Dict[str, Any], original_prompt: Optional[str] = None) -> Dict[str, Any]:
    agent = decision.get("agent", "kb")
    mode = decision.get("mode", "pass_through")
    params = decision.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    if agent == "simulation":
        result = _execute_simulation(params)
        return {"agent": "simulation", "mode": mode, "result": result}
    if agent == "processor":
        result = _execute_processor(params, original_prompt)
        return {"agent": "processor", "mode": mode, "result": result}
    result = _execute_kb(params, original_prompt)
    return {"agent": "kb", "mode": mode, "result": result}

