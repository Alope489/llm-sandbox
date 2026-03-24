"""Executor: run the selected downstream agent based on a coordinator decision.

This module receives a decision dict of the form:
    {"agent": "simulation" | "kb" | "processor",
     "mode": "pass_through" | "structured",
     "params": {...}}
and executes the appropriate downstream agent.

Telemetry instrumentation:
    ``execute`` accepts an optional ``ctx`` (``CallContext``) and threads it
    to each private runner.  Each runner produces its own
    ``dataclasses.replace()`` snapshot with the appropriate ``agent`` label
    before forwarding.  The shared ``ctx`` is never mutated at this layer.

Dependencies:
    dataclasses, os, src.multi.sim.agent, src.multi.kb_agent,
    src.linear.orchestrator, src.llm_pipeline_telemetry.

Pillar compliance:
    - Pillar 4: No hardcoding; agent names are module-level constants.
    - Pillar 7: All exceptions are caught and returned as error dicts.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Callable, Dict, Optional

from src.multi.sim.agent import SimulationAgent
import src.multi.kb_agent as kb_agent
from src.linear import orchestrator as linear
from src.llm_pipeline_telemetry import CallContext


ALLOWED_AGENTS = ("simulation", "kb", "processor")
SIM_MODE_MOCK = "mock_sim_mode"
SIM_MODE_REAL = "real_sim_mode"
_ALLOWED_SIM_MODES = (SIM_MODE_MOCK, SIM_MODE_REAL)
AgentRunner = Callable[[Dict[str, Any], Optional[str], Optional[CallContext]], Any]


def _validate_runtime_environment(agent: str, params: Dict[str, Any]) -> None:
    provider = params.get("provider")
    if isinstance(provider, str) and provider.strip():
        provider = provider.strip().lower()
    else:
        provider = (
            (os.environ.get("LLM_PROVIDER", "openai") or "openai").strip().lower()
        )

    if provider not in ("openai", "anthropic"):
        raise RuntimeError(
            "Invalid LLM_PROVIDER/provider. Expected 'openai' or 'anthropic'."
        )

    if agent in ALLOWED_AGENTS:
        if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Missing required environment variable: OPENAI_API_KEY")
        if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "Missing required environment variable: ANTHROPIC_API_KEY"
            )


def _coerce_positive_float(value: Any, *, name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive number")
    try:
        result = float(value)
    except TypeError, ValueError:
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
    except TypeError, ValueError:
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
        duration_hours = _coerce_positive_float(
            params.get("duration_hours"), name="duration_hours"
        )
    except ValueError:
        duration_hours = None
    if duration_hours is not None:
        sanitized["duration_hours"] = duration_hours

    try:
        max_iterations = _coerce_positive_int(
            params.get("max_iterations"), name="max_iterations"
        )
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


def _apply_mode(
    agent: str, mode: str, params: Dict[str, Any], original_prompt: Optional[str]
) -> Dict[str, Any]:
    if mode == "structured":
        return params
    if agent == "kb":
        return {"query": original_prompt or params.get("query", "")}
    if agent == "processor":
        return {"input_text": original_prompt or params.get("input_text", "")}
    if agent == "simulation":
        return {}
    return params


def _execute_simulation(
    params: Dict[str, Any],
    original_prompt: Optional[str] = None,
    ctx: Optional[CallContext] = None,
) -> Dict[str, Any]:
    """Execute the simulation agent, routing to the appropriate mode.

    Reads ``CURRENT_SIMULATION_MODE`` from the environment to decide the
    execution path:

    - ``"mock_sim_mode"`` (default): instantiates ``SimulationAgent`` and
      calls ``run_and_report``, returning a full optimization history.
    - ``"real_sim_mode"``: instantiates ``SimulationAgent`` and calls
      ``perform_real_simulation`` only, returning the pre-computation
      summary without running the optimization loop.

    The two paths are mutually exclusive — exactly one is taken per call.

    Args:
        params: Sanitized simulation parameters. Recognised keys:
            ``provider``, ``duration_hours``, ``max_iterations``,
            ``initial_cooling_rate_K_per_min``.
        original_prompt: Unused; accepted for interface uniformity.
        ctx: Optional ``CallContext`` propagated from the coordinator.
            A snapshot labelled ``agent="sim_agent"`` is forwarded.

    Returns:
        ``{"history": list, "output": str}`` when in ``mock_sim_mode``, or
        ``{"prefetch_output": str}`` when in ``real_sim_mode``.

    Raises:
        ValueError: If ``CURRENT_SIMULATION_MODE`` is set to a value not
            in ``_ALLOWED_SIM_MODES``.

    Preconditions:
        - ``params`` has already been sanitised by
          ``_sanitize_simulation_params``.

    Complexity:
        Θ(max_iterations) for ``mock_sim_mode``; Θ(1) LLM calls for
        ``real_sim_mode`` (tool-calling loop is bounded by
        ``MAX_TOOL_CALLS``).
    """
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

    sim_ctx = dataclasses.replace(ctx, agent="sim_agent") if ctx is not None else None

    sim_mode = (
        (os.environ.get("CURRENT_SIMULATION_MODE") or SIM_MODE_MOCK).strip().lower()
    )
    if sim_mode not in _ALLOWED_SIM_MODES:
        raise ValueError(
            f"Invalid CURRENT_SIMULATION_MODE {sim_mode!r}. "
            f"Allowed: {_ALLOWED_SIM_MODES}"
        )

    if sim_mode == SIM_MODE_REAL:
        prefetch_result = agent.perform_real_simulation()
        # Here we return the result from a real simulation
        return {"prefetch_output": prefetch_result}
    elif sim_mode == SIM_MODE_MOCK:
        initial_rate = safe_params.get("initial_cooling_rate_K_per_min")
        if initial_rate is None:
            history, output = agent.run_and_report(ctx=sim_ctx)
        else:
            history, output = agent.run_and_report(
                initial_cooling_rate_K_per_min=initial_rate, ctx=sim_ctx
            )
        return {"history": history, "output": output}
    else:
        return {
            "error": {
                "type": "ValueError",
                "message": f"Invalid CURRENT_SIMULATION_MODE {sim_mode!r}. Allowed: {_ALLOWED_SIM_MODES}",
            }
        }


def _execute_kb(
    params: Dict[str, Any],
    original_prompt: Optional[str] = None,
    ctx: Optional[CallContext] = None,
) -> str:
    query = params.get("query") or original_prompt or ""
    return kb_agent.ask(query, ctx=ctx)


def _execute_processor(
    params: Dict[str, Any],
    original_prompt: Optional[str] = None,
    ctx: Optional[CallContext] = None,
) -> Dict[str, Any]:
    del ctx  # processor path forks into a separate linear run with its own context
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


def execute(
    decision: Dict[str, Any],
    original_prompt: Optional[str] = None,
    ctx: Optional[CallContext] = None,
) -> Dict[str, Any]:
    """Execute the downstream agent specified by *decision*.

    Args:
        decision: Routing decision dict with ``agent``, ``mode``, and
            ``params`` keys.
        original_prompt: The raw user prompt forwarded to agents that need it.
        ctx: Optional ``CallContext`` propagated from ``coordinator.run``.
            Each runner creates a ``dataclasses.replace()`` snapshot with its
            own ``agent`` label before forwarding.

    Returns:
        ``{"agent": ..., "mode": ..., "result": ...}`` on success, or
        ``{"agent": ..., "mode": ..., "error": {...}}`` on failure.

    Postconditions:
        - The shared ``ctx`` is never mutated.
        - On exception, the error is captured in the return dict (not raised).

    Complexity:
        O(1) for dispatch; downstream agent complexity varies.
    """
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
        # Run the relevant agent with the appropriate parameters, and save the result
        result = AGENT_REGISTRY[agent](effective_params, original_prompt, ctx)
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
