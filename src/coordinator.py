"""Coordinator agent: route a raw prompt to simulation, KB, or processor agents.

This module uses the unified LLM wrapper in src.wrapper to classify a prompt
into one of three existing agents and to decide whether parameters should be
structured or passed through. It returns a small decision dict and can also
invoke the executor to fully run the downstream agent.

Telemetry instrumentation:
    ``run`` creates a ``CallContext(pipeline="multi_agent")`` at entry and
    wraps all pipeline stages in a ``try/except`` block.
    ``log_pipeline_outcome_and_stats`` is **always** called — with
    ``status="success"`` when both stages complete or ``status="error"`` (plus
    exception details) when either raises.  This ensures every multi-agent
    pipeline run emits exactly one summary record.

Dependencies:
    dataclasses, time, datetime, json, os, src.wrapper,
    src.llm_pipeline_telemetry.

Pillar compliance:
    - Pillar 1: Functional parity with pre-telemetry code.
    - Pillar 7: Always-emit pattern in try/except; re-raises on error.
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional

from src.llm_pipeline_telemetry import CallContext, log_pipeline_outcome_and_stats
from src.wrapper import complete


AgentType = Literal["simulation", "kb", "processor"]
ModeType = Literal["pass_through", "structured"]


def _validate_runtime_environment() -> None:
    """Validate that the runtime environment is properly configured.

    Raises:
        RuntimeError: If LLM_PROVIDER is invalid or the required API key is absent.
    """
    provider = (os.environ.get("LLM_PROVIDER", "openai") or "openai").strip().lower()
    if provider not in ("openai", "anthropic"):
        raise RuntimeError("Invalid LLM_PROVIDER. Expected 'openai' or 'anthropic'.")
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Missing required environment variable: OPENAI_API_KEY")
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Missing required environment variable: ANTHROPIC_API_KEY")


def _build_routing_messages(prompt: str) -> list[dict]:
    system_content = """
You are a routing controller for an LLM application.
There are exactly three internal agents:
- simulation: runs a nickel-based superalloy optimization loop via a SimulationAgent.
- kb: answers general questions using a knowledge-base agent with web-search fallback.
- processor: runs a structured materials/simulation analysis pipeline (extraction + processing).

Given the user's prompt, choose the most appropriate agent and whether the call should be:
- pass_through: send the raw prompt as-is (or as a simple "query") to the agent.
- structured: derive a small JSON params object tailored to that agent.

Respond ONLY with a single JSON object, no explanations, matching this schema:
{"agent": "simulation" | "kb" | "processor",
 "mode": "pass_through" | "structured",
 "params": {}}
Keep params minimal and only include keys the downstream agent can meaningfully use.
""".strip()
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]


def _classify_with_llm(prompt: str, ctx: Optional[CallContext] = None) -> str:
    """Classify the prompt using the LLM and return the raw JSON string.

    Args:
        prompt: Raw user prompt.
        ctx: Optional ``CallContext`` snapshot for telemetry attribution.

    Returns:
        Raw LLM response string (expected to be a JSON object).
    """
    _validate_runtime_environment()
    messages = _build_routing_messages(prompt)
    return complete(messages, ctx=ctx) or ""


def _default_decision(prompt: str) -> Dict[str, Any]:
    return {
        "agent": "kb",
        "mode": "pass_through",
        "params": {"query": prompt},
    }


def _parse_decision(raw: str, prompt: str) -> Dict[str, Any]:
    if not raw or not isinstance(raw, str):
        return _default_decision(prompt)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return _default_decision(prompt)
    try:
        data = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return _default_decision(prompt)
    agent = data.get("agent")
    mode = data.get("mode")
    params = data.get("params")
    if agent not in ("simulation", "kb", "processor"):
        agent = "kb"
    if mode not in ("pass_through", "structured"):
        mode = "pass_through"
    if not isinstance(params, dict):
        params = {}
    if agent == "kb" and "query" not in params:
        params["query"] = prompt
    if agent == "processor" and "input_text" not in params:
        params["input_text"] = prompt
    return {"agent": agent, "mode": mode, "params": params}


def route_prompt(prompt: str) -> Dict[str, Any]:
    """Classify prompt and return a routing decision dict.

    Args:
        prompt: Raw user prompt.

    Returns:
        Decision dict with ``agent``, ``mode``, and ``params`` keys.
    """
    raw = _classify_with_llm(prompt)
    return _parse_decision(raw, prompt)


def run(prompt: str) -> Dict[str, Any]:
    """Route the prompt and execute the selected downstream agent.

    Always emits a ``pipeline_outcome_and_stats`` record — on both success and
    failure.  On failure the exception is re-raised after the record is emitted.

    Args:
        prompt: Raw user prompt.

    Returns:
        Result dict from ``executor.execute`` on success.

    Raises:
        Any exception from ``_classify_with_llm`` or ``execute`` is re-raised
        after the error record is emitted.

    Postconditions:
        - Exactly one ``pipeline_outcome_and_stats`` record is emitted.

    Complexity:
        O(1) for routing; downstream agent complexity varies.
    """
    from src.executor import execute

    ctx = CallContext(pipeline="multi_agent")
    start_ts = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    try:
        routing_ctx = dataclasses.replace(ctx, agent="coordinator", span="routing")
        raw = _classify_with_llm(prompt, ctx=routing_ctx)
        """Decide which agent to spin up"""
        decision = _parse_decision(raw, prompt)
        """Have the executor spin up the agent and run it, and store the result"""
        result = execute(decision, original_prompt=prompt, ctx=ctx)
        log_pipeline_outcome_and_stats(
            ctx,
            total_wall_clock_runtime_ms=(time.perf_counter() - t0) * 1000,
            start_ts=start_ts,
            end_ts=datetime.now(timezone.utc),
            status="success",
        )
        return result
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
