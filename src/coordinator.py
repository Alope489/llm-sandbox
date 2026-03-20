"""Coordinator agent: route a raw prompt to simulation, KB, processor, or materials agents.

This module uses the unified LLM wrapper in src.wrapper to classify a prompt
into one of three existing agents and to decide whether parameters should be
structured or passed through. It returns a small decision dict and can also
invoke the executor to fully run the downstream agent.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Literal

from src.wrapper import complete


AgentType = Literal["simulation", "kb", "processor", "materials"]
ModeType = Literal["pass_through", "structured"]
_MATERIALS_KEYWORDS = (
    "lattice",
    "crystal",
    "metal",
    "alloy",
    "materials project",
)


def _validate_runtime_environment() -> None:
    provider = (os.environ.get("LLM_PROVIDER", "openai") or "openai").strip().lower()
    if provider not in ("openai", "anthropic"):
        raise RuntimeError(
            "Invalid LLM_PROVIDER. Expected 'openai' or 'anthropic'."
        )
    if provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("Missing required environment variable: OPENAI_API_KEY")
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Missing required environment variable: ANTHROPIC_API_KEY")


def _build_routing_messages(prompt: str) -> list[dict]:
    system_content = """
You are a routing controller for an LLM application.
There are exactly four internal agents:
- simulation: runs a nickel-based superalloy optimization loop via a SimulationAgent.
- kb: answers general questions using a knowledge-base agent with web-search fallback.
- processor: runs a structured materials/simulation analysis pipeline (extraction + processing).
- materials: runs a materials-science step via AtomAgent.

Given the user's prompt, choose the most appropriate agent and whether the call should be:
- pass_through: send the raw prompt as-is (or as a simple "query") to the agent.
- structured: derive a small JSON params object tailored to that agent.

Respond ONLY with a single JSON object, no explanations, matching this schema:
{"agent": "simulation" | "kb" | "processor" | "materials",
 "mode": "pass_through" | "structured",
 "params": {}}
Keep params minimal and only include keys the downstream agent can meaningfully use.
""".strip()
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]


def _classify_with_llm(prompt: str) -> str:
    _validate_runtime_environment()
    messages = _build_routing_messages(prompt)
    return complete(messages) or ""


def _default_decision(prompt: str) -> Dict[str, Any]:
    return {
        "agent": "kb",
        "mode": "pass_through",
        "params": {"query": prompt},
    }


def _is_materials_prompt(prompt: str) -> bool:
    lowered = (prompt or "").strip().lower()
    return any(keyword in lowered for keyword in _MATERIALS_KEYWORDS)


def _materials_decision(prompt: str) -> Dict[str, Any]:
    return {
        "agent": "materials",
        "mode": "pass_through",
        "params": {"prompt": prompt},
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
    if agent not in ("simulation", "kb", "processor", "materials"):
        agent = "kb"
    if mode not in ("pass_through", "structured"):
        mode = "pass_through"
    if not isinstance(params, dict):
        params = {}
    if agent == "kb" and "query" not in params:
        params["query"] = prompt
    if agent == "processor" and "input_text" not in params:
        params["input_text"] = prompt
    if agent == "materials" and "prompt" not in params:
        params["prompt"] = prompt
    return {"agent": agent, "mode": mode, "params": params}


def route_prompt(prompt: str) -> Dict[str, Any]:
    if _is_materials_prompt(prompt):
        return _materials_decision(prompt)
    raw = _classify_with_llm(prompt)
    return _parse_decision(raw, prompt)


def run(prompt: str) -> Dict[str, Any]:
    from src.executor import execute

    decision = route_prompt(prompt)
    return execute(decision, original_prompt=prompt)

