from __future__ import annotations

"""
Multi-agent reasoning utilities and metrics export for the multi pipeline.

This module defines a simple step-record schema that is compatible with the
ChemCrow-backed Simulation LLM role and future agents. Each step record is a
dictionary with the following keys:

- agent:           Logical agent name (e.g. "ChemCrow", "KB-Agent").
- output:          Text response from the agent.
- step_start:      Epoch timestamp (float) at call start.
- step_end:        Epoch timestamp (float) at call end.
- latency_seconds: step_end - step_start (float).
- tokens_in:       Prompt tokens (int, may be 0 for placeholders).
- tokens_out:      Completion tokens (int, may be 0 for placeholders).

These fields align with ChemCrowDriver.run_chemistry_sim and run_simulation_llm
so that fluid-model / traffic-intensity calculations can treat each step as a
generic service-time sample S with latency_seconds as the service duration.
"""

from typing import Any, Dict, List

from src.multi.agent import run_simulation_llm
from src.multi.materials.atom_driver import AtomAgentDriver


def run_chemistry_step(prompt: str) -> Dict[str, Any]:
    """
    Run a single chemistry-specific Simulation LLM step backed by ChemCrow.

    The underlying call uses run_simulation_llm(prompt, is_chemistry=True) and
    returns a step record suitable for aggregation by the Reasoning Agent.
    """
    metrics = run_simulation_llm(prompt, is_chemistry=True)
    return {
        "agent": "ChemCrow",
        **metrics,
    }


def run_materials_step(prompt: str) -> Dict[str, Any]:
    """
    Run a single materials-specific step backed by AtomAgentDriver.

    Instantiates AtomAgentDriver, executes run_materials_sim(prompt), and
    returns a step record compatible with export_metrics aggregation.
    """
    driver = AtomAgentDriver()
    metrics = driver.run_materials_sim(prompt)
    return {
        "agent": "AtomAgent",
        **metrics,
    }


def export_metrics(steps: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Package a list of step records into a JSON-serializable bundle.

    This structure is intended as the canonical input for downstream
    fluid-model / traffic-intensity analysis.
    """
    return {"steps": list(steps)}


