"""Multi-agent orchestration entrypoints for the simulation LLM role.

This module introduces a Simulation LLM facade that can either:
- Use the ChemCrow-backed chemistry driver for explicitly flagged chemistry tasks.
- Fall back to the base LLM for non-chemistry simulation reasoning.

Chemistry routing is controlled via Option B (explicit flag): call sites must
set is_chemistry=True when the Simulation role is being used for chemistry.
"""

from __future__ import annotations

import time
from typing import Any, Dict

from src.multi.chemistry.chem_driver import ChemCrowDriver
from src.wrapper import complete

_CHEM_DRIVER: ChemCrowDriver | None = None


def _get_chem_driver() -> ChemCrowDriver:
    """Return a singleton ChemCrowDriver instance for this process."""
    global _CHEM_DRIVER
    if _CHEM_DRIVER is None:
        _CHEM_DRIVER = ChemCrowDriver()
    return _CHEM_DRIVER


def run_simulation_llm(
    prompt: str,
    *,
    is_chemistry: bool = False,
) -> Dict[str, Any]:
    """Run the Simulation LLM role with optional ChemCrow routing.

    Parameters
    ----------
    prompt:
        Natural-language description of the simulation or analysis task.
    is_chemistry:
        When True, bypasses any existing numeric simulation logic and instead
        routes the call to ChemCrowDriver, treating it as a chemistry-specific
        reasoning task. When False, the prompt is sent directly to the base LLM
        via src.wrapper.complete.

    Returns
    -------
    dict
        A dictionary with the following keys, matching the structure used by
        other agents so that the Reasoning Agent can aggregate metrics without
        special casing:
        - output:           Text response from ChemCrow or the base LLM.
        - step_start:       Epoch timestamp (float) at call start.
        - step_end:         Epoch timestamp (float) at call end.
        - latency_seconds:  step_end - step_start.
        - tokens_in:        Placeholder for prompt tokens (int, currently 0).
        - tokens_out:       Placeholder for completion tokens (int, currently 0).
    """
    if is_chemistry:
        # Explicit chemistry routing: use ChemCrowDriver as the bolted-down
        # simulation layer for chemistry-related tasks.
        driver = _get_chem_driver()
        return driver.run_chemistry_sim(prompt)

    # Default path: delegate to the base LLM and wrap with timing metadata so
    # downstream components always receive a consistent dictionary shape.
    step_start = time.time()
    output = complete([{"role": "user", "content": prompt}])
    step_end = time.time()

    return {
        "output": output,
        "step_start": step_start,
        "step_end": step_end,
        "latency_seconds": step_end - step_start,
        "tokens_in": 0,
        "tokens_out": 0,
    }

