"""Thin wrapper around ChemCrowDriver for chemistry-specific LLM calls.

This module provides a simple, function-style interface over ChemCrowDriver so
that other agents can call into the ChemCrow-backed simulation layer without
managing driver instances directly.

The public entrypoint, run_chemistry_llm, returns a metrics-bearing dictionary
with keys aligned to other agent outputs:
- output
- step_start
- step_end
- latency_seconds
- tokens_in
- tokens_out
"""

from __future__ import annotations

from typing import Any, Dict

from src.multi.chemistry.chem_driver import ChemCrowDriver

_CHEM_DRIVER: ChemCrowDriver | None = None


def _get_driver() -> ChemCrowDriver:
    """Return a singleton ChemCrowDriver instance for this process."""
    global _CHEM_DRIVER
    if _CHEM_DRIVER is None:
        _CHEM_DRIVER = ChemCrowDriver()
    return _CHEM_DRIVER


def run_chemistry_llm(prompt: str) -> Dict[str, Any]:
    """Run a chemistry-related task through ChemCrowDriver.

    Parameters
    ----------
    prompt:
        Natural-language description of the chemistry task to solve.

    Returns
    -------
    dict
        The structured result from ChemCrowDriver.run_chemistry_sim, including:
        output, step_start, step_end, latency_seconds, tokens_in, tokens_out.
    """
    driver = _get_driver()
    return driver.run_chemistry_sim(prompt)

