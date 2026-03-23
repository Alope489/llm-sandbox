"""Thin wrapper around ChemCrowDriver for chemistry-specific tasks.

This module provides a simple, function-style interface over ChemCrowDriver so
that other agents can call into the ChemCrow-backed simulation layer without
managing driver instances directly.

The public entrypoint, run_chemistry_llm, returns a metrics-bearing dictionary
with keys aligned to other agent outputs while routing between:
- RDKit SMILES validation
- PubChemPy common-name lookup
- Optional Sonnet 4.6 reasoning for complex chemistry prompts only

Keys aligned to other agent outputs:
- output
- step_start
- step_end
- latency_seconds
- tokens_in
- tokens_out
"""

from __future__ import annotations

import re
from typing import Any, Dict

from src.multi.chemistry.chem_driver import ChemCrowDriver, ChemistryResult, ChemistryRunResult

_CHEM_DRIVER: ChemCrowDriver | None = None


def _get_driver() -> ChemCrowDriver:
    """Return a singleton ChemCrowDriver instance for this process."""
    global _CHEM_DRIVER
    if _CHEM_DRIVER is None:
        _CHEM_DRIVER = ChemCrowDriver()
    return _CHEM_DRIVER


def run_chemistry_llm(prompt: str) -> ChemistryRunResult:
    """Run a chemistry-related task through ChemCrowDriver.

    Parameters
    ----------
    prompt:
        Natural-language description of the chemistry task to solve.

    Returns
    -------
    ChemistryRunResult
        Typed chemistry run result containing a structured ChemistryResult payload.
    """
    driver = _get_driver()
    return driver.run_chemistry_sim(prompt)


def _looks_like_smiles(candidate: str) -> bool:
    """Simple router check to classify likely SMILES input."""
    text = candidate.strip()
    if not text or " " in text:
        return False
    if re.fullmatch(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+", text) is None:
        return False
    if re.search(r"[\[\]\(\)=#@+\-\\/%.0-9]", text):
        return True
    uppercase_count = sum(1 for ch in text if ch.isupper())
    return uppercase_count >= 2 and len(text) <= 12


def run_chemistry_task(query: str) -> ChemistryResult:
    """Auto-route chemistry query to SMILES validation or name lookup.

    Routing strategy:
    - If query looks like SMILES, validate with RDKit.
    - Otherwise treat query as a common chemical name and query PubChemPy.
    - Only call Sonnet 4.6 reasoning for complex prompts (inside driver).
    """
    driver = _get_driver()
    run_result = driver.run_chemistry_sim(query)
    if _looks_like_smiles(query):
        run_result.output.route = "smiles"
    else:
        run_result.output.route = "name_lookup"
    # Explicitly return the typed chemistry model for predictable downstream use.
    return run_result.output

