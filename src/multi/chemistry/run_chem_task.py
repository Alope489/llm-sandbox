from __future__ import annotations

"""
CLI entrypoint for running a single chemistry task with real tooling.

This script exercises ChemCrowDriver on a fixed test task.

It prints the raw JSON metrics dictionary returned by ChemCrowDriver.run_chemistry_sim.
Keys include:
- output:          Text response from tool routing and optional reasoning.
- step_start:      Epoch timestamp (float) at call start.
- step_end:        Epoch timestamp (float) at call end.
- latency_seconds: step_end - step_start (float, must be > 0).
- tokens_in:       Placeholder for prompt tokens (int, currently 0).
- tokens_out:      Placeholder for completion tokens (int, currently 0).

Tooling behavior:
- Uses RDKit for SMILES validation.
- Uses PubChemPy for common-name property lookup.
- Uses Sonnet reasoning only when the prompt is complex enough.

Usage (from repo root):
    python -m src.multi.chemistry.run_chem_task [--pretty]
"""

import argparse
from dataclasses import asdict
import json
import sys
from typing import Optional

from dotenv import load_dotenv
from src.multi.chemistry.chem_driver import ChemCrowDriver, ChemistryRunResult

load_dotenv()


TEST_PROMPT = (
    "acetone"
)


def run_test_task() -> ChemistryRunResult:
    """Run the fixed ChemCrow test task and return the typed metrics result."""
    driver = ChemCrowDriver()
    return driver.run_chemistry_sim(TEST_PROMPT)


def _validate_metrics(result: ChemistryRunResult) -> Optional[str]:
    """
    Validate that the typed metrics result has non-zero latency and timestamps.

    Returns:
        None if the metrics are valid; otherwise a string describing the failure.
    """
    if not hasattr(result, "step_start") or not hasattr(result, "step_end"):
        return "Missing step_start or step_end in result."

    latency = result.latency_seconds
    if not isinstance(latency, (int, float)):
        return "latency_seconds is missing or not a number."
    if latency <= 0:
        return "latency_seconds is not greater than zero."

    return None


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a single ChemCrow-backed chemistry task via ChemCrowDriver and "
            "print the raw JSON metrics dictionary."
        )
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output with indentation.",
    )
    args = parser.parse_args(argv)

    result = run_test_task()
    error = _validate_metrics(result)

    if error is not None:
        payload = {
            "status": "failure",
            "reason": error,
            "result": asdict(result),
        }
        print(json.dumps(payload, indent=2 if args.pretty else None))
        sys.exit(1)

    print(json.dumps(asdict(result), indent=2 if args.pretty else None))
    sys.exit(0)


if __name__ == "__main__":
    main()

