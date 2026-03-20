from __future__ import annotations

"""
CLI entrypoint for running a single ChemCrow-backed chemistry task with metrics.

This script exercises ChemCrowDriver on a fixed test task:
- Find the exact boiling point of 2-amino-5-nitrobenzophenone.
- Identify its GHS hazard statements using available tools.

It prints the raw JSON metrics dictionary returned by ChemCrowDriver.run_chemistry_sim,
which includes:
- output:          Text response from ChemCrow.
- step_start:      Epoch timestamp (float) at call start.
- step_end:        Epoch timestamp (float) at call end.
- latency_seconds: step_end - step_start (float, must be > 0).
- tokens_in:       Placeholder for prompt tokens (int, currently 0).
- tokens_out:      Placeholder for completion tokens (int, currently 0).

Tool-use verification:
- To satisfy the Tool-Use Check, run this module and inspect the terminal output
  for ChemCrow's own tool logs, such as:
    "Action: Query PubChem"
    "Action: Search GHS"

Usage (from repo root):
    python -m src.multi.chemistry.run_chem_task [--pretty]
"""

import argparse
import json
import sys
from typing import Any, Dict, Optional

from src.multi.chemistry.chem_driver import ChemCrowDriver


TEST_PROMPT = (
    "Find the exact boiling point of 2-amino-5-nitrobenzophenone and identify its "
    "GHS hazard statements using your available tools."
)


def run_test_task() -> Dict[str, Any]:
    """Run the fixed ChemCrow test task and return the metrics dictionary."""
    driver = ChemCrowDriver()
    return driver.run_chemistry_sim(TEST_PROMPT)


def _validate_metrics(result: Dict[str, Any]) -> Optional[str]:
    """
    Validate that the metrics dictionary has non-zero latency and timestamps.

    Returns:
        None if the metrics are valid; otherwise a string describing the failure.
    """
    if "step_start" not in result or "step_end" not in result:
        return "Missing step_start or step_end in result."

    latency = result.get("latency_seconds")
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
            "result": result,
        }
        print(json.dumps(payload, indent=2 if args.pretty else None))
        sys.exit(1)

    print(json.dumps(result, indent=2 if args.pretty else None))
    sys.exit(0)


if __name__ == "__main__":
    main()

