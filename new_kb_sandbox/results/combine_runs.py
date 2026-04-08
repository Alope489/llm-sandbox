"""Combine per-run CSVs from multiple benchmark invocations into combined output CSVs.

When benchmark runs are executed individually (``--runs 1`` per invocation) to
avoid long-running terminal sessions, each invocation writes its own timestamped
result directory containing ``metrics_per_run.csv`` and (when available)
``metrics_per_query.csv``.  This script collects those files from a folder of
timestamped subdirectories, re-numbers the ``run`` column sequentially (in
timestamp order), and produces three combined CSVs in that folder:

* ``metrics_per_run.csv``   — concatenation of all per-run rows, re-numbered.
* ``metrics_per_query.csv`` — concatenation of all per-query rows, re-numbered
  (written only when at least one run directory contains the file).
* ``metrics_averaged.csv``  — one row per step, averaged across all runs, using
  the same ``average_rows`` helper used by the benchmark itself.

Usage::

    python new_kb_sandbox/combine_runs.py <folder_containing_run_subdirs>

The folder should contain one or more timestamped subdirectories (e.g.
``2026-04-07T035232/``), each produced by a single benchmark invocation.  The
script works with both single-file and multi-file benchmark results — it is
regime-agnostic.

Dependencies:
    Standard library: argparse, csv, logging, sys, pathlib.
    Internal: new_kb_sandbox._shared (average_rows, write_csv).

Pillar compliance:
    - Pillar 1: Reuses average_rows/write_csv from _shared.py — identical
      averaging logic to the benchmark itself; no re-implementation.  Combined
      outputs are equivalent to a single multi-run invocation.
    - Pillar 2: Validates inputs; warns on missing/skipped directories; exits
      non-zero on fatal errors.
    - Pillar 3: Google-style docstrings on all public functions; file-level header.
    - Pillar 4: Single-responsibility; no hardcoding; all config via CLI.
    - Pillar 6: Θ(R × S × Q) for per-query loading; Θ(R × S) for averaging.
    - Pillar 7: Structured warnings; graceful SystemExit on fatal errors;
      graceful skip when per-query CSV is absent from a run directory.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

# _shared.py lives in new_kb_sandbox/, one level above this script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _shared import average_rows, write_csv  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
_logger = logging.getLogger(__name__)

_PER_RUN_FILENAME = "metrics_per_run.csv"
_PER_QUERY_FILENAME = "metrics_per_query.csv"
_OUTPUT_FILENAME = "metrics_averaged.csv"


def _coerce(value: str) -> int | float | str | None:
    """Coerce a CSV string value to its most specific native Python type.

    ``csv.DictReader`` returns all values as strings.  ``average_rows`` expects
    native numeric types for numeric columns.  This helper converts to ``int``
    first, then ``float``, then falls back to the original string.  Empty
    strings are returned as ``None`` (representing nullable CSV fields).

    Args:
        value: Raw string value read from a CSV cell.

    Returns:
        ``None`` for empty string; ``int`` if the value is a whole number;
        ``float`` if the value is a decimal; the original ``str`` otherwise.

    Examples:
        >>> _coerce("42")
        42
        >>> _coerce("3.14")
        3.14
        >>> _coerce("")
        >>> _coerce("gpt-4o-mini")
        'gpt-4o-mini'

    Complexity:
        O(1).
    """
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _discover_run_dirs(folder: Path) -> list[Path]:
    """Return sorted immediate subdirectories of *folder* that contain a per-run CSV.

    Subdirectories are sorted alphabetically, which is equivalent to
    chronological order for ISO-8601 timestamp directory names.  Subdirectories
    that do not contain ``metrics_per_run.csv`` are skipped with a warning.

    Args:
        folder: Directory to scan for run subdirectories.

    Returns:
        Sorted list of ``Path`` objects for qualifying subdirectories.

    Raises:
        SystemExit: If *folder* does not exist or is not a directory.

    Preconditions:
        - ``folder`` must exist and be a directory.

    Postconditions:
        - Every returned path is a subdirectory of ``folder`` containing
          ``metrics_per_run.csv``.

    Complexity:
        O(N log N) where N = number of immediate subdirectories.
    """
    if not folder.exists():
        raise SystemExit(f"ERROR: folder does not exist: {folder}")
    if not folder.is_dir():
        raise SystemExit(f"ERROR: path is not a directory: {folder}")

    candidates = sorted(p for p in folder.iterdir() if p.is_dir())
    run_dirs: list[Path] = []
    for candidate in candidates:
        csv_path = candidate / _PER_RUN_FILENAME
        if csv_path.is_file():
            run_dirs.append(candidate)
        else:
            _logger.warning(
                "Skipping subdirectory '%s' — no %s found.",
                candidate.name,
                _PER_RUN_FILENAME,
            )

    return run_dirs


def _load_rows(csv_path: Path) -> list[dict]:
    """Read all rows from a benchmark CSV file with type coercion.

    Suitable for both ``metrics_per_run.csv`` and ``metrics_per_query.csv``.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of row dicts with values coerced to native Python types via
        ``_coerce``.

    Raises:
        SystemExit: If the file cannot be read.

    Complexity:
        O(R × C) where R = row count, C = column count.
    """
    try:
        with csv_path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            return [{k: _coerce(v) for k, v in row.items()} for row in reader]
    except OSError as exc:
        raise SystemExit(f"ERROR: cannot read {csv_path}: {exc}") from exc


def combine(folder: Path) -> Path:
    """Discover run directories, combine per-run/per-query rows, and write output CSVs.

    Reads ``metrics_per_run.csv`` (and optionally ``metrics_per_query.csv``)
    from each qualifying subdirectory of *folder*, re-numbers the ``run``
    column sequentially (1, 2, 3, … in timestamp order), concatenates all
    rows, and writes three combined CSVs to *folder*:

    * ``metrics_per_run.csv``   — all per-run rows with re-numbered ``run``.
    * ``metrics_per_query.csv`` — all per-query rows with re-numbered ``run``
      (written only when at least one run directory contains the file; run
      directories that lack it are skipped with a warning).
    * ``metrics_averaged.csv``  — one averaged row per step, produced by
      ``average_rows`` from ``_shared.py``.

    The combined outputs are equivalent to having executed all runs in a single
    benchmark invocation: the ``run`` column reflects timestamp-sorted order,
    and all per-query/per-run measurements are preserved verbatim.

    Args:
        folder: Directory containing timestamped run subdirectories.

    Returns:
        Path to the written ``metrics_averaged.csv`` file.

    Raises:
        SystemExit: If *folder* is invalid, no qualifying run directories are
            found, or the output cannot be written.

    Preconditions:
        - ``folder`` must exist and be a directory.
        - At least one immediate subdirectory must contain ``metrics_per_run.csv``.

    Postconditions:
        - ``folder / metrics_per_run.csv`` exists with all rows re-numbered.
        - ``folder / metrics_averaged.csv`` exists with one averaged row per step.
        - ``folder / metrics_per_query.csv`` exists iff at least one run directory
          contained a non-empty ``metrics_per_query.csv``.

    Complexity:
        Θ(R × S × Q) for per-query loading; Θ(R × S) for per-run loading and
        averaging; where R = run count, S = steps, Q = queries per step.
    """
    run_dirs = _discover_run_dirs(folder)

    if not run_dirs:
        raise SystemExit(
            f"ERROR: no subdirectories containing {_PER_RUN_FILENAME} found in {folder}"
        )
    if len(run_dirs) < 2:
        _logger.warning(
            "Only 1 run directory found — metrics_averaged.csv will have std dev = 0."
        )

    print(f"Found {len(run_dirs)} run director{'y' if len(run_dirs) == 1 else 'ies'}.")

    all_rows: list[dict] = []
    all_query_rows: list[dict] = []
    step_counts: list[int] = []

    for run_number, run_dir in enumerate(run_dirs, start=1):
        rows = _load_rows(run_dir / _PER_RUN_FILENAME)
        if not rows:
            _logger.warning("Skipping '%s' — CSV is empty.", run_dir.name)
            continue
        for row in rows:
            row["run"] = run_number
        all_rows.extend(rows)
        step_counts.append(len(rows))

        pq_path = run_dir / _PER_QUERY_FILENAME
        if pq_path.is_file():
            pq_rows = _load_rows(pq_path)
            for row in pq_rows:
                row["run"] = run_number
            all_query_rows.extend(pq_rows)
        else:
            _logger.warning(
                "Skipping per-query rows for '%s' — no %s found.",
                run_dir.name,
                _PER_QUERY_FILENAME,
            )

    if not all_rows:
        raise SystemExit("ERROR: all CSVs were empty — nothing to combine.")

    unique_step_counts = set(step_counts)
    if len(unique_step_counts) > 1:
        _logger.warning(
            "Runs have different step counts %s — combining anyway.",
            sorted(unique_step_counts),
        )

    steps_per_run = step_counts[0]
    print(
        f"Loaded {len(all_rows)} rows "
        f"({steps_per_run} steps × {len(step_counts)} runs)."
    )

    per_run_out = folder / _PER_RUN_FILENAME
    if per_run_out.exists():
        _logger.warning("Overwriting existing %s.", per_run_out)
    write_csv(per_run_out, all_rows)
    print(f"Written: {per_run_out}")

    if all_query_rows:
        per_query_out = folder / _PER_QUERY_FILENAME
        if per_query_out.exists():
            _logger.warning("Overwriting existing %s.", per_query_out)
        write_csv(per_query_out, all_query_rows)
        print(f"Written: {per_query_out}")
    else:
        _logger.warning(
            "No per-query rows found across any run directory — skipping %s.",
            _PER_QUERY_FILENAME,
        )

    output_path = folder / _OUTPUT_FILENAME
    if output_path.exists():
        _logger.warning("Overwriting existing %s.", output_path)
    averaged = average_rows(all_rows)
    write_csv(output_path, averaged)
    print(f"Written: {output_path}")
    return output_path


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Populated ``argparse.Namespace`` with a ``folder`` attribute.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Combine metrics_per_run.csv files from multiple benchmark run "
            "directories into a single metrics_averaged.csv."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "folder",
        type=Path,
        help=(
            "Directory containing timestamped run subdirectories, each holding "
            "a metrics_per_run.csv produced by a --runs 1 benchmark invocation."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the combine_runs script.

    Returns:
        None

    Raises:
        SystemExit: On invalid input or write failure.
    """
    args = _parse_args()
    combine(args.folder)


if __name__ == "__main__":
    main()
