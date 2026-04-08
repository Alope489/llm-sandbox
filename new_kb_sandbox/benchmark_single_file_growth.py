"""Benchmark Test 1 — Single-file growth regime.

At each logical step N, a single file containing the concatenated bytes of the
first ``N × chunks_per_step`` raw source chunks is uploaded to a fresh OpenAI
vector store.  All 6 benchmark queries are then issued against that store.
The variable under test is KB size; the file structure is always one file.

``--max-files`` controls the number of logical growth steps; total raw chunks
loaded = ``max_files × chunks_per_step``.  The default ``chunks_per_step=2``
produces ~10 KB steps (2 raw ~5 KB chunks per step).

Compare against ``benchmark_multi_file_growth.py`` (Test 2) where the same
total KB bytes are spread across N separate logical files per step.

Usage::

    python new_kb_sandbox/benchmark_single_file_growth.py [OPTIONS]

See ``--help`` for full option list.

Dependencies:
    openai, tqdm, python-dotenv, new_kb_sandbox._shared, src.llm_pipeline_telemetry.

Logging behaviour (changed from default):
    The ``llm.telemetry`` logger is forced to ``WARNING`` level inside
    ``main()`` before any API calls are made.  This suppresses the per-call
    ``INFO``/``DEBUG`` JSON records that ``log_llm_call`` normally emits.
    Rationale: a full benchmark (e.g. 5 runs × 50 steps × 6 queries = 1 500
    calls) would otherwise produce 1 500+ structured log lines that obscure the
    ``tqdm`` progress bars and add no actionable signal during a run.
    All telemetry data is still captured in-process via ``CallContext.records``
    and is available to callers of ``run_benchmark`` after the run completes;
    only the side-channel log emission is suppressed.
    If you need ``DEBUG``/``INFO`` telemetry during development, pass
    ``--log-level DEBUG`` or raise the level explicitly before invoking
    ``main()``.

Pillar compliance:
    - Pillar 1: Fresh VS per step; single-file regime isolates KB-size effect.
    - Pillar 4: All config via CLI args / env vars; no hardcoding.
    - Pillar 5: OPENAI_API_KEY from .env only.
    - Pillar 6: Θ(R×S×Q) calls, Θ(R×S²) upload bytes at defaults.
    - Pillar 7: Mandatory pre-flight cost confirmation; retry on upload/query.
"""
import argparse
import logging
import os
import sys
from collections.abc import Callable
from pathlib import Path

from dotenv import load_dotenv

# Allow running as a script directly from the project root or from within
# new_kb_sandbox/.  Adds the new_kb_sandbox/ directory to sys.path so that
# ``_shared`` is importable without a package install.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _shared import (  # noqa: E402
    build_growing_file,
    collect_chunk_paths,
    confirm_run,
    estimate_cost,
    load_queries,
    run_benchmark,
)

load_dotenv()

_DEFAULT_KB_DIR = (
    Path(__file__).resolve().parent
    / "my_knowledge_base_data"
    / "materials_science_kb"
)
_DEFAULT_QUERIES_FILE = Path(__file__).resolve().parent / "prompts" / "kb_benchmark_queries.txt"
_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "single_file"
_VS_NAME_PREFIX = "kb-bench-single"


def make_get_files_for_step(
    chunks: list[Path],
    chunks_per_step: int = 2,
) -> Callable[[int, Path], list[Path]]:
    """Return the file-step callable for the single-file growth regime.

    At each logical step N the returned callable concatenates the first
    ``N * chunks_per_step`` raw source chunks into one growing file inside
    ``tmp_dir`` and returns it as a single-element list.  The temp file is
    created by ``build_growing_file`` and lives in ``tmp_dir`` so that
    ``run_benchmark`` can identify and clean it up in the ``finally`` block.

    Args:
        chunks: Ordered list of source chunk paths (e.g. from
            ``collect_chunk_paths``).  Must have at least
            ``max_files * chunks_per_step`` entries.
        chunks_per_step: Number of raw ~5 KB source chunks consumed per
            logical step.  Default ``2`` produces ~10 KB logical files.

    Returns:
        Callable ``(step: int, tmp_dir: Path) -> list[Path]`` whose single
        returned path always has ``path.parent == tmp_dir``, making it safe
        for temp-file cleanup inside ``run_benchmark``.

    Examples:
        >>> get_files = make_get_files_for_step(chunks, chunks_per_step=2)
        >>> file_paths = get_files(3, tmp_dir)
        >>> len(file_paths)
        1

    Complexity:
        O(step × chunks_per_step) per call — dominated by binary concatenation.
    """
    def _get_files(step: int, tmp_dir: Path) -> list[Path]:
        return [build_growing_file(chunks, step * chunks_per_step, tmp_dir)]

    return _get_files


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Populated ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(
        description="KB growth benchmark — single-file growth regime (Test 1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=50,
        help="Number of logical growth steps. Total raw chunks loaded = max_files × chunks_per_step.",
    )
    parser.add_argument(
        "--chunks-per-step",
        type=int,
        default=2,
        help="Raw ~5 KB chunks consumed per logical step (default 2 → ~10 KB steps).",
    )
    parser.add_argument("--runs", type=int, default=5, help="Number of independent repeat runs.")
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=_DEFAULT_QUERIES_FILE,
        help="Path to the newline-and-'---'-delimited benchmark queries file.",
    )
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=_DEFAULT_KB_DIR,
        help="Directory containing kb_chunk_*.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help=(
            "Parent results directory.  A YYYY-MM-DDTHHMMSS subdirectory is "
            "created at run start; old subdirectories are never modified."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model identifier.",
    )
    parser.add_argument(
        "--keep-vector-stores",
        action="store_true",
        help="Skip VS deletion; use cleanup_vector_stores.py to recover.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Per-query and per-upload retry count.",
    )
    parser.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=2.0,
        help="Base sleep between retries in seconds (linear back-off).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip pre-flight cost confirmation prompt.",
    )
    args = parser.parse_args()
    if args.chunks_per_step < 1:
        parser.error("--chunks-per-step must be >= 1")
    return args


def main() -> None:
    """Entry point for Test 1 benchmark runner.

    Validates environment, confirms cost estimate, then delegates to
    ``run_benchmark`` in ``_shared.py`` with the single-file factory.

    Logging override: immediately forces ``logging.getLogger("llm.telemetry")``
    to ``WARNING`` before any API calls.  This deliberately silences the
    ``INFO``-level per-call telemetry records that ``log_llm_call`` emits on
    each successful query, preventing them from interleaving with ``tqdm``
    progress output during long benchmark runs.  In-process telemetry data
    (``CallContext.records``) is unaffected.

    Returns:
        None

    Raises:
        SystemExit: If OPENAI_API_KEY is absent or the user declines the
            cost confirmation.
    """
    args = _parse_args()
    logging.getLogger("llm.telemetry").setLevel(logging.WARNING)

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required.  Set it in .env or the environment.")

    from src.llm_pipeline_telemetry import get_openai_client  # noqa: PLC0415

    client = get_openai_client()
    queries = load_queries(args.queries_file)
    chunks = collect_chunk_paths(args.kb_dir, args.max_files * args.chunks_per_step)

    if not args.yes:
        avg_kb_bytes = (
            sum(c.stat().st_size for c in chunks[:args.chunks_per_step])
            if chunks else 5120 * args.chunks_per_step
        )
        estimates = estimate_cost(
            runs=args.runs,
            max_files=args.max_files,
            query_count=len(queries),
            avg_kb_bytes_per_step=avg_kb_bytes,
        )
        confirm_run(estimates)

    run_dir = run_benchmark(
        make_get_files_for_step(chunks, args.chunks_per_step),
        runs=args.runs,
        max_files=args.max_files,
        queries=queries,
        chunks=chunks,
        client=client,
        model=args.model,
        output_dir=args.output_dir,
        keep_vector_stores=args.keep_vector_stores,
        max_retries=args.max_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
        vs_name_prefix=_VS_NAME_PREFIX,
        chunks_per_step=args.chunks_per_step,
    )
    print(f"Results written to {run_dir}")


if __name__ == "__main__":
    main()
