"""Benchmark Test 2 — Multi-file growth regime.

At each step N, N separate 5 KB chunk files are uploaded to a fresh OpenAI
vector store.  All 6 benchmark queries are then issued against that store.
The only variable between Test 1 and Test 2 is file structure: identical
total KB bytes at each step, different file granularity.

Compare against ``benchmark_single_file_growth.py`` (Test 1) where the same
total bytes are merged into a single growing file.

Usage::

    python new_kb_sandbox/benchmark_multi_file_growth.py [OPTIONS]

See ``--help`` for full option list.

Dependencies:
    openai, tqdm, python-dotenv, new_kb_sandbox._shared, src.llm_pipeline_telemetry.

Pillar compliance:
    - Pillar 1: Fresh VS per step; multi-file regime isolates file-count effect.
    - Pillar 4: All config via CLI args / env vars; no hardcoding.
    - Pillar 5: OPENAI_API_KEY from .env only.
    - Pillar 6: Θ(R×S×Q) calls, Θ(R×S²) upload bytes at defaults.
    - Pillar 7: Mandatory pre-flight cost confirmation; retry on upload/query.
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Allow running as a script directly from the project root or from within
# new_kb_sandbox/.  Adds the new_kb_sandbox/ directory to sys.path so that
# ``_shared`` is importable without a package install.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _shared import (  # noqa: E402
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
_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "multi_file"
_VS_NAME_PREFIX = "kb-bench-multi"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Populated ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(
        description="KB growth benchmark — multi-file growth regime (Test 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--max-files", type=int, default=50, help="Number of 5 KB steps.")
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
    return parser.parse_args()


def main() -> None:
    """Entry point for Test 2 benchmark runner.

    Validates environment, confirms cost estimate, then delegates to
    ``run_benchmark`` in ``_shared.py`` with a multi-file lambda.

    Returns:
        None

    Raises:
        SystemExit: If OPENAI_API_KEY is absent or the user declines the
            cost confirmation.
    """
    args = _parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required.  Set it in .env or the environment.")

    from src.llm_pipeline_telemetry import get_openai_client  # noqa: PLC0415

    client = get_openai_client()
    queries = load_queries(args.queries_file)
    chunks = collect_chunk_paths(args.kb_dir, args.max_files)

    if not args.yes:
        avg_kb_bytes = chunks[0].stat().st_size if chunks else 5120
        estimates = estimate_cost(
            runs=args.runs,
            max_files=args.max_files,
            query_count=len(queries),
            avg_kb_bytes_per_step=avg_kb_bytes,
        )
        confirm_run(estimates)

    run_dir = run_benchmark(
        lambda step, _: chunks[:step],
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
    )
    print(f"Results written to {run_dir}")


if __name__ == "__main__":
    main()
