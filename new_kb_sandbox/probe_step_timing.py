"""Standalone timing probe for a single step of both KB-growth benchmark regimes.

Runs exactly one step of both the single-file (Test 1) and multi-file (Test 2)
benchmark regimes against the real OpenAI API, prints a per-phase timing
breakdown for each regime, then projects total wall time for a full benchmark
run based on the measured timings.

Usage::

    python new_kb_sandbox/probe_step_timing.py --step 50
    python new_kb_sandbox/probe_step_timing.py --step 25 --max-files 50 --runs 5
    python new_kb_sandbox/probe_step_timing.py --step 10 --model gpt-4o-mini --yes

The script is read-only with respect to the knowledge base: it creates and
deletes transient OpenAI vector stores but never modifies local files.

Dependencies:
    openai, python-dotenv, new_kb_sandbox._shared, src.llm_pipeline_telemetry.

Pillar compliance:
    - Pillar 1: Reuses build_growing_file, upload_with_retry, query_with_retry
      from _shared.py — same code paths as the real benchmark runners.
    - Pillar 3: Google-style docstrings on all functions; file-level header.
    - Pillar 4: All configuration via CLI args / env vars; no hardcoding.
    - Pillar 5: OPENAI_API_KEY from .env only; never printed.
    - Pillar 6: Projection formula documents its Θ assumptions explicitly.
    - Pillar 7: try/finally ensures VS deletion and temp-file cleanup even on
      query failure; structured log messages via llm.telemetry logger.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Allow running as a script from the project root or from within new_kb_sandbox/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _shared import (  # noqa: E402
    build_combined_chunk_file,
    build_growing_file,
    collect_chunk_paths,
    load_queries,
    query_with_retry,
    upload_with_retry,
)

load_dotenv()

_logger = logging.getLogger("llm.telemetry")

_DEFAULT_KB_DIR = (
    Path(__file__).resolve().parent
    / "my_knowledge_base_data"
    / "materials_science_kb"
)
_DEFAULT_QUERIES_FILE = (
    Path(__file__).resolve().parent / "prompts" / "kb_benchmark_queries.txt"
)


# ---------------------------------------------------------------------------
# Core probe logic
# ---------------------------------------------------------------------------


def _run_regime_step(
    regime: str,
    chunks: list[Path],
    step: int,
    client: object,
    model: str,
    queries: list[str],
    max_retries: int,
    retry_sleep_seconds: float,
    chunks_per_step: int = 2,
) -> dict:
    """Run one step of either the single-file or multi-file benchmark regime.

    Creates a fresh vector store, uploads the appropriate files, issues every
    query in *queries*, then deletes the vector store.  Timing is recorded for
    each sub-phase via ``time.perf_counter()``.  The vector store is always
    deleted in a ``finally`` block, even if a query fails.

    For the single-file regime a temporary concatenated ``.txt`` file is
    created in a temp directory and deleted in a ``finally`` block regardless
    of outcome.

    Args:
        regime: Either ``"single"`` (one growing concatenated file) or
            ``"multi"`` (N separate logical files).
        chunks: Full ordered list of source chunk paths; only the first
            ``step * chunks_per_step`` entries are used.
        step: 1-indexed logical step number.
        client: Authenticated ``openai.OpenAI`` instance.
        model: OpenAI model identifier (e.g. ``"gpt-4o-mini"``).
        queries: List of query strings to issue against the vector store.
        max_retries: Per-operation retry count passed to ``upload_with_retry``
            and ``query_with_retry``.
        retry_sleep_seconds: Base sleep between retries (linear back-off).
        chunks_per_step: Number of raw ~5 KB chunks per logical step/file.
            Default ``2`` → ~10 KB per logical unit.

    Returns:
        A dict with keys:
        ``regime`` (str), ``step`` (int), ``kb_size_bytes`` (int),
        ``file_count`` (int), ``vs_create_ms`` (float),
        ``preload_ms`` (float), ``preload_status`` (str),
        ``query_results`` (list[dict] — one per query, each containing
        ``elapsed_ms``, ``provider_server_latency_ms``, ``input_tokens``,
        ``output_tokens``, ``has_citation`` (bool)),
        ``vs_delete_ms`` (float), ``step_total_ms`` (float).

    Raises:
        openai.APIConnectionError: If the vector store cannot be reached after
            retries.
        openai.BadRequestError: If file upload is rejected (e.g. bad extension).
        Any exception raised by ``upload_with_retry`` or ``query_with_retry``
        after all retries are exhausted.

    Preconditions:
        - ``1 <= step <= len(chunks) // chunks_per_step``.
        - ``chunks_per_step >= 1``.
        - ``client`` is authenticated and ``OPENAI_API_KEY`` is set.

    Postconditions:
        - The created vector store is deleted (or attempted) before returning.
        - Any temp file created for the single-file regime is deleted.

    Complexity:
        Θ(B) for upload where B = cumulative byte size of
        ``chunks[:step * chunks_per_step]``.
        Θ(Q) API calls where Q = len(queries).
    """
    from src.llm_pipeline_telemetry import CallContext  # noqa: PLC0415

    ms_ts = int(time.time() * 1000)
    hex4 = uuid.uuid4().hex[:4]
    vs_name = f"kb-bench-probe-{regime}-step{step:04d}-{ms_ts}-{hex4}"

    kb_size_bytes = sum(c.stat().st_size for c in chunks[:step * chunks_per_step])
    t_step_start = time.perf_counter()

    ctx = CallContext(pipeline="kb_growth_probe")
    vs_id: Optional[str] = None
    tmp_files: list[Path] = []
    tmp_dir_path: Optional[Path] = None

    try:
        # -- VS create --------------------------------------------------------
        t0 = time.perf_counter()
        vs = client.vector_stores.create(name=vs_name)  # type: ignore[attr-defined]
        vs_create_ms = (time.perf_counter() - t0) * 1000.0
        vs_id = vs.id

        # -- File preparation -------------------------------------------------
        if regime == "single":
            tmp_dir_path = Path(tempfile.mkdtemp(prefix="kb_probe_"))
            f = build_growing_file(chunks, step * chunks_per_step, tmp_dir_path)
            file_paths = [f]
            tmp_files = [f]
        elif chunks_per_step == 1:
            file_paths = chunks[:step]
        else:
            tmp_dir_path = Path(tempfile.mkdtemp(prefix="kb_probe_"))
            file_paths = [
                build_combined_chunk_file(chunks, i, chunks_per_step, step, tmp_dir_path)
                for i in range(step)
            ]
            tmp_files = list(file_paths)

        file_count = len(file_paths)

        # -- Upload + poll (preload) ------------------------------------------
        t0 = time.perf_counter()
        preload_status = upload_with_retry(
            client,  # type: ignore[arg-type]
            vector_store_id=vs_id,
            file_paths=file_paths,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        preload_ms = (time.perf_counter() - t0) * 1000.0

        # -- Queries ----------------------------------------------------------
        query_results: list[dict] = []
        for q_idx, query in enumerate(queries, start=1):
            result = query_with_retry(
                client,  # type: ignore[arg-type]
                model=model,
                vector_store_id=vs_id,
                query=query,
                ctx=ctx,
                max_retries=max_retries,
                retry_sleep_seconds=retry_sleep_seconds,
                progress_bar=None,
            )
            query_results.append(
                {
                    "q_idx": q_idx,
                    "elapsed_ms": result["elapsed_ms"],
                    "provider_server_latency_ms": result.get(
                        "provider_server_latency_ms"
                    ),
                    "input_tokens": result["input_tokens"],
                    "output_tokens": result["output_tokens"],
                    "has_citation": result.get("has_citation", False),
                }
            )

        # -- VS delete --------------------------------------------------------
        t0 = time.perf_counter()
        client.vector_stores.delete(vs_id)  # type: ignore[attr-defined]
        vs_id = None
        vs_delete_ms = (time.perf_counter() - t0) * 1000.0

    except Exception:
        _logger.warning(
            "probe_step_timing: exception during regime=%s step=%d", regime, step
        )
        raise
    finally:
        if vs_id is not None:
            try:
                client.vector_stores.delete(vs_id)  # type: ignore[attr-defined]
            except Exception:
                _logger.warning(
                    "probe_step_timing: failed to delete VS %s during cleanup", vs_id
                )
        for f in tmp_files:
            f.unlink(missing_ok=True)
        if tmp_dir_path is not None:
            try:
                tmp_dir_path.rmdir()
            except OSError:
                pass

    step_total_ms = (time.perf_counter() - t_step_start) * 1000.0

    return {
        "regime": regime,
        "step": step,
        "kb_size_bytes": kb_size_bytes,
        "file_count": file_count,
        "vs_create_ms": round(vs_create_ms, 1),
        "preload_ms": round(preload_ms, 1),
        "preload_status": preload_status,
        "query_results": query_results,
        "vs_delete_ms": round(vs_delete_ms, 1),
        "step_total_ms": round(step_total_ms, 1),
    }


def _project_total(
    single: dict,
    multi: dict,
    max_files: int,
    runs: int,
) -> dict:
    """Project full benchmark wall time from a single probed step.

    Uses the measured timings to estimate total time for a ``max_files``-step,
    ``runs``-repetition benchmark run under three modelling assumptions that
    are stated explicitly in the returned dict:

    1. **VS overhead** — VS create and delete times are constant across steps
       (independent of KB size).  Projected as
       ``runs × max_files × (vs_create_ms + vs_delete_ms)``.

    2. **Preload** — Upload + indexing time scales linearly with cumulative KB
       size.  At the probed step S, the rate is
       ``preload_ms / kb_size_bytes`` ms/byte.
       The cumulative KB across all steps grows as
       ``Σ(i=1..max_files) i × chunk_bytes``, which equals
       ``max_files × (max_files+1)/2 × chunk_bytes``.
       Total preload ≈ ``runs × rate × max_files×(max_files+1)/2 × chunk_bytes``.

    3. **Query latency** — The probed step is taken as an upper bound (ceiling),
       since earlier steps have smaller KBs and shorter latencies.  An
       optimistic estimate uses the probed mean directly; a pessimistic
       estimate doubles it to account for server-load variance.

    Args:
        single: Result dict returned by ``_run_regime_step("single", ...)``.
        multi: Result dict returned by ``_run_regime_step("multi", ...)``.
        max_files: Total number of steps in the real benchmark.
        runs: Number of independent repeat runs in the real benchmark.

    Returns:
        A dict with the following projection keys for each regime
        (``single_`` and ``multi_`` prefixed):
        ``vs_overhead_s``, ``preload_total_s``,
        ``query_optimistic_s``, ``query_pessimistic_s``,
        ``wall_optimistic_s``, ``wall_pessimistic_s``.
        Also includes ``assumptions`` (list[str]) for display.

    Complexity:
        O(1) — purely arithmetic, no API calls.
    """
    query_count = len(single["query_results"])
    projections: dict = {"assumptions": [
        "VS create/delete latency is constant across all steps.",
        f"Preload scales linearly with cumulative KB size (rate measured at step {single['step']}).",
        f"Query latency at the probed step is used as the ceiling (pessimistic = 2×).",
        f"Projection is for {runs} run(s) × {max_files} step(s) × {query_count} queries/step.",
    ]}

    for result in (single, multi):
        regime = result["regime"]
        chunk_bytes = result["kb_size_bytes"] / result["step"]

        # -- VS overhead: constant per step -----------------------------------
        vs_per_step_ms = result["vs_create_ms"] + result["vs_delete_ms"]
        vs_overhead_s = runs * max_files * vs_per_step_ms / 1000.0

        # -- Preload: linear rate × triangular sum of KB sizes ----------------
        rate_ms_per_byte = result["preload_ms"] / max(result["kb_size_bytes"], 1)
        triangular_bytes = chunk_bytes * max_files * (max_files + 1) / 2
        preload_total_s = runs * rate_ms_per_byte * triangular_bytes / 1000.0

        # -- Query latency: ceiling from probed step --------------------------
        query_elapsed = [r["elapsed_ms"] for r in result["query_results"]]
        mean_query_ms = sum(query_elapsed) / len(query_elapsed) if query_elapsed else 0.0
        total_queries = runs * max_files * query_count
        query_optimistic_s = total_queries * mean_query_ms / 1000.0
        query_pessimistic_s = query_optimistic_s * 2.0

        projections[regime] = {
            "vs_overhead_s": round(vs_overhead_s, 1),
            "preload_total_s": round(preload_total_s, 1),
            "query_optimistic_s": round(query_optimistic_s, 1),
            "query_pessimistic_s": round(query_pessimistic_s, 1),
            "wall_optimistic_s": round(
                vs_overhead_s + preload_total_s + query_optimistic_s, 1
            ),
            "wall_pessimistic_s": round(
                vs_overhead_s + preload_total_s + query_pessimistic_s, 1
            ),
        }

    return projections


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_ms(ms: float) -> str:
    """Format milliseconds as a right-aligned 10-char string with commas."""
    return f"{ms:>10,.0f} ms"


def _fmt_duration(seconds: float) -> str:
    """Format seconds as a human-readable duration string (s, min, or h).

    Args:
        seconds: Duration in seconds.

    Returns:
        A concise human-readable string such as ``"34 s"``, ``"4.2 min"``,
        or ``"2.1 h"``.

    Complexity:
        O(1).
    """
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def _print_regime_result(result: dict) -> None:
    """Print the per-phase timing breakdown for one regime step.

    Args:
        result: Dict returned by ``_run_regime_step``.

    Returns:
        None
    """
    regime_label = (
        "SINGLE-FILE REGIME (Test 1 — one growing concatenated file)"
        if result["regime"] == "single"
        else "MULTI-FILE REGIME  (Test 2 — N separate chunk files)"
    )
    kb_size_kb = result["kb_size_bytes"] / 1024
    print(f"\n{regime_label}")
    print(f"  KB size:    {kb_size_kb:.1f} KB  ({result['file_count']} file(s))")
    print(f"  VS create: {_fmt_ms(result['vs_create_ms'])}")
    print(
        f"  Preload:   {_fmt_ms(result['preload_ms'])}"
        f"   [status: {result['preload_status']}]"
    )

    total_query_ms = 0.0
    n_queries = len(result["query_results"])
    for qr in result["query_results"]:
        elapsed = qr["elapsed_ms"]
        server = qr.get("provider_server_latency_ms")
        net_ms = (elapsed - server) if server is not None else None
        server_str = f"{server:,} ms" if server is not None else "N/A"
        net_str = f"{net_ms:,.0f} ms" if net_ms is not None else "N/A"
        citation_tag = "cited" if qr.get("has_citation") else "MISS"
        print(
            f"  Query {qr['q_idx']}/{n_queries}: {_fmt_ms(elapsed)}"
            f"   [server: {server_str} | net: {net_str}"
            f" | in: {qr['input_tokens']:,} tok | out: {qr['output_tokens']:,} tok"
            f" | {citation_tag}]"
        )
        total_query_ms += elapsed

    miss_count = sum(1 for qr in result["query_results"] if not qr.get("has_citation"))
    citation_summary = (
        f"all {n_queries} cited"
        if miss_count == 0
        else f"{miss_count}/{n_queries} MISSED citation"
    )
    print(f"  VS delete: {_fmt_ms(result['vs_delete_ms'])}")
    print(
        f"  ── Step total: {_fmt_duration(result['step_total_ms'] / 1000)}"
        f"  (queries alone: {_fmt_duration(total_query_ms / 1000)})"
        f"  |  Citations: {citation_summary}"
    )


def _print_projection(proj: dict, max_files: int, runs: int) -> None:
    """Print the full-benchmark projection table.

    Args:
        proj: Dict returned by ``_project_total``.
        max_files: Total steps in the real benchmark.
        runs: Total runs in the real benchmark.

    Returns:
        None
    """
    print(f"\n{'='*66}")
    print(f"  PROJECTED FULL BENCHMARK  ({max_files} steps × {runs} run(s))")
    print(f"{'='*66}")

    for regime_key, label in (
        ("single", "Single-file (Test 1)"),
        ("multi", "Multi-file  (Test 2)"),
    ):
        p = proj[regime_key]
        print(f"\n  {label}")
        print(
            f"    VS overhead (create+delete): "
            f"{_fmt_duration(p['vs_overhead_s']):>10}"
            f"  [{runs * max_files} VS × "
            f"{(p['vs_overhead_s'] / (runs * max_files) * 1000):.0f} ms each]"
        )
        print(
            f"    Preload total (all steps):   "
            f"{_fmt_duration(p['preload_total_s']):>10}"
            f"  [linear rate × triangular KB growth]"
        )
        print(
            f"    Queries — optimistic:        "
            f"{_fmt_duration(p['query_optimistic_s']):>10}"
            f"  [probed latency held constant]"
        )
        print(
            f"    Queries — pessimistic:       "
            f"{_fmt_duration(p['query_pessimistic_s']):>10}"
            f"  [2× headroom for server variance]"
        )
        print(
            f"    ── Wall time (optimistic):   "
            f"{_fmt_duration(p['wall_optimistic_s']):>10}"
        )
        print(
            f"    ── Wall time (pessimistic):  "
            f"{_fmt_duration(p['wall_pessimistic_s']):>10}"
        )

    print(f"\n  Assumptions:")
    for assumption in proj["assumptions"]:
        print(f"    • {assumption}")
    print(
        "\n  NOTE: Probed latency is measured at the largest step — earlier"
        "\n        steps run faster, so optimistic estimate may still be high."
    )
    print(f"{'='*66}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse and validate command-line arguments.

    Returns:
        Populated ``argparse.Namespace``.

    Raises:
        SystemExit: If ``--step`` exceeds ``--max-files`` or required args
            are missing.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Probe timing for one step of both KB-growth benchmark regimes "
            "and project full benchmark wall time."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--step",
        type=int,
        required=True,
        help="Step to probe (1-indexed). Must be ≤ --max-files.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=50,
        help=(
            "Total logical steps for projection. "
            "Total raw chunks loaded = max_files × chunks_per_step."
        ),
    )
    parser.add_argument(
        "--chunks-per-step",
        type=int,
        default=2,
        help="Raw ~5 KB chunks per logical step/file (default 2 → ~10 KB steps).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Total runs in the real benchmark (used for projection only).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI model identifier.",
    )
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=_DEFAULT_KB_DIR,
        help="Directory containing kb_chunk_*.txt files.",
    )
    parser.add_argument(
        "--queries-file",
        type=Path,
        default=_DEFAULT_QUERIES_FILE,
        help="Path to the benchmark queries file.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Per-operation retry count.",
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
        help="Skip pre-flight confirmation prompt.",
    )
    args = parser.parse_args()

    if args.step < 1:
        parser.error("--step must be >= 1")
    if args.chunks_per_step < 1:
        parser.error("--chunks-per-step must be >= 1")
    if args.step > args.max_files:
        parser.error(
            f"--step ({args.step}) must be <= --max-files ({args.max_files})"
        )

    return args


def main() -> None:
    """Entry point for the step timing probe.

    Loads environment, parses arguments, optionally prompts for confirmation,
    runs both regimes for the specified step, prints per-phase timings, and
    prints the full-benchmark projection.

    Returns:
        None

    Raises:
        SystemExit: If OPENAI_API_KEY is absent or the user declines
            confirmation.
    """
    logging.basicConfig(level=logging.WARNING)

    args = _parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is required.  Set it in .env or the environment."
        )

    from src.llm_pipeline_telemetry import get_openai_client  # noqa: PLC0415

    client = get_openai_client()
    chunks = collect_chunk_paths(args.kb_dir, args.max_files * args.chunks_per_step)
    queries = load_queries(args.queries_file)
    query_count = len(queries)
    step_kb = sum(c.stat().st_size for c in chunks[: args.step * args.chunks_per_step]) / 1024

    print(f"\n{'='*66}")
    print(
        f"  KB-GROWTH BENCHMARK — STEP TIMING PROBE"
    )
    print(
        f"  Probing step {args.step} / {args.max_files}"
        f"  ({step_kb:.1f} KB cumulative)"
    )
    print(
        f"  Model: {args.model}  |  Queries/step: {query_count}"
        f"  |  Projection: {args.runs} run(s) × {args.max_files} step(s)"
    )
    print(f"{'='*66}")

    if not args.yes:
        vs_count = 2  # one per regime
        api_calls = vs_count * query_count
        print(
            f"\n  This probe will make:"
            f"\n    {vs_count} vector store create + delete pairs"
            f"\n    {vs_count} file uploads"
            f"\n    {api_calls} query API calls"
            f"\n    Model: {args.model}"
        )
        answer = input("\n  Proceed? [y/N] ").strip().lower()
        if answer != "y":
            raise SystemExit("Aborted.")

    kwargs = dict(
        client=client,
        model=args.model,
        queries=queries,
        max_retries=args.max_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
        chunks_per_step=args.chunks_per_step,
    )

    print(f"\nRunning single-file regime probe (step {args.step})…")
    single_result = _run_regime_step(
        "single", chunks, args.step, **kwargs  # type: ignore[arg-type]
    )
    _print_regime_result(single_result)

    print(f"\nRunning multi-file regime probe (step {args.step})…")
    multi_result = _run_regime_step(
        "multi", chunks, args.step, **kwargs  # type: ignore[arg-type]
    )
    _print_regime_result(multi_result)

    proj = _project_total(single_result, multi_result, args.max_files, args.runs)
    _print_projection(proj, args.max_files, args.runs)


if __name__ == "__main__":
    main()
