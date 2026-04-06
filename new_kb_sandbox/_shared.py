"""Shared helpers for the KB-growth benchmark suite.

All pure helper functions have no side-effects and are fully unit-testable
without any API calls.  The stateful benchmark loop ``run_benchmark`` is the
only function that creates network connections.

Dependencies:
    Standard library: csv, dataclasses, logging, math, time, uuid, collections,
    datetime, pathlib, typing.
    Third-party: openai, tqdm, python-dotenv.
    Internal: src.llm_pipeline_telemetry.

Pillar compliance:
    - Pillar 1: All CSV fields match the plan specification exactly; kb_size_bytes
      assertion enforces that both test regimes upload identical total bytes at
      every step, isolating file-structure as the sole variable.
    - Pillar 2: Every pure helper is unit-testable without mocking.
    - Pillar 3: Google-style docstrings with full contracts on every public item.
    - Pillar 4: All configuration flows in via function arguments; no module-level
      constants hold user data; SOLID/DRY throughout.
    - Pillar 5: OPENAI_API_KEY read from .env only; no secrets in CSV output.
    - Pillar 6: Θ analysis documented per function; benchmark loop is
      Θ(R×S×Q) API calls, Θ(R×S²) upload bytes, Θ(R×S) row dicts.
    - Pillar 7: Retry on upload and query; partial-result flush before re-raise;
      temp-file cleanup in finally; structured logging via llm.telemetry.
"""
from __future__ import annotations

import csv
import dataclasses
import logging
import math
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from dotenv import load_dotenv

from src.llm_pipeline_telemetry import CallContext, log_llm_call

if TYPE_CHECKING:
    from openai import OpenAI

load_dotenv()

_logger = logging.getLogger("llm.telemetry")

# Columns for which population std-dev is emitted in metrics_averaged.csv.
# Maps source column name → std-dev column name.
_STD_COLUMN_MAP: dict[str, str] = {
    "ask_elapsed_ms_mean": "ask_elapsed_ms_std",
    "ask_input_tokens_mean": "ask_input_tokens_std",
    "ask_output_tokens_mean": "ask_output_tokens_std",
    "ask_throughput_tokens_per_sec_mean": "ask_throughput_tokens_per_sec_std",
    "ask_aggregate_throughput_tokens_per_sec": "ask_aggregate_throughput_tokens_per_sec_std",
}

# Columns copied verbatim from the first row (not numeric or not meaningful to average).
_STRING_COLUMNS: frozenset[str] = frozenset({"model", "preload_status"})

# Columns that are summed (not averaged) across runs.
_SUM_COLUMNS: frozenset[str] = frozenset({"server_latency_missing_count"})


# ---------------------------------------------------------------------------
# Pure helpers — no side effects, fully unit-testable
# ---------------------------------------------------------------------------


def load_queries(path: Path) -> list[str]:
    """Load benchmark queries from a text file delimited by ``\\n---\\n``.

    Args:
        path: Path to the queries file.  Each query is separated from the next
            by a line containing only ``---``.

    Returns:
        A list of non-empty, stripped query strings.

    Raises:
        ValueError: If the file contains no non-empty query strings after
            splitting and stripping.
        OSError: If the file cannot be read.

    Examples:
        >>> from pathlib import Path
        >>> p = Path("new_kb_sandbox/prompts/kb_benchmark_queries.txt")
        >>> queries = load_queries(p)
        >>> len(queries) > 0
        True

    Preconditions:
        - ``path`` must point to an existing, readable text file encoded as UTF-8.

    Postconditions:
        - Every string in the returned list is non-empty after stripping.

    Complexity:
        Θ(n) where n is the number of characters in the file.
    """
    raw = path.read_text(encoding="utf-8")
    queries = [chunk.strip() for chunk in raw.split("\n---\n") if chunk.strip()]
    if not queries:
        raise ValueError(f"No queries found in {path}")
    return queries


def build_growing_file(chunks: list[Path], step: int, tmp_dir: Path) -> Path:
    """Concatenate the first *step* chunk files into a single binary temp file.

    All I/O is in binary mode (``rb`` / ``wb``) so that no platform newline
    translation occurs.  The output byte length is guaranteed to equal the sum
    of the source chunk sizes:
    ``len(out.read_bytes()) == sum(c.stat().st_size for c in chunks[:step])``.

    Args:
        chunks: Ordered list of source chunk ``Path`` objects.  Only the first
            ``step`` entries are read.
        step: Number of chunks to concatenate (1-indexed; ``chunks[:step]``).
        tmp_dir: Directory in which to write the temporary output file.

    Returns:
        Path to the newly created concatenated file inside ``tmp_dir``.
        The file name is ``growing_{step:04d}.bin``.

    Raises:
        IndexError: If ``step`` exceeds ``len(chunks)``.
        OSError: If any chunk file cannot be read or the output cannot be written.

    Examples:
        >>> # After call, out.stat().st_size == sum of first `step` chunk sizes
        >>> out = build_growing_file(chunks, step=3, tmp_dir=tmp)
        >>> out.stat().st_size == sum(c.stat().st_size for c in chunks[:3])
        True

    Preconditions:
        - ``tmp_dir`` must exist and be writable.
        - ``0 < step <= len(chunks)``.
        - Each file in ``chunks[:step]`` must be readable.

    Postconditions:
        - The returned path exists inside ``tmp_dir``.
        - File content equals the binary concatenation of ``chunks[:step]``
          with no separator.

    Complexity:
        Θ(B) where B = sum of byte sizes of ``chunks[:step]``.
    """
    out_path = tmp_dir / f"growing_{step:04d}.bin"
    with out_path.open("wb") as out_fh:
        for chunk in chunks[:step]:
            with chunk.open("rb") as in_fh:
                out_fh.write(in_fh.read())
    return out_path


def collect_chunk_paths(kb_dir: Path, max_files: int) -> list[Path]:
    """Return the first *max_files* ``kb_chunk_*.txt`` paths from *kb_dir*, sorted.

    Sorting is lexicographic on the file name, which matches numeric order for
    zero-padded names such as ``kb_chunk_0000.txt``.

    Args:
        kb_dir: Directory containing ``kb_chunk_*.txt`` files.
        max_files: Minimum number of files required.  Only the first
            ``max_files`` are returned.

    Returns:
        A list of exactly ``max_files`` ``Path`` objects, sorted by name.

    Raises:
        ValueError: If fewer than ``max_files`` matching files exist in
            ``kb_dir``.

    Examples:
        >>> paths = collect_chunk_paths(Path("new_kb_sandbox/..."), max_files=10)
        >>> len(paths)
        10

    Preconditions:
        - ``kb_dir`` must exist and be readable.

    Postconditions:
        - Returned list has exactly ``max_files`` entries.
        - Paths are sorted lexicographically.

    Complexity:
        Θ(F log F) where F is the total number of matching files in kb_dir.
    """
    paths = sorted(kb_dir.glob("kb_chunk_*.txt"))
    if len(paths) < max_files:
        raise ValueError(
            f"Only {len(paths)} chunk files found in {kb_dir}; need at least {max_files}"
        )
    return paths[:max_files]


def percentile(values: list[float], pct: float) -> float:
    """Return the *pct*-th percentile of *values* using ceiling index selection.

    Matches the algorithm used in the existing ``benchmark_kb_scaling.py``
    script, ensuring consistency across all benchmark measurements.

    Args:
        values: A list of numeric values.  May be empty.
        pct: Percentile to compute, in [0, 100].

    Returns:
        The *pct*-th percentile value.  Returns ``0.0`` for an empty list.

    Examples:
        >>> percentile([1.0, 2.0, 3.0, 4.0], 50)
        2.0
        >>> percentile([], 95)
        0.0

    Preconditions:
        - ``0 <= pct <= 100``.

    Postconditions:
        - Returns a value that exists in ``values`` (no interpolation).

    Complexity:
        Θ(n log n) where n = len(values) due to sorting.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, math.ceil((pct / 100.0) * len(ordered)) - 1))
    return ordered[idx]


def _population_stdev(values: list[float]) -> float:
    """Compute population standard deviation of *values*.

    Args:
        values: Non-empty list of numeric values.

    Returns:
        Population standard deviation; ``0.0`` for lists with fewer than 2
        elements.

    Complexity:
        Θ(n).
    """
    if len(values) <= 1:
        return 0.0
    n = len(values)
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def compute_step_metrics(query_results: list[dict]) -> dict:
    """Aggregate per-query result dicts into a single step metrics dict.

    Pure function — no side effects.  Produces all ``ask_*``,
    ``server_latency_missing_count``, and ``citation_miss_count`` fields that
    appear in ``metrics_per_run.csv``.

    Throughput columns:

    * ``ask_throughput_tokens_per_sec_mean``: mean of per-query
      ``throughput_output_tokens_per_sec`` values (always non-null; uses
      ``client_elapsed_ms`` fallback per ``log_llm_call`` logic).
    * ``ask_aggregate_throughput_tokens_per_sec``: ``output_tokens_total /
      (sum_latency_ms / 1000)`` where each per-query latency uses
      ``provider_server_latency_ms`` when available, else ``elapsed_ms``.
      Not distorted by short queries; use this column for KB-growth trend
      analysis.

    Args:
        query_results: List of dicts as returned by ``query_with_retry``.
            Each dict must contain: ``elapsed_ms`` (float), ``input_tokens``
            (int), ``output_tokens`` (int), ``throughput_output_tokens_per_sec``
            (float), ``provider_server_latency_ms`` (int or None),
            ``has_citation`` (bool).

    Returns:
        A dict with fields: ``ask_elapsed_ms_mean``, ``ask_elapsed_ms_p50``,
        ``ask_elapsed_ms_p95``, ``ask_elapsed_ms_min``, ``ask_elapsed_ms_max``,
        ``ask_input_tokens_mean``, ``ask_input_tokens_total``,
        ``ask_output_tokens_mean``, ``ask_output_tokens_total``,
        ``ask_throughput_tokens_per_sec_mean``,
        ``ask_aggregate_throughput_tokens_per_sec``,
        ``ask_provider_server_latency_ms_mean`` (nullable),
        ``server_latency_missing_count``, ``citation_miss_count``.

    Raises:
        ValueError: If ``query_results`` is empty.

    Preconditions:
        - ``len(query_results) >= 1``.

    Postconditions:
        - ``ask_throughput_tokens_per_sec_mean > 0`` whenever any output tokens
          were produced (non-null by construction).
        - ``ask_aggregate_throughput_tokens_per_sec >= 0.0`` always.
        - ``ask_provider_server_latency_ms_mean`` is ``None`` iff all queries
          had ``provider_server_latency_ms=None``.

    Complexity:
        Θ(Q log Q) where Q = len(query_results) due to sorting for percentile.
    """
    if not query_results:
        raise ValueError("query_results must be non-empty")

    elapsed = [r["elapsed_ms"] for r in query_results]
    input_tokens = [r["input_tokens"] for r in query_results]
    output_tokens = [r["output_tokens"] for r in query_results]
    throughputs = [r["throughput_output_tokens_per_sec"] for r in query_results]
    server_latencies: list[Optional[int]] = [r.get("provider_server_latency_ms") for r in query_results]
    has_citations = [r.get("has_citation", False) for r in query_results]

    n = len(query_results)
    server_latency_missing_count = sum(1 for sl in server_latencies if sl is None)

    # Aggregate throughput: use server latency where available, else client elapsed.
    sum_latency_ms = sum(
        float(sl) if sl is not None else r["elapsed_ms"]
        for r, sl in zip(query_results, server_latencies)
    )
    total_output = sum(output_tokens)
    agg_throughput = (total_output / (sum_latency_ms / 1000.0)) if sum_latency_ms > 0 else 0.0

    # Provider server latency mean (nullable).
    valid_server_lats = [float(sl) for sl in server_latencies if sl is not None]
    provider_latency_mean: Optional[float] = (
        sum(valid_server_lats) / len(valid_server_lats) if valid_server_lats else None
    )

    return {
        "ask_elapsed_ms_mean": round(sum(elapsed) / n, 3),
        "ask_elapsed_ms_p50": round(percentile(elapsed, 50), 3),
        "ask_elapsed_ms_p95": round(percentile(elapsed, 95), 3),
        "ask_elapsed_ms_min": round(min(elapsed), 3),
        "ask_elapsed_ms_max": round(max(elapsed), 3),
        "ask_input_tokens_mean": round(sum(input_tokens) / n, 3),
        "ask_input_tokens_total": sum(input_tokens),
        "ask_output_tokens_mean": round(sum(output_tokens) / n, 3),
        "ask_output_tokens_total": total_output,
        "ask_throughput_tokens_per_sec_mean": round(sum(throughputs) / n, 3),
        "ask_aggregate_throughput_tokens_per_sec": round(agg_throughput, 3),
        "ask_provider_server_latency_ms_mean": (
            round(provider_latency_mean, 3) if provider_latency_mean is not None else None
        ),
        "server_latency_missing_count": server_latency_missing_count,
        "citation_miss_count": sum(1 for hc in has_citations if not hc),
    }


def average_rows(rows: list[dict]) -> list[dict]:
    """Group per-run rows by step and average all numeric columns across runs.

    Produces one row per step with:

    * Means for all numeric columns (including ``preload_elapsed_ms``).
    * Population standard deviation for ``ask_elapsed_ms_mean``,
      ``ask_input_tokens_mean``, ``ask_output_tokens_mean``,
      ``ask_throughput_tokens_per_sec_mean``, and
      ``ask_aggregate_throughput_tokens_per_sec``.
    * ``server_latency_missing_count`` **summed** (not averaged) across runs.
    * ``ask_provider_server_latency_ms_mean`` averaged over non-``None`` values;
      remains ``None`` when all runs returned ``None``.
    * String columns (``model``, ``preload_status``) taken from the first row.

    Args:
        rows: List of row dicts as produced by the benchmark loop (i.e. all
            rows from ``metrics_per_run.csv``).  Must be non-empty.  Each row
            must contain at least a ``step`` key.

    Returns:
        A list of averaged row dicts, one per unique step, sorted by step.
        The column order mirrors ``rows[0]`` minus ``run`` and
        ``vector_store_id``, followed by the five std-dev columns.

    Raises:
        ValueError: If ``rows`` is empty.

    Preconditions:
        - ``len(rows) >= 1``.

    Postconditions:
        - Returned list is sorted by ``step`` ascending.
        - Each returned dict has no ``run`` or ``vector_store_id`` key.
        - Each returned dict has exactly one entry per std-dev column.

    Complexity:
        Θ(R × S) where R = number of runs per step, S = number of distinct
        steps.
    """
    if not rows:
        raise ValueError("rows must be non-empty")

    by_step: defaultdict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_step[int(row["step"])].append(row)

    result: list[dict] = []
    for step in sorted(by_step.keys()):
        group = by_step[step]
        first = group[0]

        # Ordered key list excluding run/vector_store_id.
        base_keys = [k for k in first.keys() if k not in ("run", "vector_store_id")]

        avg_row: dict = {}
        for key in base_keys:
            vals = [row[key] for row in group if key in row]
            if not vals:
                continue
            if key in _STRING_COLUMNS:
                avg_row[key] = vals[0]
            elif key in _SUM_COLUMNS:
                avg_row[key] = sum(int(v) for v in vals)
            elif key == "ask_provider_server_latency_ms_mean":
                non_null = [v for v in vals if v is not None]
                avg_row[key] = round(sum(non_null) / len(non_null), 3) if non_null else None
            elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
                avg_row[key] = round(sum(vals) / len(vals), 3)
            else:
                avg_row[key] = vals[0]

        # Append population std-dev columns.
        for src_col, std_col in _STD_COLUMN_MAP.items():
            src_vals = [row.get(src_col) for row in group]
            numeric_vals = [
                v for v in src_vals
                if v is not None and isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            avg_row[std_col] = round(_population_stdev(numeric_vals), 3) if numeric_vals else 0.0

        result.append(avg_row)

    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write *rows* to a CSV file at *path*, creating parent directories as needed.

    ``None`` values are written as empty strings (standard CSV convention for
    nullable numeric columns such as ``ask_provider_server_latency_ms_mean``).

    Args:
        path: Destination file path.
        rows: Non-empty list of dicts with identical key sets.

    Returns:
        None

    Raises:
        ValueError: If ``rows`` is empty.
        OSError: If the file cannot be written.

    Preconditions:
        - ``len(rows) >= 1``.
        - All dicts in ``rows`` have the same keys.

    Postconditions:
        - A UTF-8 CSV file with a header row exists at ``path``.
        - The file has ``len(rows)`` data rows.

    Complexity:
        Θ(R × C) where R = len(rows), C = number of columns.
    """
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})


def estimate_cost(
    runs: int,
    max_files: int,
    query_count: int,
    avg_kb_bytes_per_step: int,
) -> dict:
    """Estimate the API call count and data volume before a benchmark run.

    Uses conservative flat estimates suitable for pre-flight confirmation:
    * ``query_calls``: Θ(R × S × Q) — total Responses API calls.
    * ``upload_calls``: Θ(R × S) — total vector-store upload operations.
    * ``rough_input_tokens``: ``query_calls × 1 500`` — conservative flat
      estimate per call.
    * ``upload_mb``: total upload bytes in MB (Θ(R × S²) in data volume).

    Args:
        runs: Number of independent benchmark repetitions (R).
        max_files: Number of KB-size steps (S).
        query_count: Number of queries per step (Q).
        avg_kb_bytes_per_step: Average bytes uploaded per step (used for the
            upload volume estimate).

    Returns:
        A dict with keys ``query_calls`` (int), ``upload_calls`` (int),
        ``rough_input_tokens`` (int), and ``upload_mb`` (float).

    Examples:
        >>> estimate_cost(runs=5, max_files=50, query_count=6, avg_kb_bytes_per_step=5120)
        {'query_calls': 1500, 'upload_calls': 250, 'rough_input_tokens': 2250000, 'upload_mb': ...}

    Complexity:
        O(1).
    """
    query_calls = runs * max_files * query_count
    upload_calls = runs * max_files
    rough_input_tokens = query_calls * 1500
    # Quadratic: each run uploads 1+2+...+S steps; total = R × S(S+1)/2 × avg_bytes
    total_bytes = runs * (max_files * (max_files + 1) // 2) * avg_kb_bytes_per_step
    upload_mb = total_bytes / (1024 * 1024)
    return {
        "query_calls": query_calls,
        "upload_calls": upload_calls,
        "rough_input_tokens": rough_input_tokens,
        "upload_mb": round(upload_mb, 2),
    }


def confirm_run(estimates: dict) -> None:
    """Print cost estimates and prompt the user to confirm before proceeding.

    Blocks on stdin.  Raises ``SystemExit`` if the user does not enter ``y``
    or ``Y``.

    Args:
        estimates: Dict as returned by ``estimate_cost``.

    Returns:
        None

    Raises:
        SystemExit: If the user enters anything other than ``y`` or ``Y``.

    Examples:
        >>> # Monkeypatch input() to return "y" in tests.

    Preconditions:
        - Running in an interactive terminal (stdin readable).

    Postconditions:
        - Returns normally only when the user confirms with ``y``/``Y``.

    Complexity:
        O(1).
    """
    print("--- Benchmark cost estimate ---")
    print(f"  Query API calls  : {estimates['query_calls']}")
    print(f"  Upload operations: {estimates['upload_calls']}")
    print(f"  Rough input tokens: {estimates['rough_input_tokens']:,}")
    print(f"  Estimated upload data: {estimates['upload_mb']:.1f} MB (quadratic)")
    answer = input("Proceed? [y/N]: ").strip()
    if answer.lower() != "y":
        raise SystemExit("Benchmark cancelled by user.")


def upload_with_retry(
    client: "OpenAI",
    *,
    vector_store_id: str,
    file_paths: list[Path],
    max_retries: int,
    retry_sleep_seconds: float,
) -> str:
    """Upload files to an existing vector store with automatic retry on failure.

    On each attempt, opens all file streams in binary mode, calls
    ``vector_stores.file_batches.upload_and_poll``, and closes all streams in
    a ``finally`` block (ensuring no file descriptor leak even on exception).
    On non-final-attempt failure, sleeps for ``retry_sleep_seconds × (attempt+1)``
    before retrying (linear back-off).

    Args:
        client: Authenticated ``openai.OpenAI`` client.
        vector_store_id: ID of the pre-existing OpenAI vector store.
        file_paths: Local file paths to upload.
        max_retries: Maximum number of retry attempts (0 = no retry).
        retry_sleep_seconds: Base sleep duration in seconds; multiplied by the
            attempt index for linear back-off.

    Returns:
        The batch status string (e.g. ``"completed"``).

    Raises:
        Any exception raised by ``upload_and_poll`` after all retries are
        exhausted.

    Preconditions:
        - ``vector_store_id`` must refer to an existing, accessible vector store.
        - All paths in ``file_paths`` must be readable.

    Postconditions:
        - All file streams opened during each attempt are closed before the
          function returns (or raises).

    Complexity:
        Θ(B) per attempt where B is the total upload byte volume; at most
        (max_retries + 1) attempts.
    """
    for attempt in range(max_retries + 1):
        streams = [fp.open("rb") for fp in file_paths]
        try:
            batch = client.vector_stores.file_batches.upload_and_poll(
                vector_store_id=vector_store_id,
                files=streams,
            )
            return str(getattr(batch, "status", "unknown"))
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(retry_sleep_seconds * (attempt + 1))
        finally:
            for s in streams:
                s.close()
    raise RuntimeError("unreachable")  # pragma: no cover


def _has_file_citation(response: object) -> bool:
    """Return True iff the Responses API response contains a ``file_citation`` annotation.

    Mirrors the implementation in ``src/multi/file_store.py``.

    Args:
        response: A parsed ``openai.types.responses.Response`` object.

    Returns:
        ``True`` if at least one ``file_citation`` annotation is present.

    Complexity:
        O(n) where n = total annotation count across all output items.
    """
    output = getattr(response, "output", None)
    if not output:
        return False
    for item in output:
        if getattr(item, "type", None) != "message":
            continue
        for block in getattr(item, "content", []):
            if getattr(block, "type", None) != "output_text":
                continue
            if any(
                getattr(ann, "type", None) == "file_citation"
                for ann in getattr(block, "annotations", [])
            ):
                return True
    return False


def query_with_retry(
    client: "OpenAI",
    *,
    model: str,
    vector_store_id: str,
    query: str,
    ctx: CallContext,
    max_retries: int,
    retry_sleep_seconds: float,
    progress_bar: object = None,
) -> dict:
    """Issue a single file-search query with automatic retry on failure.

    Uses ``client.with_raw_response.responses.create`` so the
    ``openai-processing-ms`` header is always accessible.  Emits an
    ``llm_call`` record via ``log_llm_call`` on each successful call.

    The ``progress_bar`` argument accepts any object with a ``set_postfix``
    method (duck-typed; no ``tqdm`` import needed in this module).

    Args:
        client: Authenticated ``openai.OpenAI`` client with ``with_raw_response``
            support.
        model: OpenAI model identifier (e.g. ``"gpt-4o-mini"``).
        vector_store_id: ID of the vector store to query.
        query: Natural-language question string.
        ctx: ``CallContext`` for the current benchmark step.  A snapshot is
            created via ``dataclasses.replace(ctx, agent="benchmark",
            span="query")`` before each call; the original ``ctx`` is not
            mutated, but its shared ``records`` list is appended to.
        max_retries: Maximum number of retry attempts (0 = no retry).
        retry_sleep_seconds: Sleep between retries in seconds (linear back-off).
        progress_bar: Optional object with ``set_postfix(**kwargs)`` — any
            ``tqdm`` bar or duck-typed stub.

    Returns:
        A dict with keys:
        ``elapsed_ms`` (float), ``input_tokens`` (int), ``output_tokens``
        (int), ``provider_server_latency_ms`` (int or None),
        ``throughput_output_tokens_per_sec`` (float), ``has_citation`` (bool).

    Raises:
        Any exception raised by the OpenAI SDK after all retries are exhausted.

    Preconditions:
        - ``vector_store_id`` must refer to an accessible vector store.

    Postconditions:
        - On success, exactly one ``llm_call`` record is appended to
          ``ctx.records`` via ``log_llm_call``.
        - ``throughput_output_tokens_per_sec`` is always non-negative.

    Complexity:
        O(1) amortized per call (single HTTP round-trip); at most
        (max_retries + 1) attempts.
    """
    call_ctx = dataclasses.replace(ctx, agent="benchmark", span="query")

    for attempt in range(max_retries + 1):
        if progress_bar is not None:
            progress_bar.set_postfix(status=f"querying (attempt {attempt + 1})")
        call_start_ts = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()
        try:
            raw = client.with_raw_response.responses.create(
                model=model,
                input=query,
                tools=[{"type": "file_search", "vector_store_ids": [vector_store_id]}],
            )
            client_elapsed_ms = (time.perf_counter() - t0) * 1000.0
            call_end_ts = datetime.now(timezone.utc).isoformat()

            raw_ms_header = raw.headers.get("openai-processing-ms")
            provider_server_latency_ms: Optional[int] = (
                int(raw_ms_header)
                if raw_ms_header and int(raw_ms_header) > 0
                else None
            )

            parsed = raw.parse()
            usage = getattr(parsed, "usage", None)
            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
            has_citation = _has_file_citation(parsed)

            log_llm_call(
                call_ctx,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                provider_server_latency_ms=provider_server_latency_ms,
                client_elapsed_ms=client_elapsed_ms,
                call_start_ts=call_start_ts,
                call_end_ts=call_end_ts,
                status="ok",
            )
            # Throughput is computed by log_llm_call; retrieve from the record.
            throughput = call_ctx.records[-1]["throughput_output_tokens_per_sec"]

            if progress_bar is not None:
                progress_bar.set_postfix(
                    status="done",
                    latency_ms=round(client_elapsed_ms, 0),
                )
            return {
                "elapsed_ms": round(client_elapsed_ms, 3),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "provider_server_latency_ms": provider_server_latency_ms,
                "throughput_output_tokens_per_sec": throughput,
                "has_citation": has_citation,
            }
        except Exception:
            if attempt >= max_retries:
                raise
            time.sleep(retry_sleep_seconds)
    raise RuntimeError("unreachable")  # pragma: no cover


def run_benchmark(
    get_files_for_step: Callable[[int, Path], list[Path]],
    *,
    runs: int,
    max_files: int,
    queries: list[str],
    chunks: list[Path],
    client: "OpenAI",
    model: str,
    output_dir: Path,
    keep_vector_stores: bool,
    max_retries: int,
    retry_sleep_seconds: float,
    vs_name_prefix: str,
) -> Path:
    """Run the full KB-growth benchmark and write results to a timestamped directory.

    Creates ``output_dir / <YYYY-MM-DDTHHMMSS> / metrics_per_run.csv`` and
    ``metrics_averaged.csv`` on success.  On exception during a step, flushes
    all completed rows to ``metrics_per_run.csv`` as a partial result and
    re-raises; the averaged CSV is not written for partial runs.

    Growth regime is injected via *get_files_for_step*:

    * Test 1 (single-file): ``lambda step, tmp: [build_growing_file(chunks, step, tmp)]``
    * Test 2 (multi-file):  ``lambda step, _: chunks[:step]``

    The benchmark loop is instrumented with nested ``tqdm`` progress bars:
    an outer run bar and a per-run step bar (``leave=False``).

    Args:
        get_files_for_step: Callable ``(step: int, tmp_dir: Path) -> list[Path]``
            that returns the file paths to upload at a given step.  Files whose
            ``.parent == tmp_dir`` are cleaned up in the ``finally`` block.
        runs: Number of independent repeat runs (R).
        max_files: Number of KB-size steps per run (S).
        queries: List of query strings (Q per step).
        chunks: Full ordered list of source chunk paths (used for the
            ``kb_size_bytes`` assertion).
        client: Authenticated ``openai.OpenAI`` singleton.
        model: Model identifier string.
        output_dir: Parent results directory.  A timestamped subdirectory is
            created inside it; existing subdirectories are never modified.
        keep_vector_stores: When ``True``, vector stores are not deleted after
            each step.  Use ``cleanup_vector_stores.py`` to recover.
        max_retries: Per-query and per-upload retry count.
        retry_sleep_seconds: Base sleep between retries in seconds.
        vs_name_prefix: Prefix for vector store names (e.g.
            ``"kb-bench-single"``).

    Returns:
        Path to the timestamped run directory that contains the CSV files.

    Raises:
        Any exception that occurs during upload or query after all retries are
        exhausted.  Completed rows are flushed to ``metrics_per_run.csv``
        before re-raising.

    Preconditions:
        - ``runs >= 1``, ``max_files >= 1``, ``len(queries) >= 1``.
        - ``len(chunks) >= max_files``.
        - All chunk files in ``chunks[:max_files]`` must be readable.
        - ``OPENAI_API_KEY`` must be set in the environment.

    Postconditions:
        - On success: ``run_dir / metrics_per_run.csv`` and
          ``run_dir / metrics_averaged.csv`` exist.
        - On partial run (exception): only ``metrics_per_run.csv`` exists with
          completed rows.
        - No previous run directories under ``output_dir`` are modified.

    Complexity:
        - API calls:    Θ(R × S × Q)
        - VS creations: Θ(R × S)
        - Upload bytes: Θ(R × S²) — quadratic due to growing KB each step.
        - Memory:       Θ(R × S) row dicts ≈ 40 KB at defaults; negligible.
        - Temp files:   at most one file on disk at a time (Test 1 only).
    """
    from tqdm import tqdm  # imported here so _shared.py is usable without tqdm in unit tests

    run_dir = output_dir / datetime.now().strftime("%Y-%m-%dT%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = run_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    per_run_rows: list[dict] = []

    run_bar = tqdm(range(1, runs + 1), desc="runs", unit="run")
    try:
        for run in run_bar:
            step_bar = tqdm(range(1, max_files + 1), desc="steps", unit="step", leave=False)
            for step in step_bar:
                tmp_files: list[Path] = []
                ctx = CallContext(pipeline="kb_growth_benchmark")
                try:
                    file_paths = get_files_for_step(step, tmp_dir)
                    # Collect only temp files for guaranteed cleanup.
                    tmp_files = [p for p in file_paths if p.parent == tmp_dir]

                    kb_size_bytes = sum(f.stat().st_size for f in file_paths)
                    expected_kb_size = step * chunks[0].stat().st_size
                    assert kb_size_bytes == expected_kb_size, (
                        f"kb_size_bytes mismatch at run={run} step={step}: "
                        f"{kb_size_bytes} != {expected_kb_size}"
                    )
                    _logger.debug(
                        "run=%d step=%d kb_size_bytes=%d", run, step, kb_size_bytes
                    )

                    ms_ts = int(time.time() * 1000)
                    uuid4_hex4 = uuid.uuid4().hex[:4]
                    vs_name = f"{vs_name_prefix}-{run}-{step}-{ms_ts}-{uuid4_hex4}"
                    vs = client.vector_stores.create(name=vs_name)

                    step_bar.set_postfix(
                        status="uploading", kb=f"{kb_size_bytes // 1024}KB"
                    )
                    t_upload_start = time.perf_counter()
                    preload_status = upload_with_retry(
                        client,
                        vector_store_id=vs.id,
                        file_paths=file_paths,
                        max_retries=max_retries,
                        retry_sleep_seconds=retry_sleep_seconds,
                    )
                    preload_elapsed_ms = round(
                        (time.perf_counter() - t_upload_start) * 1000.0, 3
                    )

                    query_results: list[dict] = []
                    for q_idx, query in enumerate(queries, start=1):
                        step_bar.set_postfix(
                            status=f"querying {q_idx}/{len(queries)}"
                        )
                        result = query_with_retry(
                            client,
                            model=model,
                            vector_store_id=vs.id,
                            query=query,
                            ctx=ctx,
                            max_retries=max_retries,
                            retry_sleep_seconds=retry_sleep_seconds,
                            progress_bar=step_bar,
                        )
                        query_results.append(result)

                    step_metrics = compute_step_metrics(query_results)
                    step_bar.set_postfix(status="deleting VS")
                    if not keep_vector_stores:
                        client.vector_stores.delete(vs.id)

                    row = {
                        "run": run,
                        "step": step,
                        "kb_size_bytes": kb_size_bytes,
                        "file_count": len(file_paths),
                        "model": model,
                        "vector_store_id": vs.id,
                        "preload_elapsed_ms": preload_elapsed_ms,
                        "preload_status": preload_status,
                        **step_metrics,
                    }
                    per_run_rows.append(row)
                    step_bar.set_postfix(
                        status="done",
                        latency_ms=round(step_metrics["ask_elapsed_ms_mean"], 0),
                        input_tok=step_metrics["ask_input_tokens_total"],
                    )

                except Exception as exc:
                    _logger.warning(
                        "Benchmark interrupted at run=%d step=%d: %s: %s",
                        run,
                        step,
                        type(exc).__name__,
                        exc,
                    )
                    if per_run_rows:
                        write_csv(run_dir / "metrics_per_run.csv", per_run_rows)
                    raise
                finally:
                    for f in tmp_files:
                        f.unlink(missing_ok=True)
    finally:
        run_bar.close()

    write_csv(run_dir / "metrics_per_run.csv", per_run_rows)
    averaged = average_rows(per_run_rows)
    write_csv(run_dir / "metrics_averaged.csv", averaged)
    _logger.info("Results written to %s", run_dir)
    return run_dir
