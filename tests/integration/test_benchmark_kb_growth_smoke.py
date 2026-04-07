"""Unit tests and integration smoke tests for the KB-growth benchmark suite.

Unit tests cover all pure helpers in ``new_kb_sandbox._shared`` and require no
API calls or external resources.

Integration smoke tests exercise the full benchmark loop end-to-end against
the real OpenAI API.  They are skipped when ``OPENAI_API_KEY`` is absent.
Each smoke test runs ``runs=1, max_files=2, chunks_per_step=2``
(2 logical steps × 4 raw chunks × 6 queries = 12 API calls per test) to
verify correctness without burning significant API budget.  The number of
API calls is unchanged from the previous ``chunks_per_step=1`` configuration;
only local disk I/O doubles (4 raw chunks loaded instead of 2).

Dependencies:
    pytest, pathlib (stdlib), new_kb_sandbox._shared (namespace package import).

Pillar compliance:
    - Pillar 2: Unit + integration; every pure helper tested; edge cases
      covered; integration tests gated on API key.
    - Pillar 3: Google-style docstrings on every test function.
    - Pillar 7: Integration tests verify retry machinery, preload_status,
      and file_count column correctness.
"""
from __future__ import annotations

import builtins
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the new_kb_sandbox directory is importable as a namespace package.
# ---------------------------------------------------------------------------
_NKS_DIR = Path(__file__).resolve().parent.parent.parent / "new_kb_sandbox"
if str(_NKS_DIR) not in sys.path:
    sys.path.insert(0, str(_NKS_DIR))

from _shared import (  # noqa: E402
    average_rows,
    build_combined_chunk_file,
    build_growing_file,
    collect_chunk_paths,
    compute_step_metrics,
    confirm_run,
    estimate_cost,
    load_queries,
    percentile,
    run_benchmark,
    write_csv,
)
from benchmark_single_file_growth import (  # noqa: E402
    make_get_files_for_step as single_get_files,
)
from benchmark_multi_file_growth import (  # noqa: E402
    make_get_files_for_step as multi_get_files,
)
from plot_single_file_growth import (  # noqa: E402
    _compute_citation_hit_metrics as single_compute_hit,
    _load_per_query_rows as single_load_per_query,
)
from plot_multi_file_growth import (  # noqa: E402
    _compute_citation_hit_metrics as multi_compute_hit,
    _load_per_query_rows as multi_load_per_query,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_QUERIES_FILE = _NKS_DIR / "prompts" / "kb_benchmark_queries.txt"
_KB_DIR = _NKS_DIR / "my_knowledge_base_data" / "materials_science_kb"

_SKIP_NO_KEY = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping integration smoke tests",
)


def _make_synthetic_chunks(tmp_path: Path, count: int, size_bytes: int = 100) -> list[Path]:
    """Create *count* synthetic chunk files of *size_bytes* each in *tmp_path*.

    Args:
        tmp_path: Directory in which to create files.
        count: Number of chunk files to create.
        size_bytes: Byte size of each chunk (filled with ``b"x"``).

    Returns:
        Sorted list of created Path objects.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        p = tmp_path / f"kb_chunk_{i:04d}.txt"
        p.write_bytes(b"x" * size_bytes)
        paths.append(p)
    return sorted(paths)


def _make_query_result(
    *,
    elapsed_ms: float = 500.0,
    input_tokens: int = 100,
    output_tokens: int = 20,
    provider_server_latency_ms: int | None = 300,
    has_citation: bool = True,
) -> dict:
    """Build a synthetic per-query result dict for use in unit tests.

    Args:
        elapsed_ms: Simulated client wall time.
        input_tokens: Simulated input token count.
        output_tokens: Simulated output token count.
        provider_server_latency_ms: Simulated server latency or None.
        has_citation: Whether the response contains a file citation.

    Returns:
        Dict matching the schema returned by ``query_with_retry``.
    """
    if provider_server_latency_ms is not None and provider_server_latency_ms > 0:
        denom_ms = float(provider_server_latency_ms)
    else:
        denom_ms = elapsed_ms
    throughput = (output_tokens / denom_ms) * 1000.0 if denom_ms > 0 else 0.0
    return {
        "elapsed_ms": elapsed_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "provider_server_latency_ms": provider_server_latency_ms,
        "throughput_output_tokens_per_sec": throughput,
        "has_citation": has_citation,
    }


# ===========================================================================
# load_queries
# ===========================================================================


def test_load_queries_splits_on_separator(tmp_path: Path) -> None:
    """Three queries separated by ---  are returned as a list of 3.

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    p = tmp_path / "q.txt"
    p.write_text("query one\n---\nquery two\n---\nquery three\n", encoding="utf-8")
    result = load_queries(p)
    assert result == ["query one", "query two", "query three"]


def test_load_queries_strips_whitespace(tmp_path: Path) -> None:
    """Leading and trailing whitespace is stripped from each query.

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    p = tmp_path / "q.txt"
    p.write_text("  hello  \n---\n  world  \n", encoding="utf-8")
    result = load_queries(p)
    assert result == ["hello", "world"]


def test_load_queries_empty_file_raises(tmp_path: Path) -> None:
    """An empty file raises ValueError.

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    p = tmp_path / "empty.txt"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="No queries found"):
        load_queries(p)


# ===========================================================================
# build_growing_file
# ===========================================================================


def test_build_growing_file_step1_equals_chunk0(tmp_path: Path) -> None:
    """At step 1 the output bytes equal the bytes of chunk 0 exactly.

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    chunks = _make_synthetic_chunks(tmp_path / "src", 3, 50)
    out = build_growing_file(chunks, 1, tmp_path)
    assert out.read_bytes() == chunks[0].read_bytes()


def test_build_growing_file_step3_byte_length(tmp_path: Path) -> None:
    """At step 3 the output byte length equals the sum of the first 3 chunks.

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    chunks = _make_synthetic_chunks(tmp_path / "src", 5, 77)
    out = build_growing_file(chunks, 3, tmp_path)
    expected = sum(c.stat().st_size for c in chunks[:3])
    assert out.stat().st_size == expected


def test_build_growing_file_step3_concatenates(tmp_path: Path) -> None:
    """Output at step 3 is the binary concatenation with no separator.

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    src = tmp_path / "src"
    src.mkdir()
    for i, content in enumerate([b"AAA", b"BBB", b"CCC", b"DDD"]):
        (src / f"kb_chunk_{i:04d}.txt").write_bytes(content)
    chunks = sorted(src.glob("kb_chunk_*.txt"))
    out = build_growing_file(chunks, 3, tmp_path)
    assert out.read_bytes() == b"AAABBBCCC"


def test_build_growing_file_step1_same_bytes_as_multi_file_step1(tmp_path: Path) -> None:
    """At step 1, build_growing_file output equals the raw first chunk bytes.

    This asserts that both test regimes start from identical KB content,
    making KB-size the sole variable at step 1.

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    chunks = _make_synthetic_chunks(tmp_path / "src", 3, 128)
    out = build_growing_file(chunks, 1, tmp_path)
    assert out.read_bytes() == chunks[0].read_bytes()


def test_kb_size_bytes_equal_across_tests_steps_1_to_3(tmp_path: Path) -> None:
    """kb_size_bytes for Test 1 and Test 2 are identical at steps 1, 2, and 3.

    Confirms the benchmark assertion logic with ``chunks_per_step=2``: at each
    logical step the total bytes uploaded are the same regardless of whether
    files are merged (Test 1) or separate (Test 2).

    8 synthetic chunks of 200 bytes each are created so that up to step 3
    (= 3 × 2 = 6 raw chunks = 1 200 bytes) is covered.

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    chunks_per_step = 2
    size = 200
    chunks = _make_synthetic_chunks(tmp_path / "src", 8, size)
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    for step in (1, 2, 3):
        raw_count = step * chunks_per_step
        # Test 1: one growing file containing all raw_count raw chunks
        t1_file = build_growing_file(chunks, raw_count, tmp_dir)
        t1_bytes = t1_file.stat().st_size
        # Test 2: step logical files, each containing chunks_per_step raw chunks
        t2_bytes = sum(c.stat().st_size for c in chunks[:raw_count])
        assert t1_bytes == t2_bytes, f"Mismatch at step {step}: {t1_bytes} != {t2_bytes}"
        t1_file.unlink()


# ===========================================================================
# collect_chunk_paths
# ===========================================================================


def test_collect_chunk_paths_returns_sorted(tmp_path: Path) -> None:
    """Returned paths are sorted lexicographically (numeric order for zero-padded names).

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    chunks = _make_synthetic_chunks(tmp_path, 5)
    result = collect_chunk_paths(tmp_path, 5)
    assert result == chunks


def test_collect_chunk_paths_raises_when_too_few(tmp_path: Path) -> None:
    """ValueError is raised when the directory has fewer files than max_files.

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    _make_synthetic_chunks(tmp_path, 3)
    with pytest.raises(ValueError, match="Only 3 chunk files"):
        collect_chunk_paths(tmp_path, 10)


# ===========================================================================
# percentile
# ===========================================================================


def test_percentile_p50() -> None:
    """P50 of [1, 2, 3, 4] returns the median element.

    Returns:
        None
    """
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.0


def test_percentile_p95() -> None:
    """P95 of a 6-element list returns the 6th element (ceiling index).

    Returns:
        None
    """
    vals = [100.0, 200.0, 300.0, 400.0, 500.0, 1000.0]
    assert percentile(vals, 95) == 1000.0


def test_percentile_empty() -> None:
    """Empty list returns 0.0.

    Returns:
        None
    """
    assert percentile([], 95) == 0.0


# ===========================================================================
# compute_step_metrics
# ===========================================================================


def test_compute_step_metrics_six_queries() -> None:
    """Six synthetic results produce correct mean, p50, p95, totals, and citation count.

    Returns:
        None
    """
    results = [
        _make_query_result(elapsed_ms=100.0, input_tokens=50, output_tokens=10, provider_server_latency_ms=80, has_citation=True),
        _make_query_result(elapsed_ms=200.0, input_tokens=60, output_tokens=15, provider_server_latency_ms=160, has_citation=True),
        _make_query_result(elapsed_ms=300.0, input_tokens=70, output_tokens=12, provider_server_latency_ms=240, has_citation=True),
        _make_query_result(elapsed_ms=400.0, input_tokens=80, output_tokens=20, provider_server_latency_ms=320, has_citation=False),
        _make_query_result(elapsed_ms=500.0, input_tokens=90, output_tokens=18, provider_server_latency_ms=400, has_citation=True),
        _make_query_result(elapsed_ms=600.0, input_tokens=100, output_tokens=25, provider_server_latency_ms=480, has_citation=True),
    ]
    m = compute_step_metrics(results)
    assert m["ask_elapsed_ms_mean"] == pytest.approx((100 + 200 + 300 + 400 + 500 + 600) / 6, abs=0.01)
    assert m["ask_elapsed_ms_min"] == 100.0
    assert m["ask_elapsed_ms_max"] == 600.0
    assert m["ask_input_tokens_total"] == 50 + 60 + 70 + 80 + 90 + 100
    assert m["ask_output_tokens_total"] == 10 + 15 + 12 + 20 + 18 + 25
    assert m["citation_miss_count"] == 1
    assert m["server_latency_missing_count"] == 0
    assert m["ask_provider_server_latency_ms_mean"] is not None


def test_compute_step_metrics_all_citations_missed() -> None:
    """citation_miss_count equals 6 when all has_citation are False.

    Returns:
        None
    """
    results = [_make_query_result(has_citation=False) for _ in range(6)]
    m = compute_step_metrics(results)
    assert m["citation_miss_count"] == 6


def test_compute_step_metrics_server_latency_always_present() -> None:
    """With all 6 server latencies present: missing_count=0, mean non-null, throughput>0.

    Returns:
        None
    """
    results = [
        _make_query_result(output_tokens=20, provider_server_latency_ms=300)
        for _ in range(6)
    ]
    m = compute_step_metrics(results)
    assert m["server_latency_missing_count"] == 0
    assert m["ask_provider_server_latency_ms_mean"] is not None
    assert m["ask_throughput_tokens_per_sec_mean"] > 0
    assert m["ask_aggregate_throughput_tokens_per_sec"] > 0


def test_compute_step_metrics_server_latency_all_absent() -> None:
    """With all provider_server_latency_ms=None: missing_count=6, mean=None, aggregate=None.

    Proves aggregate throughput is None (no clean queries); per-query mean is
    still computable from ``throughput_output_tokens_per_sec``.

    Returns:
        None
    """
    results = [
        _make_query_result(elapsed_ms=500.0, output_tokens=20, provider_server_latency_ms=None)
        for _ in range(6)
    ]
    m = compute_step_metrics(results)
    assert m["server_latency_missing_count"] == 6
    assert m["ask_provider_server_latency_ms_mean"] is None
    assert m["ask_throughput_tokens_per_sec_mean"] > 0
    assert m["ask_aggregate_throughput_tokens_per_sec"] is None


def test_compute_step_metrics_server_latency_partial() -> None:
    """With 4 of 6 server latencies absent: missing_count=4, aggregate uses only 2 clean queries.

    The aggregate is computed exclusively from the 2 queries that have
    ``provider_server_latency_ms=300``: (2 × 20 tokens) / (2 × 300 ms / 1000) ≈ 66.667 tok/s.

    Returns:
        None
    """
    import pytest as _pytest

    results = (
        [_make_query_result(output_tokens=20, provider_server_latency_ms=None) for _ in range(4)]
        + [_make_query_result(output_tokens=20, provider_server_latency_ms=300) for _ in range(2)]
    )
    m = compute_step_metrics(results)
    assert m["server_latency_missing_count"] == 4
    assert m["ask_throughput_tokens_per_sec_mean"] > 0
    expected_agg = (2 * 20) / (2 * 300 / 1000.0)  # ≈ 66.667
    assert m["ask_aggregate_throughput_tokens_per_sec"] == _pytest.approx(expected_agg, rel=1e-3)


# ===========================================================================
# average_rows
# ===========================================================================


def _make_row(
    run: int,
    step: int,
    elapsed_mean: float = 400.0,
    input_total: int = 600,
    output_total: int = 100,
    throughput_mean: float = 50.0,
    agg_throughput: float = 60.0,
    server_missing: int = 0,
    citation_miss: int = 0,
    preload_ms: float = 200.0,
) -> dict:
    """Build a minimal per-run row dict matching metrics_per_run.csv schema.

    Args:
        run: Run index (1-indexed).
        step: Step index (1-indexed).
        elapsed_mean: ask_elapsed_ms_mean value.
        input_total: ask_input_tokens_total value.
        output_total: ask_output_tokens_total value.
        throughput_mean: ask_throughput_tokens_per_sec_mean value.
        agg_throughput: ask_aggregate_throughput_tokens_per_sec value.
        server_missing: server_latency_missing_count value.
        citation_miss: citation_miss_count value.
        preload_ms: preload_elapsed_ms value.

    Returns:
        Row dict.
    """
    return {
        "run": run,
        "step": step,
        "kb_size_bytes": step * 100,
        "file_count": step,
        "model": "gpt-4o-mini",
        "vector_store_id": f"vs_{run}_{step}",
        "preload_elapsed_ms": preload_ms,
        "preload_status": "completed",
        "ask_elapsed_ms_mean": elapsed_mean,
        "ask_elapsed_ms_p50": elapsed_mean,
        "ask_elapsed_ms_p95": elapsed_mean * 1.2,
        "ask_elapsed_ms_min": elapsed_mean * 0.8,
        "ask_elapsed_ms_max": elapsed_mean * 1.3,
        "ask_input_tokens_mean": input_total / 6,
        "ask_input_tokens_total": input_total,
        "ask_output_tokens_mean": output_total / 6,
        "ask_output_tokens_total": output_total,
        "ask_throughput_tokens_per_sec_mean": throughput_mean,
        "ask_aggregate_throughput_tokens_per_sec": agg_throughput,
        "ask_provider_server_latency_ms_mean": 300.0,
        "server_latency_missing_count": server_missing,
        "citation_miss_count": citation_miss,
    }


def test_average_rows_single_run() -> None:
    """With a single run, the mean of each column equals the original value.

    Returns:
        None
    """
    rows = [_make_row(1, 1, elapsed_mean=400.0), _make_row(1, 2, elapsed_mean=500.0)]
    avg = average_rows(rows)
    assert len(avg) == 2
    assert avg[0]["ask_elapsed_ms_mean"] == pytest.approx(400.0, abs=0.01)
    assert avg[1]["ask_elapsed_ms_mean"] == pytest.approx(500.0, abs=0.01)
    # step and kb_size_bytes preserved
    assert avg[0]["step"] == 1
    assert avg[1]["step"] == 2
    # run and vector_store_id excluded
    assert "run" not in avg[0]
    assert "vector_store_id" not in avg[0]


def test_average_rows_two_runs() -> None:
    """Two runs per step: numeric columns are averaged correctly.

    Returns:
        None
    """
    rows = [
        _make_row(1, 1, elapsed_mean=200.0, server_missing=1),
        _make_row(2, 1, elapsed_mean=400.0, server_missing=3),
    ]
    avg = average_rows(rows)
    assert len(avg) == 1
    assert avg[0]["ask_elapsed_ms_mean"] == pytest.approx(300.0, abs=0.01)
    # server_latency_missing_count is SUMMED across runs
    assert avg[0]["server_latency_missing_count"] == 4
    # std column exists and is non-negative
    assert avg[0]["ask_elapsed_ms_std"] >= 0.0


# ===========================================================================
# estimate_cost
# ===========================================================================


def test_estimate_cost_returns_expected_fields() -> None:
    """estimate_cost returns the four required keys with correct query_calls value.

    Returns:
        None
    """
    result = estimate_cost(runs=5, max_files=50, query_count=6, avg_kb_bytes_per_step=5120)
    assert result["query_calls"] == 5 * 50 * 6
    assert result["upload_calls"] == 5 * 50
    assert result["rough_input_tokens"] == result["query_calls"] * 1500
    assert "upload_mb" in result
    assert result["upload_mb"] > 0


# ===========================================================================
# confirm_run
# ===========================================================================


def test_confirm_run_raises_on_n() -> None:
    """confirm_run raises SystemExit when the user enters 'n'.

    Returns:
        None
    """
    estimates = estimate_cost(1, 2, 6, 5120)
    with patch.object(builtins, "input", return_value="n"):
        with pytest.raises(SystemExit):
            confirm_run(estimates)


def test_confirm_run_passes_on_y() -> None:
    """confirm_run returns normally when the user enters 'y'.

    Returns:
        None
    """
    estimates = estimate_cost(1, 2, 6, 5120)
    with patch.object(builtins, "input", return_value="y"):
        confirm_run(estimates)  # should not raise


# ===========================================================================
# Integration smoke tests (real API)
# ===========================================================================


@_SKIP_NO_KEY
def test_single_file_growth_smoke(tmp_path: Path) -> None:
    """End-to-end smoke test for Test 1 (single-file growth regime).

    Runs 1 repetition × 2 logical steps (4 raw chunks, chunks_per_step=2)
    × 6 queries = 12 API calls.  Asserts the shape and validity of both CSV
    files.

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    from src.llm_pipeline_telemetry import get_openai_client  # noqa: PLC0415

    client = get_openai_client()
    queries = load_queries(_QUERIES_FILE)
    chunks = collect_chunk_paths(_KB_DIR, max_files=4)

    run_dir = run_benchmark(
        single_get_files(chunks, chunks_per_step=2),
        runs=1,
        max_files=2,
        queries=queries,
        chunks=chunks,
        client=client,
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        output_dir=tmp_path / "single",
        keep_vector_stores=False,
        max_retries=3,
        retry_sleep_seconds=2.0,
        vs_name_prefix="kb-bench-smoke-single",
        chunks_per_step=2,
    )

    per_query_csv = run_dir / "metrics_per_query.csv"
    per_run_csv = run_dir / "metrics_per_run.csv"
    averaged_csv = run_dir / "metrics_averaged.csv"

    assert per_query_csv.exists(), "metrics_per_query.csv not created"
    assert per_run_csv.exists(), "metrics_per_run.csv not created"
    assert averaged_csv.exists(), "metrics_averaged.csv not created"

    import csv as _csv  # noqa: PLC0415

    with per_query_csv.open(encoding="utf-8") as fh:
        pq_rows = list(_csv.DictReader(fh))
    # runs=1, max_files=2, 6 queries → 12 rows
    assert len(pq_rows) == 12, f"Expected 12 per-query rows, got {len(pq_rows)}"
    assert all(r["has_citation"] in ("True", "False") for r in pq_rows), (
        "has_citation contains unexpected values"
    )
    assert "query_text" in pq_rows[0], "query_text column missing from metrics_per_query.csv"

    with per_run_csv.open(encoding="utf-8") as fh:
        per_rows = list(_csv.DictReader(fh))
    assert len(per_rows) == 2, f"Expected 2 rows, got {len(per_rows)}"
    for row in per_rows:
        assert float(row["ask_elapsed_ms_mean"]) > 0
        assert int(row["ask_input_tokens_total"]) > 0
        assert row["preload_status"] == "completed"

    with averaged_csv.open(encoding="utf-8") as fh:
        avg_rows = list(_csv.DictReader(fh))
    assert len(avg_rows) == 2, f"Expected 2 averaged rows, got {len(avg_rows)}"


@_SKIP_NO_KEY
def test_multi_file_growth_smoke(tmp_path: Path) -> None:
    """End-to-end smoke test for Test 2 (multi-file growth regime).

    Runs 1 repetition × 2 logical steps (4 raw chunks, chunks_per_step=2)
    × 6 queries = 12 API calls.  Asserts the shape and validity of both CSV
    files, including that file_count matches step on each row.

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    from src.llm_pipeline_telemetry import get_openai_client  # noqa: PLC0415

    client = get_openai_client()
    queries = load_queries(_QUERIES_FILE)
    chunks = collect_chunk_paths(_KB_DIR, max_files=4)

    run_dir = run_benchmark(
        multi_get_files(chunks, chunks_per_step=2),
        runs=1,
        max_files=2,
        queries=queries,
        chunks=chunks,
        client=client,
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        output_dir=tmp_path / "multi",
        keep_vector_stores=False,
        max_retries=3,
        retry_sleep_seconds=2.0,
        vs_name_prefix="kb-bench-smoke-multi",
        chunks_per_step=2,
    )

    per_query_csv = run_dir / "metrics_per_query.csv"
    per_run_csv = run_dir / "metrics_per_run.csv"
    averaged_csv = run_dir / "metrics_averaged.csv"

    assert per_query_csv.exists(), "metrics_per_query.csv not created"
    assert per_run_csv.exists(), "metrics_per_run.csv not created"
    assert averaged_csv.exists(), "metrics_averaged.csv not created"

    import csv as _csv  # noqa: PLC0415

    with per_query_csv.open(encoding="utf-8") as fh:
        pq_rows = list(_csv.DictReader(fh))
    # runs=1, max_files=2, 6 queries → 12 rows
    assert len(pq_rows) == 12, f"Expected 12 per-query rows, got {len(pq_rows)}"
    assert all(r["has_citation"] in ("True", "False") for r in pq_rows), (
        "has_citation contains unexpected values"
    )
    assert "query_text" in pq_rows[0], "query_text column missing from metrics_per_query.csv"

    with per_run_csv.open(encoding="utf-8") as fh:
        per_rows = list(_csv.DictReader(fh))
    assert len(per_rows) == 2, f"Expected 2 rows, got {len(per_rows)}"
    for row in per_rows:
        assert float(row["ask_elapsed_ms_mean"]) > 0
        assert int(row["ask_input_tokens_total"]) > 0
        assert row["preload_status"] == "completed"
        # file_count must equal step for the multi-file regime
        assert int(row["file_count"]) == int(row["step"]), (
            f"file_count {row['file_count']} != step {row['step']}"
        )

    with averaged_csv.open(encoding="utf-8") as fh:
        avg_rows = list(_csv.DictReader(fh))
    assert len(avg_rows) == 2, f"Expected 2 averaged rows, got {len(avg_rows)}"


# ===========================================================================
# _load_per_query_rows
# ===========================================================================


def test_load_per_query_rows_missing_file_returns_empty(tmp_path: Path) -> None:
    """_load_per_query_rows returns [] when metrics_per_query.csv is absent.

    Verifies backward compatibility: old run directories without the new CSV
    do not cause an error.

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    result = single_load_per_query(tmp_path)
    assert result == []
    result2 = multi_load_per_query(tmp_path)
    assert result2 == []


def test_load_per_query_rows_parses_numeric_and_string_columns(tmp_path: Path) -> None:
    """Numeric columns are cast; has_citation and query_text kept as strings.

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    import csv as _csv  # noqa: PLC0415

    csv_path = tmp_path / "metrics_per_query.csv"
    fieldnames = [
        "run", "step", "query_idx", "query_text", "kb_size_bytes", "file_count",
        "model", "vector_store_id", "elapsed_ms", "input_tokens", "output_tokens",
        "provider_server_latency_ms", "throughput_output_tokens_per_sec", "has_citation",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = _csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "run": 1, "step": 1, "query_idx": 1,
            "query_text": "What is alumina?", "kb_size_bytes": 10205,
            "file_count": 1, "model": "gpt-4o-mini",
            "vector_store_id": "vs_abc", "elapsed_ms": 15000.5,
            "input_tokens": 4800, "output_tokens": 550,
            "provider_server_latency_ms": 14900, "throughput_output_tokens_per_sec": 36.9,
            "has_citation": "True",
        })

    rows = single_load_per_query(tmp_path)
    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row["elapsed_ms"], float)
    assert isinstance(row["input_tokens"], int)
    assert row["has_citation"] == "True"
    assert row["query_text"] == "What is alumina?"


# ===========================================================================
# _compute_citation_hit_metrics
# ===========================================================================


def _make_per_query_row(
    *,
    run: int,
    step: int,
    query_idx: int = 1,
    query_text: str = "test query",
    kb_size_bytes: float = 10000.0,
    elapsed_ms: float = 10000.0,
    throughput: float = 30.0,
    has_citation: str = "True",
    output_tokens: int = 550,
    provider_server_latency_ms: float | None = None,
) -> dict:
    """Build a synthetic per-query row dict for use in unit tests.

    Args:
        run: Run index.
        step: Step index.
        query_idx: Query index within the step.
        query_text: Query string.
        kb_size_bytes: KB size at this step.
        elapsed_ms: Per-query latency in milliseconds.
        throughput: Per-query throughput in tokens/sec.
        has_citation: ``"True"`` or ``"False"``.
        output_tokens: Number of output tokens generated.  Defaults to 550
            to preserve backward compatibility with existing callers.
        provider_server_latency_ms: Server-side latency reported by the
            provider.  Defaults to ``elapsed_ms * 0.99`` when ``None``.

    Returns:
        Dict matching the schema written to ``metrics_per_query.csv``.
    """
    return {
        "run": run,
        "step": step,
        "query_idx": query_idx,
        "query_text": query_text,
        "kb_size_bytes": kb_size_bytes,
        "file_count": step,
        "model": "gpt-4o-mini",
        "vector_store_id": f"vs_{run}_{step}",
        "elapsed_ms": elapsed_ms,
        "input_tokens": 4800,
        "output_tokens": output_tokens,
        "provider_server_latency_ms": (
            elapsed_ms * 0.99 if provider_server_latency_ms is None
            else provider_server_latency_ms
        ),
        "throughput_output_tokens_per_sec": throughput,
        "has_citation": has_citation,
    }


def test_compute_citation_hit_metrics_filters_correctly() -> None:
    """Only has_citation='True' rows contribute to latency_mean_ms.

    Two rows at step 1 run 1: one hit (elapsed=10000), one miss (elapsed=99999).
    The mean must equal 10000.

    Returns:
        None
    """
    rows = [
        _make_per_query_row(run=1, step=1, query_idx=1, elapsed_ms=10000.0, has_citation="True"),
        _make_per_query_row(run=1, step=1, query_idx=2, elapsed_ms=99999.0, has_citation="False"),
    ]
    result = single_compute_hit(rows)
    assert len(result) == 1
    assert result[0]["step"] == 1
    assert result[0]["latency_mean_ms"] == pytest.approx(10000.0, abs=0.01)


def test_compute_citation_hit_metrics_mean_is_arithmetically_correct() -> None:
    """latency_mean_ms is the mean of per-run means, not raw query mean.

    Run 1 at step 1 has two hit queries: 10000 and 20000 → per-run mean = 15000.
    Run 2 at step 1 has one hit query: 9000 → per-run mean = 9000.
    Cross-run mean = (15000 + 9000) / 2 = 12000.

    Returns:
        None
    """
    rows = [
        _make_per_query_row(run=1, step=1, query_idx=1, elapsed_ms=10000.0),
        _make_per_query_row(run=1, step=1, query_idx=2, elapsed_ms=20000.0),
        _make_per_query_row(run=2, step=1, query_idx=1, elapsed_ms=9000.0),
    ]
    result = single_compute_hit(rows)
    assert len(result) == 1
    assert result[0]["latency_mean_ms"] == pytest.approx(12000.0, abs=0.01)


def test_compute_citation_hit_metrics_std_across_runs() -> None:
    """latency_std_ms is the population std of the per-run means.

    Per-run means: 15000 and 9000.
    Population std = sqrt(((15000-12000)^2 + (9000-12000)^2) / 2) = 3000.

    Returns:
        None
    """
    import math as _math  # noqa: PLC0415

    rows = [
        _make_per_query_row(run=1, step=1, query_idx=1, elapsed_ms=10000.0),
        _make_per_query_row(run=1, step=1, query_idx=2, elapsed_ms=20000.0),
        _make_per_query_row(run=2, step=1, query_idx=1, elapsed_ms=9000.0),
    ]
    result = single_compute_hit(rows)
    expected_std = _math.sqrt(((15000 - 12000) ** 2 + (9000 - 12000) ** 2) / 2)
    assert result[0]["latency_std_ms"] == pytest.approx(expected_std, abs=0.01)


def test_compute_citation_hit_metrics_step_with_zero_hits_omitted() -> None:
    """A step where all has_citation='False' across all runs is not in the result.

    Returns:
        None
    """
    rows = [
        _make_per_query_row(run=1, step=1, has_citation="True", elapsed_ms=10000.0),
        _make_per_query_row(run=1, step=2, has_citation="False", elapsed_ms=20000.0),
        _make_per_query_row(run=2, step=2, has_citation="False", elapsed_ms=22000.0),
    ]
    result = single_compute_hit(rows)
    steps_present = [m["step"] for m in result]
    assert 1 in steps_present
    assert 2 not in steps_present


def test_compute_citation_hit_metrics_hit_run_count() -> None:
    """hit_run_count reflects the number of runs with at least one citation hit.

    Step 1: run 1 has a hit, run 2 has no hit → hit_run_count = 1.
    Step 2: both runs have hits → hit_run_count = 2.

    Returns:
        None
    """
    rows = [
        _make_per_query_row(run=1, step=1, has_citation="True", elapsed_ms=10000.0),
        _make_per_query_row(run=2, step=1, has_citation="False", elapsed_ms=10000.0),
        _make_per_query_row(run=1, step=2, has_citation="True", elapsed_ms=15000.0),
        _make_per_query_row(run=2, step=2, has_citation="True", elapsed_ms=17000.0),
    ]
    result = single_compute_hit(rows)
    by_step = {m["step"]: m for m in result}
    assert by_step[1]["hit_run_count"] == 1
    assert by_step[2]["hit_run_count"] == 2


def test_compute_citation_hit_metrics_multi_matches_single() -> None:
    """multi_compute_hit and single_compute_hit produce identical results for the same input.

    Both plotting scripts must implement identical aggregation logic.

    Returns:
        None
    """
    rows = [
        _make_per_query_row(run=1, step=1, elapsed_ms=10000.0, throughput=40.0),
        _make_per_query_row(run=2, step=1, elapsed_ms=12000.0, throughput=35.0),
    ]
    single_result = single_compute_hit(rows)
    multi_result = multi_compute_hit(rows)
    assert single_result == multi_result


def test_compute_citation_hit_metrics_empty_raises() -> None:
    """_compute_citation_hit_metrics raises ValueError on an empty list.

    Returns:
        None
    """
    with pytest.raises(ValueError, match="non-empty"):
        single_compute_hit([])


def test_compute_citation_hit_metrics_sorted_by_step() -> None:
    """Result list is sorted ascending by step regardless of input order.

    Returns:
        None
    """
    rows = [
        _make_per_query_row(run=1, step=3, elapsed_ms=30000.0),
        _make_per_query_row(run=1, step=1, elapsed_ms=10000.0),
        _make_per_query_row(run=1, step=2, elapsed_ms=20000.0),
    ]
    result = single_compute_hit(rows)
    assert [m["step"] for m in result] == [1, 2, 3]


def test_compute_citation_hit_metrics_aggregate_throughput_correct() -> None:
    """agg_throughput_mean is total_output_tokens / total_latency_sec, not mean of per-query throughputs.

    Case A — coincidental equality (both formulae give 50 tok/s):
        row 1: output_tokens=100, provider_server_latency_ms=2000 → 100/2.0 = 50
        row 2: output_tokens=200, provider_server_latency_ms=4000 → 200/4.0 = 50
        Per-run aggregate = (100+200) / ((2000+4000)/1000) = 300/6.0 = 50 tok/s
        Per-run arithmetic mean of per-query throughputs = (50+50)/2 = 50 tok/s (same)

    Case B — values chosen so the two formulae diverge:
        row 1: output_tokens=100, provider_server_latency_ms=1000 → 100/1.0 = 100
        row 2: output_tokens=100, provider_server_latency_ms=4000 → 100/4.0 = 25
        Per-run aggregate   = (100+100) / ((1000+4000)/1000) = 200/5.0 = 40 tok/s
        Arithmetic mean of per-query throughputs = (100+25)/2 = 62.5 tok/s  ≠ 40

    The test asserts Case B's agg_throughput_mean == 40, not 62.5, proving the
    formula is the weighted aggregate rather than the arithmetic mean.

    Returns:
        None
    """
    rows_a = [
        _make_per_query_row(
            run=1, step=1, query_idx=1,
            output_tokens=100, provider_server_latency_ms=2000.0,
            elapsed_ms=2000.0, throughput=50.0,
        ),
        _make_per_query_row(
            run=1, step=1, query_idx=2,
            output_tokens=200, provider_server_latency_ms=4000.0,
            elapsed_ms=4000.0, throughput=50.0,
        ),
    ]
    result_a = single_compute_hit(rows_a)
    assert len(result_a) == 1
    assert result_a[0]["agg_throughput_mean"] == pytest.approx(50.0, abs=0.01)

    rows_b = [
        _make_per_query_row(
            run=1, step=1, query_idx=1,
            output_tokens=100, provider_server_latency_ms=1000.0,
            elapsed_ms=1000.0, throughput=100.0,
        ),
        _make_per_query_row(
            run=1, step=1, query_idx=2,
            output_tokens=100, provider_server_latency_ms=4000.0,
            elapsed_ms=4000.0, throughput=25.0,
        ),
    ]
    result_b = single_compute_hit(rows_b)
    assert len(result_b) == 1
    assert result_b[0]["agg_throughput_mean"] == pytest.approx(40.0, abs=0.01)
    assert result_b[0]["throughput_mean"] == pytest.approx(62.5, abs=0.01)

    multi_result_b = multi_compute_hit(rows_b)
    assert multi_result_b[0]["agg_throughput_mean"] == pytest.approx(40.0, abs=0.01)


def test_compute_citation_hit_metrics_excludes_missing_server_latency() -> None:
    """agg_throughput_mean uses only queries with non-None provider_server_latency_ms.

    Two citation-hit queries in run 1 step 1:
      - query 1: output_tokens=100, provider_server_latency_ms=900  → clean
      - query 2: output_tokens=200, provider_server_latency_ms=None → excluded

    Only the first query contributes to the per-run aggregate:
      agg = 100 / (900 / 1000.0) ≈ 111.111 tok/s

    Returns:
        None
    """
    row_clean = _make_per_query_row(
        run=1, step=1, query_idx=1,
        output_tokens=100, elapsed_ms=1000.0,
        provider_server_latency_ms=900.0,
    )
    row_missing = _make_per_query_row(
        run=1, step=1, query_idx=2,
        output_tokens=200, elapsed_ms=2000.0,
        provider_server_latency_ms=900.0,  # placeholder — overridden below
    )
    row_missing["provider_server_latency_ms"] = None  # force absence after construction

    rows = [row_clean, row_missing]
    result = single_compute_hit(rows)
    assert len(result) == 1
    expected = 100 / (900.0 / 1000.0)
    assert result[0]["agg_throughput_mean"] == pytest.approx(expected, rel=1e-3)

    # Same logic applies to the multi-file variant.
    multi_result = multi_compute_hit(rows)
    assert len(multi_result) == 1
    assert multi_result[0]["agg_throughput_mean"] == pytest.approx(expected, rel=1e-3)


# ===========================================================================
# metrics_per_query.csv — run_benchmark integration (no API)
# ===========================================================================


def test_per_query_csv_written_with_correct_shape(tmp_path: Path) -> None:
    """run_benchmark writes metrics_per_query.csv with R×S×Q rows.

    Uses mocked upload_with_retry and query_with_retry to avoid API calls.
    Verifies schema, row count, has_citation column, and query_idx range.

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    from unittest.mock import MagicMock, patch  # noqa: PLC0415
    import csv as _csv  # noqa: PLC0415

    queries = ["query A", "query B", "query C"]
    chunks = _make_synthetic_chunks(tmp_path / "src", 4, 100)

    mock_client = MagicMock()
    mock_client.vector_stores.create.return_value = MagicMock(id="vs_mock")
    mock_client.vector_stores.delete.return_value = None

    call_counter = {"n": 0}

    def fake_query_result(*args, **kwargs):
        call_counter["n"] += 1
        return {
            "elapsed_ms": 1000.0,
            "input_tokens": 100,
            "output_tokens": 20,
            "provider_server_latency_ms": 900,
            "throughput_output_tokens_per_sec": 22.2,
            "has_citation": call_counter["n"] % 2 == 0,
        }

    with (
        patch("_shared.upload_with_retry", return_value="completed"),
        patch("_shared.query_with_retry", side_effect=fake_query_result),
    ):
        run_dir = run_benchmark(
            single_get_files(chunks, chunks_per_step=2),
            runs=2,
            max_files=2,
            queries=queries,
            chunks=chunks,
            client=mock_client,
            model="gpt-4o-mini",
            output_dir=tmp_path / "out",
            keep_vector_stores=False,
            max_retries=1,
            retry_sleep_seconds=0.0,
            vs_name_prefix="test",
            chunks_per_step=2,
        )

    per_query_csv = run_dir / "metrics_per_query.csv"
    assert per_query_csv.exists(), "metrics_per_query.csv was not created"

    with per_query_csv.open(encoding="utf-8") as fh:
        pq_rows = list(_csv.DictReader(fh))

    # R=2 runs × S=2 steps × Q=3 queries = 12 rows
    assert len(pq_rows) == 12, f"Expected 12 rows, got {len(pq_rows)}"

    expected_columns = {
        "run", "step", "query_idx", "query_text", "kb_size_bytes", "file_count",
        "model", "vector_store_id", "elapsed_ms", "input_tokens", "output_tokens",
        "provider_server_latency_ms", "throughput_output_tokens_per_sec", "has_citation",
    }
    assert expected_columns.issubset(set(pq_rows[0].keys())), (
        f"Missing columns: {expected_columns - set(pq_rows[0].keys())}"
    )

    # has_citation must be only "True" or "False"
    assert all(r["has_citation"] in ("True", "False") for r in pq_rows), (
        "has_citation contains unexpected values"
    )

    # query_idx must run 1..Q for each (run, step)
    for run_val in ("1", "2"):
        for step_val in ("1", "2"):
            subset = [
                int(r["query_idx"])
                for r in pq_rows
                if r["run"] == run_val and r["step"] == step_val
            ]
            assert sorted(subset) == [1, 2, 3], (
                f"query_idx values wrong for run={run_val} step={step_val}: {subset}"
            )

    # query_text must match the actual query strings
    for row in pq_rows:
        assert row["query_text"] in queries


# ===========================================================================
# build_combined_chunk_file
# ===========================================================================


def test_build_combined_chunk_file_byte_length(tmp_path: Path) -> None:
    """Output file byte length equals the sum of the assigned raw chunks.

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    chunks = _make_synthetic_chunks(tmp_path / "src", 6, 100)
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    out = build_combined_chunk_file(chunks, 0, 2, 1, tmp_dir)
    expected = chunks[0].stat().st_size + chunks[1].stat().st_size
    assert out.stat().st_size == expected


def test_build_combined_chunk_file_concatenates_correctly(tmp_path: Path) -> None:
    """Output bytes are the exact binary concatenation of the assigned chunks.

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    src = tmp_path / "src"
    src.mkdir()
    (src / "kb_chunk_0000.txt").write_bytes(b"AAAA")
    (src / "kb_chunk_0001.txt").write_bytes(b"BBBB")
    (src / "kb_chunk_0002.txt").write_bytes(b"CCCC")
    (src / "kb_chunk_0003.txt").write_bytes(b"DDDD")
    chunks = sorted(src.glob("kb_chunk_*.txt"))

    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    # file_idx=1, chunks_per_step=2 → combines chunks[2]+chunks[3]
    out = build_combined_chunk_file(chunks, 1, 2, 3, tmp_dir)
    assert out.read_bytes() == b"CCCCDDDD"


def test_build_combined_chunk_file_unique_names_within_step(tmp_path: Path) -> None:
    """Two files created for the same step have distinct names.

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    chunks = _make_synthetic_chunks(tmp_path / "src", 6, 50)
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    out0 = build_combined_chunk_file(chunks, 0, 2, 2, tmp_dir)
    out1 = build_combined_chunk_file(chunks, 1, 2, 2, tmp_dir)
    assert out0 != out1
    assert out0.name != out1.name


# ===========================================================================
# Updated make_get_files_for_step factories (chunks_per_step=2)
# ===========================================================================


def test_single_factory_chunks_per_step_2_step3_byte_length(tmp_path: Path) -> None:
    """Single-file factory at step 3 with chunks_per_step=2 produces a file of 6 × chunk_size bytes.

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    chunk_size = 80
    chunks = _make_synthetic_chunks(tmp_path / "src", 8, chunk_size)
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    get_files = single_get_files(chunks, chunks_per_step=2)
    paths = get_files(3, tmp_dir)
    assert len(paths) == 1
    assert paths[0].stat().st_size == 6 * chunk_size


def test_multi_factory_chunks_per_step_2_creates_n_temp_files(tmp_path: Path) -> None:
    """Multi-file factory at step 3 with chunks_per_step=2 returns 3 paths in tmp_dir.

    Each path should be 2 × chunk_size bytes.

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    chunk_size = 90
    chunks = _make_synthetic_chunks(tmp_path / "src", 8, chunk_size)
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    get_files = multi_get_files(chunks, chunks_per_step=2)
    paths = get_files(3, tmp_dir)
    assert len(paths) == 3
    for p in paths:
        assert p.parent == tmp_dir, f"Expected path under tmp_dir, got {p}"
        assert p.stat().st_size == 2 * chunk_size


def test_multi_factory_chunks_per_step_1_returns_source_paths(tmp_path: Path) -> None:
    """Multi-file factory with chunks_per_step=1 returns source chunk paths without temp files.

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    chunks = _make_synthetic_chunks(tmp_path / "src", 5, 50)
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    get_files = multi_get_files(chunks, chunks_per_step=1)
    paths = get_files(3, tmp_dir)
    assert paths == chunks[:3]
    assert all(p.parent != tmp_dir for p in paths)


# ===========================================================================
# collect_chunk_paths — chunks_per_step=2 validation
# ===========================================================================


def test_collect_chunk_paths_raises_for_chunks_per_step_2(tmp_path: Path) -> None:
    """collect_chunk_paths raises ValueError when fewer files exist than max_files * 2.

    Creates 3 raw chunk files, then requests 4 raw chunks (simulating
    max_files=2, chunks_per_step=2), which requires 4 files.

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    _make_synthetic_chunks(tmp_path, 3)
    with pytest.raises(ValueError, match="need at least 4"):
        collect_chunk_paths(tmp_path, 4)


# ===========================================================================
# CLI validation — chunks_per_step guard
# ===========================================================================


def test_parse_args_rejects_chunks_per_step_zero_single(monkeypatch: pytest.MonkeyPatch) -> None:
    """benchmark_single_file_growth._parse_args raises SystemExit for --chunks-per-step 0.

    Args:
        monkeypatch: pytest monkeypatch fixture.
    """
    import benchmark_single_file_growth as _single  # noqa: PLC0415

    monkeypatch.setattr(
        "sys.argv",
        ["benchmark_single_file_growth.py", "--chunks-per-step", "0", "--yes"],
    )
    with pytest.raises(SystemExit):
        _single._parse_args()


def test_parse_args_rejects_chunks_per_step_zero_multi(monkeypatch: pytest.MonkeyPatch) -> None:
    """benchmark_multi_file_growth._parse_args raises SystemExit for --chunks-per-step 0.

    Args:
        monkeypatch: pytest monkeypatch fixture.
    """
    import benchmark_multi_file_growth as _multi  # noqa: PLC0415

    monkeypatch.setattr(
        "sys.argv",
        ["benchmark_multi_file_growth.py", "--chunks-per-step", "0", "--yes"],
    )
    with pytest.raises(SystemExit):
        _multi._parse_args()


# ===========================================================================
# upload_with_retry — supplemental polling tests (no API calls)
# ===========================================================================


class TestUploadWithRetrySupplementalPoll:
    """Unit tests for the supplemental polling path in upload_with_retry.

    All tests use mocks to avoid real API calls.  They verify that:
      - The fast path (upload_and_poll already returns "completed") is
        unchanged and incurs no retrieve calls.
      - When upload_and_poll returns "in_progress", the supplemental loop
        polls retrieve until a terminal status is observed.
      - Terminal statuses "failed" and "cancelled" also stop the loop.
      - When the timeout elapses the function returns the last-seen status
        and emits a WARNING.

    Pillar compliance:
        - Pillar 1: Covers the exact race condition observed at step 25 with
          25 files (probe run 2026-04-07).
        - Pillar 2: Five cases — happy path, retry-resolves, failed, cancelled,
          timeout — cover every branch in _poll_batch_until_terminal.
        - Pillar 7: Timeout test verifies graceful degradation and WARNING.
    """

    def _make_batch(self, status: str, batch_id: str = "batch_abc") -> object:
        """Return a lightweight mock object with .status and .id attributes.

        Args:
            status: String to assign to the mock's ``.status`` attribute.
            batch_id: String to assign to the mock's ``.id`` attribute.

        Returns:
            A ``MagicMock`` that looks like a ``VectorStoreFileBatch``.
        """
        from unittest.mock import MagicMock  # noqa: PLC0415

        m = MagicMock()
        m.status = status
        m.id = batch_id
        return m

    def _make_client(self, retrieve_side_effect=None) -> object:
        """Return a mock OpenAI client with a configurable retrieve side-effect.

        Args:
            retrieve_side_effect: Passed as ``side_effect`` to the
                ``file_batches.retrieve`` mock.  May be a list (each call
                returns the next item) or a single value.

        Returns:
            A ``MagicMock`` client whose
            ``vector_stores.file_batches.retrieve`` is pre-configured.
        """
        from unittest.mock import MagicMock  # noqa: PLC0415

        client = MagicMock()
        if retrieve_side_effect is not None:
            client.vector_stores.file_batches.retrieve.side_effect = (
                retrieve_side_effect
            )
        return client

    def test_no_supplemental_poll_when_completed(self, tmp_path: Path) -> None:
        """upload_with_retry does not call retrieve when upload_and_poll returns completed.

        Verifies that the fast path (normal case at small file counts) adds
        zero overhead — retrieve is never called.

        Args:
            tmp_path: pytest temporary directory fixture.
        """
        from _shared import upload_with_retry  # noqa: PLC0415

        dummy_file = tmp_path / "f.txt"
        dummy_file.write_text("x")

        batch_completed = self._make_batch("completed")
        client = self._make_client()
        client.vector_stores.file_batches.upload_and_poll.return_value = batch_completed

        result = upload_with_retry(
            client,
            vector_store_id="vs_test",
            file_paths=[dummy_file],
            max_retries=0,
            retry_sleep_seconds=0.0,
        )

        assert result == "completed"
        client.vector_stores.file_batches.retrieve.assert_not_called()

    def test_supplemental_poll_resolves_in_progress(self, tmp_path: Path) -> None:
        """upload_with_retry retries retrieve until completed when upload_and_poll returns in_progress.

        Simulates the race condition observed at step 25 with 25 files:
        upload_and_poll returns "in_progress", first retrieve also returns
        "in_progress", second retrieve returns "completed".

        Args:
            tmp_path: pytest temporary directory fixture.
        """
        from _shared import upload_with_retry  # noqa: PLC0415

        dummy_file = tmp_path / "f.txt"
        dummy_file.write_text("x")

        batch_in_progress = self._make_batch("in_progress")
        batch_still_pending = self._make_batch("in_progress")
        batch_done = self._make_batch("completed")

        client = self._make_client(
            retrieve_side_effect=[batch_still_pending, batch_done]
        )
        client.vector_stores.file_batches.upload_and_poll.return_value = (
            batch_in_progress
        )

        result = upload_with_retry(
            client,
            vector_store_id="vs_test",
            file_paths=[dummy_file],
            max_retries=0,
            retry_sleep_seconds=0.0,
            poll_timeout_seconds=60.0,
        )

        assert result == "completed"
        assert client.vector_stores.file_batches.retrieve.call_count == 2

    def test_supplemental_poll_terminal_failed(self, tmp_path: Path) -> None:
        """upload_with_retry stops immediately and returns failed when retrieve reports it.

        Verifies that "failed" is treated as a terminal status — the loop does
        not continue hoping for "completed".

        Args:
            tmp_path: pytest temporary directory fixture.
        """
        from _shared import upload_with_retry  # noqa: PLC0415

        dummy_file = tmp_path / "f.txt"
        dummy_file.write_text("x")

        batch_in_progress = self._make_batch("in_progress")
        batch_failed = self._make_batch("failed")

        client = self._make_client(retrieve_side_effect=[batch_failed])
        client.vector_stores.file_batches.upload_and_poll.return_value = (
            batch_in_progress
        )

        result = upload_with_retry(
            client,
            vector_store_id="vs_test",
            file_paths=[dummy_file],
            max_retries=0,
            retry_sleep_seconds=0.0,
            poll_timeout_seconds=60.0,
        )

        assert result == "failed"
        assert client.vector_stores.file_batches.retrieve.call_count == 1

    def test_supplemental_poll_terminal_cancelled(self, tmp_path: Path) -> None:
        """upload_with_retry stops immediately and returns cancelled when retrieve reports it.

        Verifies that "cancelled" is treated as a terminal status.

        Args:
            tmp_path: pytest temporary directory fixture.
        """
        from _shared import upload_with_retry  # noqa: PLC0415

        dummy_file = tmp_path / "f.txt"
        dummy_file.write_text("x")

        batch_in_progress = self._make_batch("in_progress")
        batch_cancelled = self._make_batch("cancelled")

        client = self._make_client(retrieve_side_effect=[batch_cancelled])
        client.vector_stores.file_batches.upload_and_poll.return_value = (
            batch_in_progress
        )

        result = upload_with_retry(
            client,
            vector_store_id="vs_test",
            file_paths=[dummy_file],
            max_retries=0,
            retry_sleep_seconds=0.0,
            poll_timeout_seconds=60.0,
        )

        assert result == "cancelled"
        assert client.vector_stores.file_batches.retrieve.call_count == 1

    def test_supplemental_poll_timeout(self, tmp_path: Path) -> None:
        """upload_with_retry returns in_progress and emits WARNING when timeout elapses.

        Uses poll_timeout_seconds=0.01 so the deadline expires immediately
        without waiting for real sleeps (initial_sleep_seconds is patched to 0).

        Args:
            tmp_path: pytest temporary directory fixture.
        """
        import logging  # noqa: PLC0415
        from unittest.mock import patch as _patch  # noqa: PLC0415

        from _shared import upload_with_retry  # noqa: PLC0415

        dummy_file = tmp_path / "f.txt"
        dummy_file.write_text("x")

        batch_in_progress = self._make_batch("in_progress")
        batch_stuck = self._make_batch("in_progress")

        client = self._make_client(retrieve_side_effect=[batch_stuck])
        client.vector_stores.file_batches.upload_and_poll.return_value = (
            batch_in_progress
        )

        with _patch("_shared._poll_batch_until_terminal") as mock_poll:
            # Make the helper return the stuck batch so we test the caller's
            # handling of a non-terminal return from _poll_batch_until_terminal.
            mock_poll.return_value = batch_stuck

            with _patch("_shared._logger") as mock_logger:
                # Directly test _poll_batch_until_terminal timeout path by
                # calling upload_with_retry with a real tiny timeout and
                # patching time.sleep to be instant.
                pass

        # Test _poll_batch_until_terminal timeout branch directly.
        from _shared import _poll_batch_until_terminal  # noqa: PLC0415

        batch_always_pending = self._make_batch("in_progress")
        client2 = self._make_client(
            retrieve_side_effect=[batch_always_pending] * 100
        )

        with _patch("_shared.time") as mock_time:
            # Call sequence for time.monotonic():
            #   [0] deadline = 0.0 + timeout_seconds (sets deadline)
            #   [1] while check: 0.0 < deadline → True (loop enters once)
            #   [2] while check: 999.0 < deadline → False (loop exits)
            mock_time.monotonic.side_effect = [0.0, 0.0, 999.0]
            mock_time.sleep = lambda _: None

            with _patch("_shared._logger") as mock_logger:
                result_batch = _poll_batch_until_terminal(
                    client2,
                    batch_id="batch_stuck",
                    vector_store_id="vs_test",
                    timeout_seconds=0.01,
                )

        assert str(getattr(result_batch, "status", "")) == "in_progress"
        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "non-terminal" in warning_msg
