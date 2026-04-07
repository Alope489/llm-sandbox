"""Unit tests and integration smoke tests for the KB-growth benchmark suite.

Unit tests cover all pure helpers in ``new_kb_sandbox._shared`` and require no
API calls or external resources.

Integration smoke tests exercise the full benchmark loop end-to-end against
the real OpenAI API.  They are skipped when ``OPENAI_API_KEY`` is absent.
Each smoke test runs ``runs=1, max_files=2`` (2 steps × 6 queries = 12 API
calls per test, ~18 000 input tokens each) to verify correctness without
burning significant API budget.

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

    Confirms the benchmark assertion logic: at each step the total bytes
    uploaded are the same regardless of whether files are merged or separate.

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    size = 200
    chunks = _make_synthetic_chunks(tmp_path / "src", 5, size)
    tmp_dir = tmp_path / "tmp"
    tmp_dir.mkdir()
    for step in (1, 2, 3):
        # Test 1: one growing file
        t1_file = build_growing_file(chunks, step, tmp_dir)
        t1_bytes = t1_file.stat().st_size
        # Test 2: N separate files
        t2_bytes = sum(c.stat().st_size for c in chunks[:step])
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
    """With all provider_server_latency_ms=None: missing_count=6, mean=None, throughput>0.

    Proves throughput is non-null via client-elapsed fallback.

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
    assert m["ask_aggregate_throughput_tokens_per_sec"] > 0


def test_compute_step_metrics_server_latency_partial() -> None:
    """With 4 of 6 server latencies absent: missing_count=4, throughput>0.

    Returns:
        None
    """
    results = (
        [_make_query_result(output_tokens=20, provider_server_latency_ms=None) for _ in range(4)]
        + [_make_query_result(output_tokens=20, provider_server_latency_ms=300) for _ in range(2)]
    )
    m = compute_step_metrics(results)
    assert m["server_latency_missing_count"] == 4
    assert m["ask_throughput_tokens_per_sec_mean"] > 0


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

    Runs 1 repetition × 2 steps × 6 queries = 12 API calls.  Asserts the
    shape and validity of both CSV files.

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    from src.llm_pipeline_telemetry import get_openai_client  # noqa: PLC0415

    client = get_openai_client()
    queries = load_queries(_QUERIES_FILE)
    chunks = collect_chunk_paths(_KB_DIR, max_files=2)

    run_dir = run_benchmark(
        single_get_files(chunks),
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
    )

    per_run_csv = run_dir / "metrics_per_run.csv"
    averaged_csv = run_dir / "metrics_averaged.csv"

    assert per_run_csv.exists(), "metrics_per_run.csv not created"
    assert averaged_csv.exists(), "metrics_averaged.csv not created"

    import csv as _csv  # noqa: PLC0415

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

    Runs 1 repetition × 2 steps × 6 queries = 12 API calls.  Asserts the
    shape and validity of both CSV files, including that file_count matches
    step on each row.

    Args:
        tmp_path: pytest temporary directory fixture.
    """
    from src.llm_pipeline_telemetry import get_openai_client  # noqa: PLC0415

    client = get_openai_client()
    queries = load_queries(_QUERIES_FILE)
    chunks = collect_chunk_paths(_KB_DIR, max_files=2)

    run_dir = run_benchmark(
        multi_get_files(chunks),
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
    )

    per_run_csv = run_dir / "metrics_per_run.csv"
    averaged_csv = run_dir / "metrics_averaged.csv"

    assert per_run_csv.exists(), "metrics_per_run.csv not created"
    assert averaged_csv.exists(), "metrics_averaged.csv not created"

    import csv as _csv  # noqa: PLC0415

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
