"""Benchmark KB-agent scaling against file-count growth.

Usage:
python scripts/benchmark_kb_scaling.py --min-files 2 --max-files 40 --step 2
"""

import argparse
import csv
import math
import os
import shutil
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_kb_scaling import generate_artifacts


def _load_queries(path: Path) -> list[str]:
    return [chunk.strip() for chunk in path.read_text(encoding="utf-8").split("\n---\n") if chunk.strip()]


def _doc_number(path: Path) -> int:
    return int(path.stem.split("_")[1])


def _collect_documents(max_files: int) -> list[Path]:
    small = sorted((Path("sandbox/kb_data/small")).glob("materials_*.md"), key=_doc_number)
    medium_extra = [
        path
        for path in sorted((Path("sandbox/kb_data/medium")).glob("materials_*.md"), key=_doc_number)
        if _doc_number(path) > len(small)
    ]
    docs = small + medium_extra
    if len(docs) < max_files:
        raise ValueError(f"Not enough documents for max-files={max_files}; only found {len(docs)}")
    return docs[:max_files]


def _build_steps(min_files: int, max_files: int, step: int) -> list[int]:
    steps = list(range(min_files, max_files + 1, step))
    return steps if steps[-1] == max_files else steps + [max_files]


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil((pct / 100.0) * len(ordered)) - 1))]


def _prepare_step_corpus(step_dir: Path, docs: list[Path]) -> list[Path]:
    if step_dir.exists():
        shutil.rmtree(step_dir)
    step_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for path in docs:
        target = step_dir / path.name
        target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        copied.append(target)
    return copied


def _query_vector_store(client: OpenAI, *, model: str, vector_store_id: str, query: str) -> dict[str, float | int]:
    return _query_vector_store_with_retry(
        client,
        model=model,
        vector_store_id=vector_store_id,
        query=query,
        query_index=1,
        query_count=1,
        verbose=False,
        request_timeout_seconds=60.0,
        max_retries=0,
        retry_sleep_seconds=1.0,
    )


def _query_vector_store_with_retry(
    client: OpenAI,
    *,
    model: str,
    vector_store_id: str,
    query: str,
    query_index: int,
    query_count: int,
    verbose: bool,
    request_timeout_seconds: float,
    max_retries: int,
    retry_sleep_seconds: float,
) -> dict[str, float | int]:
    for attempt in range(max_retries + 1):
        if verbose:
            print(
                "[kb-scaling] "
                f"query {query_index}/{query_count} "
                f"attempt {attempt + 1}/{max_retries + 1} start"
            )
        t0 = time.perf_counter()
        try:
            parsed = client.responses.create(
                model=model,
                input=query,
                tools=[{"type": "file_search", "vector_store_ids": [vector_store_id]}],
                timeout=request_timeout_seconds,
            )
            usage = getattr(parsed, "usage", None)
            result = {
                "elapsed_ms": round((time.perf_counter() - t0) * 1000, 3),
                "input_tokens": int(getattr(usage, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "output_tokens", 0) or 0),
            }
            if verbose:
                print(
                    "[kb-scaling] "
                    f"query {query_index}/{query_count} done "
                    f"elapsed_ms={result['elapsed_ms']} input_tokens={result['input_tokens']}"
                )
            return result
        except Exception as exc:
            if verbose:
                print(f"[kb-scaling] query {query_index}/{query_count} failed: {type(exc).__name__}: {exc}")
            if attempt >= max_retries:
                raise
            time.sleep(retry_sleep_seconds * (attempt + 1))
    raise RuntimeError("unreachable retry state")


def _run_step(
    client: OpenAI,
    *,
    file_count: int,
    docs: list[Path],
    queries: list[str],
    model: str,
    keep_vector_store: bool,
    verbose: bool,
    request_timeout_seconds: float,
    max_retries: int,
    retry_sleep_seconds: float,
) -> dict[str, int | float | str]:
    step_dir = Path("sandbox/benchmarks/kb_scaling/work") / f"step_{file_count:03d}"
    step_docs = _prepare_step_corpus(step_dir, docs)
    local_store_bytes = sum(path.stat().st_size for path in step_docs)
    vector_store = client.vector_stores.create(name=f"kb-scaling-{int(time.time())}-{file_count}")
    before = client.vector_stores.retrieve(vector_store.id)
    t0 = time.perf_counter()
    streams = [open(path, "rb") for path in step_docs]
    try:
        batch = client.vector_stores.file_batches.upload_and_poll(vector_store_id=vector_store.id, files=streams)
    finally:
        for stream in streams:
            stream.close()
    preload_elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
    after = client.vector_stores.retrieve(vector_store.id)
    per_query = [
        _query_vector_store_with_retry(
            client,
            model=model,
            vector_store_id=vector_store.id,
            query=query,
            query_index=query_index,
            query_count=len(queries),
            verbose=verbose,
            request_timeout_seconds=request_timeout_seconds,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        for query_index, query in enumerate(queries, start=1)
    ]
    if not keep_vector_store:
        client.vector_stores.delete(vector_store.id)
    elapsed_values = [float(item["elapsed_ms"]) for item in per_query]
    input_values = [int(item["input_tokens"]) for item in per_query]
    output_values = [int(item["output_tokens"]) for item in per_query]
    return {
        "file_count": file_count,
        "query_count": len(queries),
        "model": model,
        "local_store_bytes": local_store_bytes,
        "vector_store_usage_bytes_before": int(getattr(before, "usage_bytes", 0) or 0),
        "vector_store_usage_bytes_after": int(getattr(after, "usage_bytes", 0) or 0),
        "preload_elapsed_ms": preload_elapsed_ms,
        "preload_status": str(getattr(batch, "status", "unknown")),
        "ask_elapsed_ms_mean": round(sum(elapsed_values) / len(elapsed_values), 3),
        "ask_elapsed_ms_p50": round(_percentile(elapsed_values, 50), 3),
        "ask_elapsed_ms_p95": round(_percentile(elapsed_values, 95), 3),
        "ask_input_tokens_mean": round(sum(input_values) / len(input_values), 3),
        "ask_input_tokens_total": sum(input_values),
        "ask_output_tokens_mean": round(sum(output_values) / len(output_values), 3),
        "ask_output_tokens_total": sum(output_values),
    }


def _write_csv(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    if not rows:
        raise ValueError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(
    *,
    min_files: int,
    max_files: int,
    step: int,
    queries_file: Path,
    output_dir: Path,
    model: str,
    keep_vector_stores: bool,
    verbose: bool,
    request_timeout_seconds: float,
    max_retries: int,
    retry_sleep_seconds: float,
) -> Path:
    if min_files < 1 or max_files < 1 or step < 1:
        raise ValueError("min-files, max-files, and step must all be >= 1")
    if min_files > max_files:
        raise ValueError("min-files must be <= max-files")
    queries = _load_queries(queries_file)
    if not queries:
        raise ValueError(f"No queries found in {queries_file}")
    docs = _collect_documents(max_files)
    rows: list[dict[str, int | float | str]] = []
    for idx, file_count in enumerate(_build_steps(min_files, max_files, step), start=1):
        if verbose:
            print(f"[kb-scaling] step {idx}: files={file_count} queries={len(queries)} model={model}")
        row = _run_step(
            OpenAI(),
            file_count=file_count,
            docs=docs[:file_count],
            queries=queries,
            model=model,
            keep_vector_store=keep_vector_stores,
            verbose=verbose,
            request_timeout_seconds=request_timeout_seconds,
            max_retries=max_retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )
        rows.append(row)
        if verbose:
            print(
                "[kb-scaling] "
                f"files={row['file_count']} "
                f"local_bytes={row['local_store_bytes']} "
                f"upload_ms={row['preload_elapsed_ms']} "
                f"ask_mean_ms={row['ask_elapsed_ms_mean']} "
                f"ask_input_tokens_mean={row['ask_input_tokens_mean']}"
            )
    csv_path = output_dir / "metrics.csv"
    _write_csv(csv_path, rows)
    return csv_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KB-agent file-count scaling benchmark.")
    parser.add_argument("--min-files", type=int, default=2)
    parser.add_argument("--max-files", type=int, default=40)
    parser.add_argument("--step", type=int, default=2)
    parser.add_argument("--queries-file", type=Path, default=Path("sandbox/prompts/kb_benchmark_queries.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("sandbox/benchmarks/kb_scaling"))
    parser.add_argument("--model", type=str, default=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"))
    parser.add_argument("--keep-vector-stores", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--request-timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep-seconds", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = _parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required for benchmark execution.")
    csv_path = run_benchmark(
        min_files=args.min_files,
        max_files=args.max_files,
        step=args.step,
        queries_file=args.queries_file,
        output_dir=args.output_dir,
        model=args.model,
        keep_vector_stores=args.keep_vector_stores,
        verbose=not args.quiet,
        request_timeout_seconds=args.request_timeout_seconds,
        max_retries=args.max_retries,
        retry_sleep_seconds=args.retry_sleep_seconds,
    )
    print(f"Wrote metrics CSV: {csv_path}")
    artifacts = generate_artifacts(csv_path, args.output_dir)
    for key, value in artifacts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
