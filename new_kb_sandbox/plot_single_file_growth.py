"""Generate plots and a report from a single-file-growth benchmark run.

Reads ``metrics_averaged.csv`` (and optionally ``metrics_per_query.csv``)
from a timestamped run directory produced by
``benchmark_single_file_growth.py`` and writes PNGs and ``report.md``
into the same directory.

Artifacts always produced::

    latency_vs_kb_size.png      — ask_elapsed_ms_mean vs kb_size_bytes
    input_tokens_vs_kb_size.png — ask_input_tokens_mean vs kb_size_bytes
    throughput_vs_kb_size.png   — ask_aggregate_throughput_tokens_per_sec vs kb_size_bytes
    latency_vs_step.png         — ask_elapsed_ms_mean vs step
    report.md                   — run config, delta metrics, data quality notes

Additional artifacts produced when ``metrics_per_query.csv`` is present::

    latency_vs_kb_size_citation_hits_only.png   — latency filtered to citation-hit queries
    latency_vs_step_citation_hits_only.png       — same, vs step axis
    throughput_vs_kb_size_citation_hits_only.png — throughput filtered to citation-hit queries

Usage::

    python new_kb_sandbox/plot_single_file_growth.py --results-dir new_kb_sandbox/results/single_file/2026-04-06T120000

Dependencies:
    matplotlib, csv, math, pathlib (stdlib).

Pillar compliance:
    - Pillar 1: Plots only the columns produced by the spec-defined CSV schema.
    - Pillar 3: Google-style docstrings; auto-generated report.md documents run.
    - Pillar 4: Fully independent of runner files; no hardcoded paths.
    - Pillar 6: O(S) where S = number of steps; O(R×S×Q) for filtered aggregation.
    - Pillar 7: Raises on missing averaged CSV; gracefully skips filtered plots
      when metrics_per_query.csv is absent (backward-compatible with old runs).
"""
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt

_TEST_LABEL = "Single-File Growth (Test 1)"


def _load_averaged_rows(results_dir: Path) -> list[dict]:
    """Read and parse ``metrics_averaged.csv`` from *results_dir*.

    Args:
        results_dir: Timestamped benchmark run directory containing
            ``metrics_averaged.csv``.

    Returns:
        List of row dicts sorted by ``step`` ascending, with numeric columns
        cast to their native types.

    Raises:
        FileNotFoundError: If ``metrics_averaged.csv`` is absent.
        ValueError: If the file is empty.

    Complexity:
        Θ(S) where S = number of steps.
    """
    csv_path = results_dir / "metrics_averaged.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"metrics_averaged.csv not found in {results_dir}")

    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            row: dict = {}
            for key, val in raw.items():
                if val == "" or val is None:
                    row[key] = None
                else:
                    try:
                        row[key] = int(val) if "." not in val else float(val)
                    except (ValueError, TypeError):
                        row[key] = val
            rows.append(row)

    if not rows:
        raise ValueError(f"metrics_averaged.csv in {results_dir} is empty")
    return sorted(rows, key=lambda r: int(r["step"]))


def _load_per_query_rows(results_dir: Path) -> list[dict]:
    """Read and parse ``metrics_per_query.csv`` from *results_dir*.

    Returns an empty list without raising if the file is absent, enabling
    backward compatibility with run directories produced before the per-query
    CSV was introduced.

    Args:
        results_dir: Timestamped benchmark run directory that may contain
            ``metrics_per_query.csv``.

    Returns:
        List of row dicts with numeric columns cast to their native types and
        ``has_citation`` kept as a string (``"True"`` or ``"False"`` as
        written by the csv module).  Returns ``[]`` if the file does not
        exist.

    Raises:
        ValueError: If the file exists but is empty.

    Complexity:
        Θ(R × S × Q) where R = runs, S = steps, Q = queries per step.
    """
    csv_path = results_dir / "metrics_per_query.csv"
    if not csv_path.exists():
        return []

    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        for raw in csv.DictReader(fh):
            row: dict = {}
            for key, val in raw.items():
                if key in ("has_citation", "query_text", "model", "preload_status",
                           "vector_store_id"):
                    row[key] = val
                elif val == "" or val is None:
                    row[key] = None
                else:
                    try:
                        row[key] = int(val) if "." not in val else float(val)
                    except (ValueError, TypeError):
                        row[key] = val
            rows.append(row)

    if not rows:
        raise ValueError(f"metrics_per_query.csv in {results_dir} is empty")
    return rows


def _compute_citation_hit_metrics(per_query_rows: list[dict]) -> list[dict]:
    """Aggregate per-query rows into per-step citation-hit metrics with std.

    Uses a two-stage grouping that matches the methodology of
    ``average_rows`` in ``_shared.py``:

    * Stage 1 — group by ``(step, run)``: for each run at each step, keep
      only rows where ``has_citation == "True"`` and compute their mean
      ``elapsed_ms``, mean ``throughput_output_tokens_per_sec``, and per-run
      aggregate throughput from ``output_tokens`` and
      ``provider_server_latency_ms`` (queries missing this field are excluded
      from the aggregate; ``elapsed_ms`` is never used as a fallback).
      This yields at most one value per (step, run).
    * Stage 2 — group by ``step``: compute the overall mean and population
      std across the per-run means / aggregates from Stage 1.

    Steps where no run has a single citation hit are omitted from the
    result entirely.

    Args:
        per_query_rows: List of row dicts as loaded by
            ``_load_per_query_rows``.  Each dict must contain ``step``
            (numeric), ``run`` (numeric), ``kb_size_bytes`` (numeric),
            ``elapsed_ms`` (float), ``throughput_output_tokens_per_sec``
            (float), ``output_tokens`` (float), and ``has_citation``
            (str ``"True"`` / ``"False"``).  Optionally contains
            ``provider_server_latency_ms`` (float); queries where this field
            is absent or ``None`` are excluded from the aggregate throughput
            calculation (never falls back to ``elapsed_ms``).

    Returns:
        List of dicts sorted by ``step`` ascending.  Each dict contains:

        * ``step`` (int)
        * ``kb_size_bytes`` (float)
        * ``latency_mean_ms`` (float) — mean of per-run citation-hit means
        * ``latency_std_ms`` (float) — population std across per-run means
        * ``throughput_mean`` (float) — arithmetic mean of per-run mean throughputs
        * ``throughput_std`` (float)
        * ``agg_throughput_mean`` (float) — mean of per-run aggregate throughputs
          (total_output_tokens / total_server_latency_sec), using only queries
          with non-null ``provider_server_latency_ms``; omitted when no run
          contributes any clean queries at this step
        * ``agg_throughput_std`` (float) — population std across per-run aggregates
        * ``hit_run_count`` (int) — number of runs with ≥1 citation hit

    Raises:
        ValueError: If ``per_query_rows`` is empty.

    Complexity:
        Θ(R × S × Q) for the grouping pass; Θ(R × S) for the aggregation.
    """
    if not per_query_rows:
        raise ValueError("per_query_rows must be non-empty")

    # Stage 1: per-(step, run) means of citation-hit queries.
    # key: (step, run) → list of (elapsed_ms, throughput, output_tokens, server_lat_ms_or_None)
    stage1: dict[tuple, list[tuple[float, float, float, Optional[float]]]] = defaultdict(list)
    kb_by_step: dict[int, float] = {}
    for row in per_query_rows:
        if row.get("has_citation") != "True":
            continue
        step = int(row["step"])
        run = int(row["run"])
        kb_by_step[step] = float(row["kb_size_bytes"])
        server_lat_raw = row.get("provider_server_latency_ms")
        server_lat: Optional[float] = float(server_lat_raw) if server_lat_raw is not None else None
        stage1[(step, run)].append((
            float(row["elapsed_ms"]),
            float(row["throughput_output_tokens_per_sec"]),
            float(row["output_tokens"]),
            server_lat,
        ))

    # Stage 2: collapse per-run means into per-step mean + population std.
    # key: step → list of per-run means / aggregates
    stage2_lat: dict[int, list[float]] = defaultdict(list)
    stage2_thr: dict[int, list[float]] = defaultdict(list)
    stage2_agg: dict[int, list[float]] = defaultdict(list)
    for (step, _run), vals in stage1.items():
        n = len(vals)
        stage2_lat[step].append(sum(v[0] for v in vals) / n)
        stage2_thr[step].append(sum(v[1] for v in vals) / n)
        # Aggregate throughput: exclude queries missing provider_server_latency_ms.
        clean = [(v[2], v[3]) for v in vals if v[3] is not None]
        if clean:
            total_out = sum(p[0] for p in clean)
            total_lat_ms = sum(p[1] for p in clean)
            per_run_agg = (total_out / (total_lat_ms / 1000.0)) if total_lat_ms > 0 else 0.0
            stage2_agg[step].append(per_run_agg)

    def _pop_std(values: list[float]) -> float:
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        return math.sqrt(sum((v - mean) ** 2 for v in values) / n)

    result: list[dict] = []
    for step in sorted(stage2_lat.keys()):
        lat_vals = stage2_lat[step]
        thr_vals = stage2_thr[step]
        agg_vals = stage2_agg[step]
        result.append({
            "step": step,
            "kb_size_bytes": kb_by_step[step],
            "latency_mean_ms": round(sum(lat_vals) / len(lat_vals), 3),
            "latency_std_ms": round(_pop_std(lat_vals), 3),
            "throughput_mean": round(sum(thr_vals) / len(thr_vals), 3),
            "throughput_std": round(_pop_std(thr_vals), 3),
            "agg_throughput_mean": (
                round(sum(agg_vals) / len(agg_vals), 3) if agg_vals else None
            ),
            "agg_throughput_std": round(_pop_std(agg_vals), 3) if agg_vals else None,
            "hit_run_count": len(lat_vals),
        })
    return result


def _plot_with_errorbars(
    path: Path,
    *,
    x: list,
    y: list,
    y_err: list | None,
    xlabel: str,
    ylabel: str,
    title: str,
) -> None:
    """Save a line-with-markers plot (with optional error bars) to *path*.

    Args:
        path: Output PNG file path.
        x: X-axis values.
        y: Y-axis values.
        y_err: Optional list of ± error values (population std-dev).
              Pass ``None`` or a list of zeros/Nones to skip error bars.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Plot title.

    Complexity:
        O(S) where S = len(x).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    if y_err and any(v is not None and v > 0 for v in y_err):
        ax.errorbar(
            x,
            y,
            yerr=[v if v is not None else 0.0 for v in y_err],
            marker="o",
            capsize=4,
            label="mean ± std",
        )
        ax.legend(fontsize=9)
    else:
        ax.plot(x, y, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(visible=True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_report(
    results_dir: Path,
    rows: list[dict],
    test_label: str,
    *,
    per_query_rows: list[dict] | None = None,
) -> Path:
    """Write a ``report.md`` summarising the benchmark run.

    Includes run config, step range, first-vs-last delta metrics, citation
    miss notes, a data quality warning section for steps with missing server
    latency, a methods note on standard deviation columns, and — when
    *per_query_rows* is provided — a citation-filtered artifacts section
    noting which steps were omitted due to zero citation hits.

    Args:
        results_dir: Directory in which to write ``report.md``.
        rows: Averaged rows list (sorted by step).
        test_label: Human-readable label for this test regime.
        per_query_rows: Optional list of per-query row dicts loaded from
            ``metrics_per_query.csv``.  When provided, the report includes
            a citation-filtered artifacts section.  Pass ``None`` or ``[]``
            to omit that section.

    Returns:
        Path to the written ``report.md``.

    Complexity:
        O(S) where S = len(rows); O(R × S × Q) when per_query_rows provided.
    """
    first = rows[0]
    last = rows[-1]

    def _fmt(val: object, decimals: int = 1) -> str:
        if val is None:
            return "N/A"
        if isinstance(val, float):
            return f"{val:.{decimals}f}"
        return str(val)

    lines = [
        f"# KB Growth Benchmark Report — {test_label}",
        "",
        "## Run config",
        "",
        f"- model: `{first.get('model', 'N/A')}`",
        f"- step_range: steps `{int(first['step'])}` to `{int(last['step'])}`",
        f"- step_count: `{len(rows)}`",
        "",
        "## Delta metrics (first step → last step)",
        "",
        f"| Metric | Step {int(first['step'])} | Step {int(last['step'])} | Delta |",
        "| --- | --- | --- | --- |",
    ]

    def _delta_row(label: str, col: str, decimals: int = 1) -> str:
        v0 = first.get(col)
        v1 = last.get(col)
        if v0 is None or v1 is None:
            return f"| {label} | N/A | N/A | N/A |"
        delta = v1 - v0
        sign = "+" if delta >= 0 else ""
        return (
            f"| {label} | {_fmt(v0, decimals)} | {_fmt(v1, decimals)} "
            f"| {sign}{_fmt(delta, decimals)} |"
        )

    lines += [
        _delta_row("ask_elapsed_ms_mean (ms)", "ask_elapsed_ms_mean"),
        _delta_row("ask_input_tokens_mean", "ask_input_tokens_mean", 0),
        _delta_row("ask_aggregate_throughput (tok/s)", "ask_aggregate_throughput_tokens_per_sec"),
        _delta_row("preload_elapsed_ms", "preload_elapsed_ms"),
        "",
    ]

    # Citation miss count note.
    total_citation_miss = sum(int(r.get("citation_miss_count") or 0) for r in rows)
    lines += [
        "## Citation miss count",
        "",
        f"- Total across all steps: `{total_citation_miss}` (not excluded from the "
        "unfiltered metrics above; responses without file_citation are flagged only).",
        "- Per-query citation data is available in `metrics_per_query.csv` for "
        "post-hoc filtering.",
        "",
    ]

    # Citation-filtered artifacts section (only when per_query_rows provided).
    if per_query_rows:
        hit_metrics = _compute_citation_hit_metrics(per_query_rows)
        hit_steps = {m["step"] for m in hit_metrics}
        all_steps = {int(r["step"]) for r in rows}
        zero_hit_steps = sorted(all_steps - hit_steps)

        lines += [
            "## Citation-filtered artifacts",
            "",
            "The following filtered plots include only queries that returned a "
            "file citation (`has_citation = True`).  Latency and throughput are "
            "re-aggregated using a two-stage mean (per-run mean → cross-run mean) "
            "with population std error bars, matching the methodology of the "
            "unfiltered plots.",
            "",
        ]
        if zero_hit_steps:
            lines += [
                f"- Steps omitted (zero citation hits across all runs): "
                f"{zero_hit_steps}",
                "",
            ]
        else:
            lines += [
                "- All steps had at least one citation hit across runs.",
                "",
            ]
        lines += [
            "Artifacts:",
            "- `latency_vs_kb_size_citation_hits_only.png`",
            "- `latency_vs_step_citation_hits_only.png`",
            "- `throughput_vs_kb_size_citation_hits_only.png`",
            "",
        ]

    # Data quality warning.
    affected_steps = [
        int(r["step"])
        for r in rows
        if (r.get("server_latency_missing_count") or 0) > 0
    ]
    if affected_steps:
        lines += [
            "## ⚠ Data quality warning",
            "",
            "The following steps had one or more queries where the "
            "`openai-processing-ms` response header was absent.  Throughput "
            "values for those queries used the **client-elapsed-ms fallback** "
            "denominator rather than the server-latency denominator.",
            "",
            f"Affected steps: {affected_steps}",
            "",
        ]

    # Methods note.
    r_count = None
    for r in rows:
        sc = r.get("server_latency_missing_count")
        if sc is not None:
            # server_latency_missing_count is summed across runs; can't infer R directly
            break
    lines += [
        "## Methods note",
        "",
        "Standard deviation columns (`ask_*_std`) are computed as population "
        "standard deviation across runs per step.  "
        "Approximate 95% CI: mean ± 2 × std / √R where R is the number of "
        "completed runs.",
        "",
        "## Artifacts",
        "",
        "- `metrics_per_query.csv` — one row per (run, step, query); "
        "includes `query_idx`, `query_text`, `has_citation`",
        "- `metrics_per_run.csv` — one row per (run, step)",
        "- `metrics_averaged.csv` — one row per step, averaged across runs",
        "- `latency_vs_kb_size.png`",
        "- `input_tokens_vs_kb_size.png`",
        "- `throughput_vs_kb_size.png`",
        "- `latency_vs_step.png`",
        "- `latency_vs_kb_size_citation_hits_only.png` "
        "(when `metrics_per_query.csv` present)",
        "- `latency_vs_step_citation_hits_only.png` "
        "(when `metrics_per_query.csv` present)",
        "- `throughput_vs_kb_size_citation_hits_only.png` "
        "(when `metrics_per_query.csv` present)",
    ]

    report_path = results_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def generate_artifacts(results_dir: Path) -> dict[str, Path]:
    """Load CSVs and produce all benchmark artifacts for a single-file run.

    Always writes four PNG plots and ``report.md`` from
    ``metrics_averaged.csv``.  When ``metrics_per_query.csv`` is also
    present, additionally writes three citation-hits-only filtered PNGs.

    Args:
        results_dir: Timestamped benchmark run directory (e.g.
            ``new_kb_sandbox/results/single_file/2026-04-06T120000``).

    Returns:
        A dict mapping artifact names to their ``Path`` objects.  Always
        contains: ``latency_vs_kb_size``, ``input_tokens_vs_kb_size``,
        ``throughput_vs_kb_size``, ``latency_vs_step``, ``report``.
        Contains additionally when filtered plots are generated:
        ``latency_vs_kb_size_citation_hits_only``,
        ``latency_vs_step_citation_hits_only``,
        ``throughput_vs_kb_size_citation_hits_only``.

    Raises:
        FileNotFoundError: If ``metrics_averaged.csv`` is absent.
        ValueError: If ``metrics_averaged.csv`` is empty.

    Complexity:
        O(S) for unfiltered plots; O(R × S × Q) for filtered aggregation.
    """
    rows = _load_averaged_rows(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    kb_sizes = [float(r["kb_size_bytes"]) for r in rows]
    steps = [int(r["step"]) for r in rows]
    latency_mean = [float(r["ask_elapsed_ms_mean"]) for r in rows]
    latency_std = [
        float(r["ask_elapsed_ms_std"]) if r.get("ask_elapsed_ms_std") is not None else 0.0
        for r in rows
    ]
    input_tokens_mean = [float(r["ask_input_tokens_mean"]) for r in rows]
    throughput_rows = [
        r for r in rows if r.get("ask_aggregate_throughput_tokens_per_sec") is not None
    ]
    kb_sizes_thr = [float(r["kb_size_bytes"]) for r in throughput_rows]
    throughput_mean = [float(r["ask_aggregate_throughput_tokens_per_sec"]) for r in throughput_rows]
    throughput_std = [
        float(r["ask_aggregate_throughput_tokens_per_sec_std"])
        if r.get("ask_aggregate_throughput_tokens_per_sec_std") is not None
        else 0.0
        for r in throughput_rows
    ]

    latency_vs_kb = results_dir / "latency_vs_kb_size.png"
    _plot_with_errorbars(
        latency_vs_kb,
        x=kb_sizes,
        y=latency_mean,
        y_err=latency_std,
        xlabel="KB size (bytes)",
        ylabel="Mean ask latency (ms)",
        title=f"Ask latency vs KB size — {_TEST_LABEL}",
    )

    input_tokens_vs_kb = results_dir / "input_tokens_vs_kb_size.png"
    _plot_with_errorbars(
        input_tokens_vs_kb,
        x=kb_sizes,
        y=input_tokens_mean,
        y_err=None,
        xlabel="KB size (bytes)",
        ylabel="Mean input tokens per query",
        title=f"Input tokens vs KB size — {_TEST_LABEL}",
    )

    throughput_vs_kb = results_dir / "throughput_vs_kb_size.png"
    _plot_with_errorbars(
        throughput_vs_kb,
        x=kb_sizes_thr,
        y=throughput_mean,
        y_err=throughput_std,
        xlabel="KB size (bytes)",
        ylabel="Aggregate throughput (tokens/sec)",
        title=f"Aggregate throughput vs KB size — {_TEST_LABEL}",
    )

    latency_vs_step = results_dir / "latency_vs_step.png"
    _plot_with_errorbars(
        latency_vs_step,
        x=steps,
        y=latency_mean,
        y_err=latency_std,
        xlabel="Step (file count equivalent)",
        ylabel="Mean ask latency (ms)",
        title=f"Ask latency vs step — {_TEST_LABEL}",
    )

    # --- Optional citation-hits-only filtered plots ---
    per_query_rows = _load_per_query_rows(results_dir)
    filtered_artifacts: dict[str, Path] = {}
    if per_query_rows:
        hit_metrics = _compute_citation_hit_metrics(per_query_rows)
        if hit_metrics:
            fkb = [m["kb_size_bytes"] for m in hit_metrics]
            fstep = [m["step"] for m in hit_metrics]
            flat_mean = [m["latency_mean_ms"] for m in hit_metrics]
            flat_std = [m["latency_std_ms"] for m in hit_metrics]
            fthr_mean = [m["agg_throughput_mean"] for m in hit_metrics]
            fthr_std = [m["agg_throughput_std"] for m in hit_metrics]

            lat_kb_filt = results_dir / "latency_vs_kb_size_citation_hits_only.png"
            _plot_with_errorbars(
                lat_kb_filt,
                x=fkb,
                y=flat_mean,
                y_err=flat_std,
                xlabel="KB size (bytes)",
                ylabel="Mean ask latency — citation hits (ms)",
                title=f"Ask latency vs KB size — {_TEST_LABEL} — Citation Hits Only",
            )
            filtered_artifacts["latency_vs_kb_size_citation_hits_only"] = lat_kb_filt

            lat_step_filt = results_dir / "latency_vs_step_citation_hits_only.png"
            _plot_with_errorbars(
                lat_step_filt,
                x=fstep,
                y=flat_mean,
                y_err=flat_std,
                xlabel="Step (file count equivalent)",
                ylabel="Mean ask latency — citation hits (ms)",
                title=f"Ask latency vs step — {_TEST_LABEL} — Citation Hits Only",
            )
            filtered_artifacts["latency_vs_step_citation_hits_only"] = lat_step_filt

            clean_hit = [m for m in hit_metrics if m.get("agg_throughput_mean") is not None]
            if clean_hit:
                thr_kb_filt = results_dir / "throughput_vs_kb_size_citation_hits_only.png"
                _plot_with_errorbars(
                    thr_kb_filt,
                    x=[m["kb_size_bytes"] for m in clean_hit],
                    y=[m["agg_throughput_mean"] for m in clean_hit],
                    y_err=[m["agg_throughput_std"] for m in clean_hit],
                    xlabel="KB size (bytes)",
                    ylabel="Aggregate throughput — citation hits (tokens/sec)",
                    title=f"Aggregate throughput vs KB size — {_TEST_LABEL} — Citation Hits Only",
                )
                filtered_artifacts["throughput_vs_kb_size_citation_hits_only"] = thr_kb_filt

    report = _write_report(results_dir, rows, _TEST_LABEL, per_query_rows=per_query_rows)

    return {
        "latency_vs_kb_size": latency_vs_kb,
        "input_tokens_vs_kb_size": input_tokens_vs_kb,
        "throughput_vs_kb_size": throughput_vs_kb,
        "latency_vs_step": latency_vs_step,
        "report": report,
        **filtered_artifacts,
    }


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Returns:
        Populated ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(
        description="Generate plots and report for Test 1 (single-file growth) benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help=(
            "Timestamped run directory containing metrics_averaged.csv "
            "(e.g. new_kb_sandbox/results/single_file/2026-04-06T120000)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: generate all artifacts for a single-file-growth run.

    Returns:
        None
    """
    args = _parse_args()
    artifacts = generate_artifacts(args.results_dir)
    for key, path in artifacts.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
