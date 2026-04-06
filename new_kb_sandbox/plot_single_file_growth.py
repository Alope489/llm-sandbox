"""Generate plots and a report from a single-file-growth benchmark run.

Reads ``metrics_averaged.csv`` from a timestamped run directory produced by
``benchmark_single_file_growth.py`` and writes four PNGs and ``report.md``
into the same directory.

Artifacts produced::

    latency_vs_kb_size.png      — ask_elapsed_ms_mean vs kb_size_bytes
    input_tokens_vs_kb_size.png — ask_input_tokens_mean vs kb_size_bytes
    throughput_vs_kb_size.png   — ask_throughput_tokens_per_sec_mean vs kb_size_bytes
    latency_vs_step.png         — ask_elapsed_ms_mean vs step
    report.md                   — run config, delta metrics, data quality notes

Usage::

    python new_kb_sandbox/plot_single_file_growth.py --results-dir new_kb_sandbox/results/single_file/2026-04-06T120000

Dependencies:
    matplotlib, csv, pathlib (stdlib).

Pillar compliance:
    - Pillar 1: Plots only the columns produced by the spec-defined CSV schema.
    - Pillar 3: Google-style docstrings; auto-generated report.md documents run.
    - Pillar 4: Fully independent of runner files; no hardcoded paths.
    - Pillar 6: O(S) where S = number of steps; entirely local computation.
    - Pillar 7: Raises on missing CSV; warns on partial server-latency coverage.
"""
import argparse
import csv
import math
from pathlib import Path

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
) -> Path:
    """Write a ``report.md`` summarising the benchmark run.

    Includes run config, step range, first-vs-last delta metrics, citation
    miss notes, a data quality warning section for steps with missing server
    latency, and a methods note on standard deviation columns.

    Args:
        results_dir: Directory in which to write ``report.md``.
        rows: Averaged rows list (sorted by step).
        test_label: Human-readable label for this test regime.

    Returns:
        Path to the written ``report.md``.

    Complexity:
        O(S) where S = len(rows).
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
        f"- Total across all steps: `{total_citation_miss}` (not excluded from metrics; "
        "responses without file_citation are flagged only).",
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
        "- `metrics_per_run.csv` — one row per (run, step)",
        "- `metrics_averaged.csv` — one row per step, averaged across runs",
        "- `latency_vs_kb_size.png`",
        "- `input_tokens_vs_kb_size.png`",
        "- `throughput_vs_kb_size.png`",
        "- `latency_vs_step.png`",
    ]

    report_path = results_dir / "report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def generate_artifacts(results_dir: Path) -> dict[str, Path]:
    """Load ``metrics_averaged.csv`` and produce all benchmark artifacts.

    Writes four PNG plots and ``report.md`` into *results_dir*.

    Args:
        results_dir: Timestamped benchmark run directory (e.g.
            ``new_kb_sandbox/results/single_file/2026-04-06T120000``).

    Returns:
        A dict mapping artifact names to their ``Path`` objects:
        ``latency_vs_kb_size``, ``input_tokens_vs_kb_size``,
        ``throughput_vs_kb_size``, ``latency_vs_step``, ``report``.

    Raises:
        FileNotFoundError: If ``metrics_averaged.csv`` is absent.
        ValueError: If the CSV is empty.

    Complexity:
        O(S) where S = number of steps.
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
    throughput_mean = [float(r["ask_throughput_tokens_per_sec_mean"]) for r in rows]
    throughput_std = [
        float(r["ask_throughput_tokens_per_sec_std"])
        if r.get("ask_throughput_tokens_per_sec_std") is not None
        else 0.0
        for r in rows
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
        x=kb_sizes,
        y=throughput_mean,
        y_err=throughput_std,
        xlabel="KB size (bytes)",
        ylabel="Mean throughput (tokens/sec)",
        title=f"Throughput vs KB size — {_TEST_LABEL}",
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

    report = _write_report(results_dir, rows, _TEST_LABEL)

    return {
        "latency_vs_kb_size": latency_vs_kb,
        "input_tokens_vs_kb_size": input_tokens_vs_kb,
        "throughput_vs_kb_size": throughput_vs_kb,
        "latency_vs_step": latency_vs_step,
        "report": report,
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
