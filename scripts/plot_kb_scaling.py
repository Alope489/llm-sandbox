"""Plot KB scaling benchmark metrics and generate a markdown report."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _load_rows(csv_path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "file_count": int(row["file_count"]),
                    "query_count": int(row["query_count"]),
                    "model": row["model"],
                    "local_store_bytes": int(row["local_store_bytes"]),
                    "vector_store_usage_bytes_before": int(row["vector_store_usage_bytes_before"]),
                    "vector_store_usage_bytes_after": int(row["vector_store_usage_bytes_after"]),
                    "preload_elapsed_ms": float(row["preload_elapsed_ms"]),
                    "ask_elapsed_ms_mean": float(row["ask_elapsed_ms_mean"]),
                    "ask_elapsed_ms_p50": float(row["ask_elapsed_ms_p50"]),
                    "ask_elapsed_ms_p95": float(row["ask_elapsed_ms_p95"]),
                    "ask_input_tokens_mean": float(row["ask_input_tokens_mean"]),
                    "ask_input_tokens_total": int(row["ask_input_tokens_total"]),
                    "ask_output_tokens_mean": float(row["ask_output_tokens_mean"]),
                    "ask_output_tokens_total": int(row["ask_output_tokens_total"]),
                }
            )
    return rows


def _plot_xy(path: Path, *, x: list[float | int], y: list[float | int], xlabel: str, ylabel: str, title: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _write_report(output_dir: Path, rows: list[dict[str, float | int | str]]) -> Path:
    first = rows[0]
    last = rows[-1]
    report = output_dir / "report.md"
    report.write_text(
        "\n".join(
            [
                "# KB Scaling Benchmark Report",
                "",
                "## Run config",
                "",
                f"- model: `{first['model']}`",
                f"- query_count_per_step: `{first['query_count']}`",
                f"- step_count: `{len(rows)}`",
                f"- min_files: `{first['file_count']}`",
                f"- max_files: `{last['file_count']}`",
                "",
                "## Artifacts",
                "",
                "- `metrics.csv`",
                "- `latency_vs_files.png`",
                "- `input_tokens_vs_files.png`",
                "- `input_tokens_vs_bytes.png`",
                "- `latency_vs_vector_store_bytes.png`",
                "",
                "## Observations",
                "",
                f"- Mean ask latency changes from `{first['ask_elapsed_ms_mean']}` ms to `{last['ask_elapsed_ms_mean']}` ms across the sweep.",
                f"- Mean ask input tokens change from `{first['ask_input_tokens_mean']}` to `{last['ask_input_tokens_mean']}`.",
                f"- Local KB bytes change from `{first['local_store_bytes']}` to `{last['local_store_bytes']}`.",
                "",
                "## Notes",
                "",
                "- Input token values come from provider usage telemetry and include full context assembled for the request.",
            ]
        ),
        encoding="utf-8",
    )
    return report


def generate_artifacts(csv_path: Path, output_dir: Path) -> dict[str, Path]:
    rows = sorted(_load_rows(csv_path), key=lambda item: int(item["file_count"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    latency_vs_files = output_dir / "latency_vs_files.png"
    input_tokens_vs_files = output_dir / "input_tokens_vs_files.png"
    input_tokens_vs_bytes = output_dir / "input_tokens_vs_bytes.png"
    latency_vs_vector_store_bytes = output_dir / "latency_vs_vector_store_bytes.png"
    _plot_xy(
        latency_vs_files,
        x=[int(row["file_count"]) for row in rows],
        y=[float(row["ask_elapsed_ms_mean"]) for row in rows],
        xlabel="File count",
        ylabel="Mean ask latency (ms)",
        title="KB search latency vs file count",
    )
    _plot_xy(
        input_tokens_vs_files,
        x=[int(row["file_count"]) for row in rows],
        y=[float(row["ask_input_tokens_mean"]) for row in rows],
        xlabel="File count",
        ylabel="Tokens used by LLM (from knowledge base)",
        title="KB input tokens vs file count",
    )
    _plot_xy(
        input_tokens_vs_bytes,
        x=[int(row["local_store_bytes"]) for row in rows],
        y=[float(row["ask_input_tokens_mean"]) for row in rows],
        xlabel="Knowledge base size in bytes",
        ylabel="Tokens used by LLM (from knowledge base)",
        title="KB input tokens vs local KB size",
    )
    _plot_xy(
        latency_vs_vector_store_bytes,
        x=[int(row["vector_store_usage_bytes_after"]) for row in rows],
        y=[float(row["ask_elapsed_ms_mean"]) for row in rows],
        xlabel="Vector store usage bytes after upload",
        ylabel="Mean ask latency (ms)",
        title="KB search latency vs vector store size",
    )
    return {
        "latency_vs_files": latency_vs_files,
        "input_tokens_vs_files": input_tokens_vs_files,
        "input_tokens_vs_bytes": input_tokens_vs_bytes,
        "latency_vs_vector_store_bytes": latency_vs_vector_store_bytes,
        "report": _write_report(output_dir, rows),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots and report for KB scaling benchmark.")
    parser.add_argument("--csv", type=Path, default=Path("sandbox/benchmarks/kb_scaling/metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("sandbox/benchmarks/kb_scaling"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifacts = generate_artifacts(args.csv, args.output_dir)
    for key, value in artifacts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
