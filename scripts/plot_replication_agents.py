"""Plot replication benchmark metrics and generate ranking report."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _load_rows(csv_path: Path) -> list[dict[str, int | float | str]]:
    rows: list[dict[str, int | float | str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "agent": row["agent"],
                    "replication_factor": int(row["replication_factor"]),
                    "execution_mode": row["execution_mode"],
                    "repetition": int(row["repetition"]),
                    "elapsed_ms": float(row["elapsed_ms"]),
                    "input_tokens": int(row["input_tokens"]),
                    "output_tokens": int(row["output_tokens"]),
                    "success_count": int(row["success_count"]),
                    "error_count": int(row["error_count"]),
                    "throughput_output_tokens_per_sec": float(row["throughput_output_tokens_per_sec"]),
                }
            )
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _group_mean(rows: list[dict[str, int | float | str]], metric: str) -> dict[tuple[str, int, str], float]:
    grouped: dict[tuple[str, int, str], list[float]] = {}
    for row in rows:
        key = (str(row["agent"]), int(row["replication_factor"]), str(row["execution_mode"]))
        grouped.setdefault(key, []).append(float(row[metric]))
    return {key: _mean(values) for key, values in grouped.items()}


def _speedup_rows(rows: list[dict[str, int | float | str]]) -> list[dict[str, int | float | str]]:
    elapsed = _group_mean(rows, "elapsed_ms")
    factors = sorted({int(row["replication_factor"]) for row in rows})
    agents = sorted({str(row["agent"]) for row in rows})
    return [
        {
            "agent": agent,
            "replication_factor": factor,
            "speedup": round(
                elapsed[(agent, factor, "linear")] / elapsed[(agent, factor, "parallel")]
                if elapsed.get((agent, factor, "parallel"), 0.0) > 0.0
                else 0.0,
                6,
            ),
        }
        for agent in agents
        for factor in factors
        if (agent, factor, "linear") in elapsed and (agent, factor, "parallel") in elapsed
    ]


def _rank_agents(speedups: list[dict[str, int | float | str]]) -> list[dict[str, int | float | str]]:
    agent_scores: dict[str, list[float]] = {}
    for row in speedups:
        if int(row["replication_factor"]) > 1:
            agent_scores.setdefault(str(row["agent"]), []).append(float(row["speedup"]))
    return sorted(
        [
            {
                "agent": agent,
                "mean_speedup_factor_gt1": round(_mean(values), 6),
                "max_speedup_factor_gt1": round(max(values), 6) if values else 0.0,
            }
            for agent, values in agent_scores.items()
        ],
        key=lambda item: float(item["mean_speedup_factor_gt1"]),
        reverse=True,
    )


def _plot_speedup(path: Path, speedups: list[dict[str, int | float | str]]) -> None:
    plt.figure(figsize=(8, 5))
    factors = sorted({int(row["replication_factor"]) for row in speedups if int(row["replication_factor"]) >= 2})
    for agent in sorted({str(row["agent"]) for row in speedups}):
        filtered = sorted(
            [
                row
                for row in speedups
                if str(row["agent"]) == agent and int(row["replication_factor"]) >= 2
            ],
            key=lambda row: int(row["replication_factor"]),
        )
        if not filtered:
            continue
        plt.plot(
            [int(row["replication_factor"]) for row in filtered],
            [float(row["speedup"]) for row in filtered],
            marker="o",
            label=agent,
        )
    if factors:
        ticks = list(range(min(factors), max(factors) + 1))
        plt.xlim(min(factors) - 0.1, max(factors) + 0.1)
        plt.xticks(ticks, [str(tick) for tick in ticks])
    plt.title("Replication speedup vs replication factor")
    plt.xlabel("Replication factor")
    plt.ylabel("Speedup (linear / parallel)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_elapsed(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    means = _group_mean(rows, "elapsed_ms")
    plt.figure(figsize=(9, 5))
    factors = sorted({int(row["replication_factor"]) for row in rows if int(row["replication_factor"]) >= 2})
    for agent in sorted({str(row["agent"]) for row in rows}):
        filtered = sorted(
            [
                (factor, means[(agent, factor, "parallel")])
                for factor in factors
                if (agent, factor, "parallel") in means
            ],
            key=lambda item: item[0],
        )
        if not filtered:
            continue
        plt.plot([item[0] for item in filtered], [item[1] for item in filtered], marker="o", label=f"{agent}:parallel")
    if factors:
        ticks = list(range(min(factors), max(factors) + 1))
        plt.xlim(min(factors) - 0.1, max(factors) + 0.1)
        plt.xticks(ticks, [str(tick) for tick in ticks])
    plt.title("Elapsed ms vs replication factor")
    plt.xlabel("Replication factor")
    plt.ylabel("Mean elapsed ms")
    plt.grid(True, alpha=0.3)
    plt.legend(ncols=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_throughput(path: Path, rows: list[dict[str, int | float | str]]) -> None:
    means = _group_mean(rows, "throughput_output_tokens_per_sec")
    plt.figure(figsize=(9, 5))
    factors = sorted({int(row["replication_factor"]) for row in rows if int(row["replication_factor"]) >= 2})
    for agent in sorted({str(row["agent"]) for row in rows}):
        filtered = sorted(
            [
                (factor, means[(agent, factor, "parallel")])
                for factor in factors
                if (agent, factor, "parallel") in means
            ],
            key=lambda item: item[0],
        )
        if not filtered:
            continue
        plt.plot([item[0] for item in filtered], [item[1] for item in filtered], marker="o", label=f"{agent}:parallel")
    if factors:
        ticks = list(range(min(factors), max(factors) + 1))
        plt.xlim(min(factors) - 0.1, max(factors) + 0.1)
        plt.xticks(ticks, [str(tick) for tick in ticks])
    plt.title("Output-token throughput vs replication factor")
    plt.xlabel("Replication factor")
    plt.ylabel("Mean output tokens / sec")
    plt.grid(True, alpha=0.3)
    plt.legend(ncols=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _write_report(output_dir: Path, rows: list[dict[str, int | float | str]], rankings: list[dict[str, int | float | str]]) -> Path:
    report = output_dir / "report.md"
    report.write_text(
        "\n".join(
            [
                "# Replication Benefit Benchmark Report",
                "",
                "## Scope",
                "",
                f"- agents: `{', '.join(sorted({str(row['agent']) for row in rows}))}`",
                f"- replication_factors: `{', '.join(str(factor) for factor in sorted({int(row['replication_factor']) for row in rows}))}`",
                f"- repetitions_per_point: `{len({(str(row['agent']), int(row['replication_factor']), str(row['execution_mode']), int(row['repetition'])) for row in rows}) // (len({str(row['agent']) for row in rows}) * len({int(row['replication_factor']) for row in rows}) * 2)}`",
                "",
                "## Ranking by replication benefit",
                "",
                *[
                    f"- `{index}. {item['agent']}` mean_speedup_factor_gt1=`{item['mean_speedup_factor_gt1']}` max_speedup_factor_gt1=`{item['max_speedup_factor_gt1']}`"
                    for index, item in enumerate(rankings, start=1)
                ],
                "",
                "## Artifacts",
                "",
                "- `metrics.csv`",
                "- `speedup_vs_replication.png`",
                "- `elapsed_ms_vs_replication_linear_parallel.png`",
                "- `throughput_vs_replication.png`",
                "",
                "## Notes",
                "",
                "- Speedup is computed as mean linear elapsed divided by mean parallel elapsed for each (agent, replication_factor).",
                "- Ranking score is mean speedup across replication factors greater than 1.",
            ]
        ),
        encoding="utf-8",
    )
    return report


def generate_artifacts(csv_path: Path, output_dir: Path) -> dict[str, Path]:
    rows = _load_rows(csv_path)
    if not rows:
        raise ValueError("CSV has no rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    speedups = _speedup_rows(rows)
    rankings = _rank_agents(speedups)
    speedup_path = output_dir / "speedup_vs_replication.png"
    elapsed_path = output_dir / "elapsed_ms_vs_replication_linear_parallel.png"
    throughput_path = output_dir / "throughput_vs_replication.png"
    _plot_speedup(speedup_path, speedups)
    _plot_elapsed(elapsed_path, rows)
    _plot_throughput(throughput_path, rows)
    return {
        "speedup_vs_replication": speedup_path,
        "elapsed_ms_vs_replication_linear_parallel": elapsed_path,
        "throughput_vs_replication": throughput_path,
        "report": _write_report(output_dir, rows, rankings),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots and report for replication benchmark.")
    parser.add_argument("--csv", type=Path, default=Path("sandbox/benchmarks/replication_agents/metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("sandbox/benchmarks/replication_agents"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    artifacts = generate_artifacts(args.csv, args.output_dir)
    for key, value in artifacts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
