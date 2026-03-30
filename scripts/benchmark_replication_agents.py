"""Benchmark replication benefit per sandbox agent."""

import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sandbox import coordinator, executor, kb_agent

from plot_replication_agents import generate_artifacts


def _chunks(path: Path) -> list[str]:
    return [chunk.strip() for chunk in path.read_text(encoding="utf-8").split("\n---\n") if chunk.strip()] if path.exists() else []


def _parse_replication_factors(value: str) -> list[int]:
    factors = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not factors or factors[0] < 1:
        raise ValueError("replication-factors must contain positive integers")
    return factors


def _load_seed_payloads() -> dict[str, list[str]]:
    queue_rows = [json.loads(chunk) for chunk in _chunks(Path("sandbox/prompts/coordinator_queue.txt"))]
    extractor_rows = _chunks(Path("sandbox/prompts/extractor.txt"))
    processor_rows = _chunks(Path("sandbox/prompts/processor.txt"))
    kb_rows = _chunks(Path("sandbox/prompts/kb_queries.txt"))
    return {
        "extractor": extractor_rows or [next((row["payload"] for row in queue_rows if row.get("route_hint") == "extractor"), "Extract structure from this task.")],
        "processor": processor_rows or [next((row["payload"] for row in queue_rows if row.get("route_hint") == "processor"), "Validate constraints and rank risk.")],
        "kb": kb_rows or [next((row["payload"] for row in queue_rows if row.get("route_hint") == "kb"), "Summarize this materials topic from the KB.")],
        "simulation": [
            next(
                (row["payload"] for row in queue_rows if str(row.get("route_hint", "")).strip().lower() == "simulation"),
                "Run a simulation plan for elastic constant exploration.",
            )
        ],
    }


def _build_jobs(agent: str, replication_factor: int, payloads: list[str]) -> list[dict[str, str]]:
    return [
        {
            "input_id": f"{agent}-{index + 1:04d}",
            "payload": payloads[index % len(payloads)],
            "source_hint": f"{agent}.txt",
            "route_hint": agent,
        }
        for index in range(replication_factor)
    ]


def _write_queue(path: Path, jobs: list[dict[str, str]]) -> None:
    path.write_text("\n---\n".join(json.dumps(job, ensure_ascii=True) for job in jobs), encoding="utf-8")


def _write_config(path: Path, execution_mode: str, workers: int, kb_scale: str, sim_mode: str) -> None:
    path.write_text(
        json.dumps(
            {
                "execution_mode": execution_mode,
                "worker_mode": "fixed",
                "max_workers": workers,
                "kb_scale": kb_scale,
                "sim_mode": sim_mode,
                "sim_regular": {
                    "initial_cooling_rate_K_per_min": 15.0,
                    "max_iterations": 2,
                    "duration_hours": 1.0,
                },
                "processor_tasks": [
                    "schema_validation",
                    "constraint_verification",
                    "feature_extraction",
                    "normalization",
                    "risk_ranking",
                ],
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )


def _run_once(
    *,
    agent: str,
    replication_factor: int,
    execution_mode: str,
    workers: int,
    kb_scale: str,
    sim_mode: str,
    payloads: list[str],
) -> dict[str, int | float | str]:
    with tempfile.TemporaryDirectory(prefix="replication-benchmark-") as temp_dir:
        queue_path = Path(temp_dir) / "queue.txt"
        config_path = Path(temp_dir) / "config.json"
        _write_queue(queue_path, _build_jobs(agent, replication_factor, payloads))
        _write_config(config_path, execution_mode, workers, kb_scale, sim_mode)
        output = coordinator.run(queue_path=str(queue_path), config_path=str(config_path))
        return {
            "agent": agent,
            "replication_factor": replication_factor,
            "execution_mode": execution_mode,
            "elapsed_ms": float(output["aggregate"]["elapsed_ms"]),
            "input_tokens": int(output["aggregate"]["input_tokens"]),
            "output_tokens": int(output["aggregate"]["output_tokens"]),
            "success_count": sum(1 for item in output["per_input_runs"] if item.get("status") == "success"),
            "error_count": sum(1 for item in output["per_input_runs"] if item.get("status") == "error"),
            "throughput_output_tokens_per_sec": round(
                (1000.0 * float(output["aggregate"]["output_tokens"]) / float(output["aggregate"]["elapsed_ms"]))
                if float(output["aggregate"]["elapsed_ms"]) > 0.0
                else 0.0,
                6,
            ),
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
    agents: list[str],
    replication_factors: list[int],
    repetitions: int,
    workers: int,
    kb_scale: str,
    sim_mode: str,
    output_dir: Path,
    verbose: bool,
) -> Path:
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    if workers < 1:
        raise ValueError("workers must be >= 1")
    payloads = _load_seed_payloads()
    if "kb" in agents:
        kb_agent.preload(scale=kb_scale, emit_artifacts=False)
    rows: list[dict[str, int | float | str]] = []
    for agent in agents:
        for replication_factor in replication_factors:
            for repetition in range(1, repetitions + 1):
                for execution_mode in ("linear", "parallel"):
                    if agent == "kb":
                        setattr(executor, "_KB_STATE", {"ready_scale": kb_scale})
                    row = _run_once(
                        agent=agent,
                        replication_factor=replication_factor,
                        execution_mode=execution_mode,
                        workers=workers,
                        kb_scale=kb_scale,
                        sim_mode=sim_mode,
                        payloads=payloads[agent],
                    )
                    rows.append({"repetition": repetition, **row})
                    if verbose:
                        print(
                            "[replication-benchmark] "
                            f"agent={agent} factor={replication_factor} repetition={repetition} mode={execution_mode} "
                            f"elapsed_ms={row['elapsed_ms']} success={row['success_count']} errors={row['error_count']}"
                        )
    csv_path = output_dir / "metrics.csv"
    _write_csv(csv_path, rows)
    return csv_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark replication benefit per sandbox agent.")
    parser.add_argument("--agents", type=str, default="extractor,processor,kb,simulation")
    parser.add_argument("--replication-factors", type=str, default="1,2,4,8,16")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--kb-scale", type=str, default="small")
    parser.add_argument("--sim-mode", type=str, default="regular")
    parser.add_argument("--output-dir", type=Path, default=Path("sandbox/benchmarks/replication_agents"))
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = _parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required for benchmark execution.")
    csv_path = run_benchmark(
        agents=[item.strip() for item in args.agents.split(",") if item.strip()],
        replication_factors=_parse_replication_factors(args.replication_factors),
        repetitions=args.repetitions,
        workers=args.workers,
        kb_scale=args.kb_scale.strip().lower(),
        sim_mode=args.sim_mode.strip().lower(),
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
    print(f"Wrote metrics CSV: {csv_path}")
    artifacts = generate_artifacts(csv_path, args.output_dir)
    for key, value in artifacts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
