from pathlib import Path

from scripts.benchmark_replication_agents import _build_jobs, _parse_replication_factors
from scripts.plot_replication_agents import _rank_agents, _speedup_rows, generate_artifacts


def test_parse_replication_factors_sorts_and_deduplicates() -> None:
    assert _parse_replication_factors("8,2,2,1,4") == [1, 2, 4, 8]


def test_build_jobs_sets_agent_route_hint() -> None:
    jobs = _build_jobs("processor", 3, ["p1", "p2"])
    assert [job["route_hint"] for job in jobs] == ["processor", "processor", "processor"]
    assert [job["payload"] for job in jobs] == ["p1", "p2", "p1"]


def test_speedup_and_rankings_from_rows() -> None:
    speedups = _speedup_rows(
        [
            {"agent": "extractor", "replication_factor": 2, "execution_mode": "linear", "repetition": 1, "elapsed_ms": 100.0, "input_tokens": 1, "output_tokens": 1, "success_count": 2, "error_count": 0, "throughput_output_tokens_per_sec": 10.0},
            {"agent": "extractor", "replication_factor": 2, "execution_mode": "parallel", "repetition": 1, "elapsed_ms": 50.0, "input_tokens": 1, "output_tokens": 1, "success_count": 2, "error_count": 0, "throughput_output_tokens_per_sec": 20.0},
            {"agent": "kb", "replication_factor": 2, "execution_mode": "linear", "repetition": 1, "elapsed_ms": 120.0, "input_tokens": 1, "output_tokens": 1, "success_count": 2, "error_count": 0, "throughput_output_tokens_per_sec": 8.0},
            {"agent": "kb", "replication_factor": 2, "execution_mode": "parallel", "repetition": 1, "elapsed_ms": 100.0, "input_tokens": 1, "output_tokens": 1, "success_count": 2, "error_count": 0, "throughput_output_tokens_per_sec": 9.0},
        ]
    )
    rankings = _rank_agents(speedups)
    assert rankings[0]["agent"] == "extractor"
    assert float(rankings[0]["mean_speedup_factor_gt1"]) > float(rankings[1]["mean_speedup_factor_gt1"])


def test_generate_artifacts_from_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "\n".join(
            [
                "repetition,agent,replication_factor,execution_mode,elapsed_ms,input_tokens,output_tokens,success_count,error_count,throughput_output_tokens_per_sec",
                "1,extractor,1,linear,80.0,100,20,1,0,250.0",
                "1,extractor,1,parallel,70.0,100,20,1,0,285.0",
                "1,extractor,2,linear,150.0,200,30,2,0,200.0",
                "1,extractor,2,parallel,90.0,200,30,2,0,333.0",
                "1,processor,1,linear,90.0,120,22,1,0,244.0",
                "1,processor,1,parallel,88.0,120,22,1,0,250.0",
                "1,processor,2,linear,180.0,240,35,2,0,194.0",
                "1,processor,2,parallel,110.0,240,35,2,0,318.0",
            ]
        ),
        encoding="utf-8",
    )
    artifacts = generate_artifacts(csv_path, tmp_path)
    assert artifacts["speedup_vs_replication"].exists()
    assert artifacts["elapsed_ms_vs_replication_linear_parallel"].exists()
    assert artifacts["throughput_vs_replication"].exists()
    assert artifacts["report"].exists()
