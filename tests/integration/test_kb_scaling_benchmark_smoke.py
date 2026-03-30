from pathlib import Path

from scripts.benchmark_kb_scaling import _build_steps, _collect_documents
from scripts.plot_kb_scaling import generate_artifacts


def test_build_steps_includes_max_when_not_divisible() -> None:
    assert _build_steps(2, 11, 4) == [2, 6, 10, 11]


def test_collect_documents_reuses_small_then_medium() -> None:
    docs = _collect_documents(10)
    assert len(docs) == 10
    assert docs[0].name == "materials_0001.md"
    assert docs[7].name == "materials_0008.md"
    assert docs[8].name == "materials_0009.md"


def test_generate_artifacts_from_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "\n".join(
            [
                "file_count,query_count,model,local_store_bytes,vector_store_usage_bytes_before,vector_store_usage_bytes_after,preload_elapsed_ms,preload_status,ask_elapsed_ms_mean,ask_elapsed_ms_p50,ask_elapsed_ms_p95,ask_input_tokens_mean,ask_input_tokens_total,ask_output_tokens_mean,ask_output_tokens_total",
                "2,6,gpt-4o-mini,1000,0,1200,150.0,completed,300.0,290.0,330.0,75.0,450,40.0,240",
                "4,6,gpt-4o-mini,2000,0,2400,240.0,completed,380.0,370.0,420.0,88.0,528,45.0,270",
            ]
        ),
        encoding="utf-8",
    )
    artifacts = generate_artifacts(csv_path, tmp_path)
    assert artifacts["latency_vs_files"].exists()
    assert artifacts["input_tokens_vs_files"].exists()
    assert artifacts["input_tokens_vs_bytes"].exists()
    assert artifacts["latency_vs_vector_store_bytes"].exists()
    assert artifacts["report"].exists()
