import json
import time
from pathlib import Path

import pytest

from sandbox import coordinator
from sandbox import executor


def _write_queue(path: Path) -> None:
    path.write_text(
        "\n---\n".join(
            [
                json.dumps({"input_id": "input-1", "payload": "first payload"}),
                json.dumps({"input_id": "input-2", "payload": "second payload"}),
                json.dumps({"input_id": "input-3", "payload": "third payload"}),
            ]
        ),
        encoding="utf-8",
    )


def _write_config(path: Path, execution_mode: str, worker_mode: str = "fixed", max_workers: int = 3) -> None:
    path.write_text(
        json.dumps(
            {
                "execution_mode": execution_mode,
                "worker_mode": worker_mode,
                "max_workers": max_workers,
                "kb_scale": "small",
                "sim_mode": "regular",
                "sim_regular": {
                    "initial_cooling_rate_K_per_min": 15.0,
                    "max_iterations": 1,
                    "duration_hours": 1.0,
                },
                "processor_tasks": ["schema_validation"],
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.integration
def test_sandbox_coordinator_linear_mode_runs_in_input_order_and_asserts_telemetry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.txt"
    config_path = tmp_path / "config.json"
    _write_queue(queue_path)
    _write_config(config_path, "linear")

    def fake_execute_item(input_id: str, payload: str, agent: str, _config: dict) -> dict:
        return {
            "input_id": input_id,
            "payload": payload,
            "agent": agent,
            "status": "success",
            "elapsed_ms": 10.0,
            "input_tokens": 12,
            "output_tokens": 7,
            "result": {"payload": payload},
        }

    def fake_summarize(*, original_input: str, extraction: dict, agent_outputs: list, emit_artifacts: bool = True) -> dict:
        del extraction, agent_outputs, emit_artifacts
        return {
            "summary": f"summary for {original_input}",
            "elapsed_ms": 1.0,
            "input_tokens": 3,
            "output_tokens": 2,
        }

    monkeypatch.setattr(executor, "execute_item", fake_execute_item)
    monkeypatch.setattr(coordinator.reasoning, "summarize", fake_summarize)

    output = coordinator.run(queue_path=str(queue_path), config_path=str(config_path))

    assert output["completion_order_input_ids"] == ["input-1", "input-2", "input-3"]
    assert {item["input_id"] for item in output["per_input_runs"]} == {"input-1", "input-2", "input-3"}
    for item in output["per_input_runs"]:
        assert isinstance(item["input_id"], str)
        assert isinstance(item["elapsed_ms"], (int, float))
        assert isinstance(item["input_tokens"], int)
        assert isinstance(item["output_tokens"], int)
        assert item["status"] in {"success", "error"}
    assert isinstance(output["final_reasoning_response"], str)


@pytest.mark.integration
def test_sandbox_coordinator_parallel_mode_feeds_reasoning_in_completion_order(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    queue_path = tmp_path / "queue.txt"
    config_path = tmp_path / "config.json"
    _write_queue(queue_path)
    _write_config(config_path, "parallel")
    expected_completion_order = ["input-2", "input-3", "input-1"]
    seen_reasoning_inputs: list[str] = []

    def fake_execute_item(input_id: str, payload: str, agent: str, _config: dict) -> dict:
        time.sleep({"input-1": 0.05, "input-2": 0.01, "input-3": 0.03}[input_id])
        return {
            "input_id": input_id,
            "payload": payload,
            "agent": agent,
            "status": "success",
            "elapsed_ms": 10.0,
            "input_tokens": 5,
            "output_tokens": 4,
            "result": {"payload": payload},
        }

    def fake_summarize(*, original_input: str, extraction: dict, agent_outputs: list, emit_artifacts: bool = True) -> dict:
        del extraction, agent_outputs, emit_artifacts
        seen_reasoning_inputs.append(original_input)
        return {"summary": f"summary for {original_input}", "elapsed_ms": 1.0, "input_tokens": 1, "output_tokens": 1}

    monkeypatch.setattr(executor, "execute_item", fake_execute_item)
    monkeypatch.setattr(coordinator.reasoning, "summarize", fake_summarize)

    output = coordinator.run(queue_path=str(queue_path), config_path=str(config_path))

    assert output["completion_order_input_ids"] == expected_completion_order
    assert seen_reasoning_inputs == ["second payload", "third payload", "first payload"]


@pytest.mark.integration
def test_sandbox_executor_kb_scale_from_config_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    preload_calls: list[str] = []

    def fake_preload(scale: str = "small", *, emit_artifacts: bool = True) -> dict:
        del emit_artifacts
        preload_calls.append(scale)
        return {"elapsed_ms": 2.0, "input_tokens": 0, "output_tokens": 0}

    def fake_ask(query: str, *, emit_artifacts: bool = True) -> dict:
        del query, emit_artifacts
        return {"answer": "ok", "elapsed_ms": 4.0, "input_tokens": 11, "output_tokens": 6}

    monkeypatch.setattr(executor, "_KB_STATE", {"ready_scale": None})
    monkeypatch.setattr(executor.kb_agent, "preload", fake_preload)
    monkeypatch.setattr(executor.kb_agent, "ask", fake_ask)

    first = executor.execute_item("input-kb-1", "query 1", "kb", {"kb_scale": "large", "sim_mode": "regular"})
    second = executor.execute_item("input-kb-2", "query 2", "kb", {"kb_scale": "large", "sim_mode": "regular"})

    assert first["status"] == "success"
    assert second["status"] == "success"
    assert preload_calls == ["large"]
