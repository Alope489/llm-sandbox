import json
import os
from pathlib import Path

import pytest

from sandbox.processor import TASKS, process


def _load_extractions() -> list[dict]:
    parts = [p.strip() for p in Path("sandbox/prompts/processor.txt").read_text(encoding="utf-8").split("\n---\n") if p.strip()]
    parsed = []
    for p in parts:
        try:
            obj = json.loads(p)
            parsed.append(obj.get("extraction", obj))
        except json.JSONDecodeError:
            continue
    return parsed


@pytest.mark.integration
def test_sandbox_processor_real_llm_call_prints_results() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping real LLM integration test")

    extractions = _load_extractions()
    if not extractions:
        pytest.skip("sandbox/prompts/processor.txt has no extraction JSON; run extractor integration test first")

    for i, extraction in enumerate(extractions, start=1):
        for task in TASKS:
            result = process(extraction, task)
            print(f"\n\n=== sandbox.processor extraction {i} task {task} ===")
            print("=== output ===")
            print(json.dumps(result, indent=2, ensure_ascii=True))
            assert isinstance(result.get("result"), dict)
            assert isinstance(result.get("output_prompt"), str)
            assert isinstance(result.get("elapsed_ms"), (int, float))
            assert isinstance(result.get("input_tokens"), int)
            assert isinstance(result.get("output_tokens"), int)

