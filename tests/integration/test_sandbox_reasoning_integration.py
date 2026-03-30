import json
import os
from pathlib import Path

import pytest

from sandbox.reasoning import summarize


def _load_extractor_records() -> list[dict]:
    parts = [p.strip() for p in Path("sandbox/prompts/reasoning_input.txt").read_text(encoding="utf-8").split("\n---\n") if p.strip()]
    parsed = []
    for p in parts:
        try:
            obj = json.loads(p)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("producer_file") == "extractor.py" and "original_input" in obj and "extraction" in obj:
            parsed.append(obj)
    return parsed


def _load_processor_outputs() -> list[dict]:
    parts = [p.strip() for p in Path("sandbox/prompts/reasoning_input.txt").read_text(encoding="utf-8").split("\n---\n") if p.strip()]
    parsed = []
    for p in parts:
        try:
            obj = json.loads(p)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and (obj.get("producer_file") == "processor.py" or obj.get("producer_agent") == "processor"):
            parsed.append(obj)
    return parsed


@pytest.mark.integration
def test_sandbox_reasoning_real_llm_call_prints_results() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping real LLM integration test")

    extractor_records = _load_extractor_records()
    if not extractor_records:
        pytest.skip("sandbox/prompts/reasoning_input.txt has no extractor records; run extractor integration test first")

    processor_outputs = _load_processor_outputs()
    if not processor_outputs:
        pytest.skip("sandbox/prompts/reasoning_input.txt has no processor outputs; run processor integration test first")

    record = extractor_records[-1]
    original_input = record["original_input"]
    extraction = record["extraction"]

    agent_outputs = processor_outputs[-5:]
    result = summarize(original_input=original_input, extraction=extraction, agent_outputs=agent_outputs)

    print("\n\n=== sandbox.reasoning input ===")
    print(original_input)
    print("\n=== sandbox.reasoning output ===")
    print(json.dumps(result, indent=2, ensure_ascii=True))

    assert isinstance(result.get("summary"), str)
    assert isinstance(result.get("output_prompt"), str)
    assert isinstance(result.get("elapsed_ms"), (int, float))
    assert isinstance(result.get("input_tokens"), int)
    assert isinstance(result.get("output_tokens"), int)

