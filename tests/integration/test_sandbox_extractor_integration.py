import json
import os
from pathlib import Path

import pytest

from sandbox.extractor import extract


def _load_prompts() -> list[str]:
    return [p.strip() for p in Path("sandbox/prompts/extractor.txt").read_text(encoding="utf-8").split("\n---\n") if p.strip()]


@pytest.mark.integration
def test_sandbox_extractor_real_llm_call_prints_results() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping real LLM integration test")

    for i, prompt in enumerate(_load_prompts(), start=1):
        result = extract(prompt)
        print(f"\n\n=== sandbox.extractor prompt {i} ===\n{prompt}\n")
        print("=== output ===")
        print(json.dumps(result, indent=2, ensure_ascii=True))
        assert isinstance(result.get("extraction"), dict)
        assert isinstance(result.get("output_prompt"), str)
        assert isinstance(result.get("elapsed_ms"), (int, float))
        assert isinstance(result.get("input_tokens"), int)
        assert isinstance(result.get("output_tokens"), int)

