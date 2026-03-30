import json
import os
from pathlib import Path

import pytest

from sandbox.kb_agent import ask, preload


def _load_queries() -> list[str]:
    return [chunk.strip() for chunk in Path("sandbox/prompts/kb_queries.txt").read_text(encoding="utf-8").split("\n---\n") if chunk.strip()]


@pytest.mark.integration
def test_sandbox_kb_agent_real_llm_call_prints_results() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping real LLM integration test")
    scale = os.environ.get("SANDBOX_KB_SCALE", "small")
    preload_output = preload(scale=scale)
    print("\n=== sandbox.kb_agent preload output ===")
    print(json.dumps(preload_output, indent=2, ensure_ascii=True))
    assert isinstance(preload_output.get("local_store_bytes"), int)
    assert isinstance(preload_output.get("vector_store_usage_bytes_after"), int)
    assert isinstance(preload_output.get("elapsed_ms"), (int, float))
    for idx, query in enumerate(_load_queries(), start=1):
        output = ask(query)
        print(f"\n=== sandbox.kb_agent query {idx} ===")
        print(query)
        print("\n=== output ===")
        print(json.dumps(output, indent=2, ensure_ascii=True))
        assert isinstance(output.get("answer"), str)
        assert isinstance(output.get("elapsed_ms"), (int, float))
        assert isinstance(output.get("input_tokens"), int)
        assert isinstance(output.get("output_tokens"), int)
        assert isinstance(output.get("citation_file_ids"), list)

