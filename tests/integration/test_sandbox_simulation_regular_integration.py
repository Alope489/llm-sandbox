import json
import os

import pytest

from sandbox.sim_agent import run_regular_simulation


@pytest.mark.integration
def test_sandbox_simulation_regular_real_llm_call_prints_results() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping real LLM integration test")
    output = run_regular_simulation(
        initial_cooling_rate_K_per_min=float(os.environ.get("SANDBOX_SIM_INITIAL_RATE", "15.0")),
        max_iterations=int(os.environ.get("SANDBOX_SIM_ITERATIONS", "3")),
        duration_hours=float(os.environ.get("SANDBOX_SIM_DURATION_HOURS", "4.0")),
    )
    print("\n=== sandbox.sim_agent regular output ===")
    print(json.dumps(output, indent=2, ensure_ascii=True))
    assert output.get("mode") == "regular"
    assert isinstance(output.get("history"), list)
    assert isinstance(output.get("llm_calls"), list)
    assert isinstance(output.get("elapsed_ms"), (int, float))
    assert isinstance(output.get("input_tokens"), int)
    assert isinstance(output.get("output_tokens"), int)

