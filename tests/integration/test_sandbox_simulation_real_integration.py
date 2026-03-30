import json
import os

import pytest

from sandbox.sim_agent import run_real_simulation


@pytest.mark.integration
def test_sandbox_simulation_real_mode_prints_results() -> None:
    if os.environ.get("SANDBOX_RUN_REAL_SIM", "0") != "1":
        pytest.skip("Set SANDBOX_RUN_REAL_SIM=1 to run real simulation integration test")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping real LLM integration test")
    output = run_real_simulation(
        os.environ.get(
            "SANDBOX_REAL_SIM_PROMPT",
            "Run real simulation selection for elastic constants with deterministic prefix behavior.",
        )
    )
    print("\n=== sandbox.sim_agent real output ===")
    print(json.dumps(output, indent=2, ensure_ascii=True))
    assert output.get("mode") == "real"
    assert isinstance(output.get("selected_pairs"), list)
    assert isinstance(output.get("simulation_results"), list)
    assert isinstance(output.get("elapsed_ms"), (int, float))
    assert isinstance(output.get("input_tokens"), int)
    assert isinstance(output.get("output_tokens"), int)

