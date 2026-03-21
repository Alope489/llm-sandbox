"""E2E integration tests for the simulation agent. ZERO MOCKING: real LLM only. Runs with both OpenAI and Anthropic."""
import os

import pytest

from src.multi.sim.agent import SimulationAgent

pytestmark = [pytest.mark.integration, pytest.mark.sim]

PROVIDERS = [
    pytest.param("openai", id="openai", marks=pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")),
    pytest.param("anthropic", id="anthropic", marks=pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")),
]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_sim_integration_run_and_report_real_llm(provider, monkeypatch):
    """Run the simulation agent with the real LLM: real suggestions, real simulation, real output. Runs for OpenAI and Anthropic."""
    monkeypatch.setenv("LLM_PROVIDER", provider)
    agent = SimulationAgent(max_iterations=3)
    history, output = agent.run_and_report(initial_cooling_rate_K_per_min=15.0)
    assert len(history) == 3
    for rate, y_MPa, success in history:
        assert isinstance(rate, (int, float)) and rate >= 0
        assert isinstance(y_MPa, (int, float))
        assert isinstance(success, bool)
    assert "Simulation run:" in output
    assert "Iteration 1:" in output
    assert "Iteration 3:" in output
    assert "Best (successful):" in output or "No successful runs" in output


@pytest.mark.parametrize("provider", PROVIDERS)
def test_sim_integration_optimization_loop_callback_real_llm(provider, monkeypatch):
    """Run optimization loop with on_step callback; real LLM drives suggestions. Runs for OpenAI and Anthropic."""
    monkeypatch.setenv("LLM_PROVIDER", provider)
    seen = []

    def on_step(iteration: int, rate: float, y_MPa: float, success: bool):
        seen.append((iteration, rate, y_MPa, success))

    agent = SimulationAgent(max_iterations=2)
    agent.run_optimization_loop(initial_cooling_rate_K_per_min=15.0, on_step=on_step)
    assert len(seen) == 2
    assert seen[0][0] == 1
    assert seen[1][0] == 2
    assert isinstance(seen[0][1], (int, float))
    assert isinstance(seen[0][2], (int, float))
    assert isinstance(seen[0][3], bool)
