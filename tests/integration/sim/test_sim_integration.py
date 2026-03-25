"""E2E integration tests for the simulation agent. ZERO MOCKING: real LLM only. Runs with both OpenAI and Anthropic."""
import os

import pytest

from src.multi.sim.agent import SimulationAgent, _PREDEFINED_SIM_CALLS

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


# ---------------------------------------------------------------------------
# Real-OpenAI parameterized test: _get_elastic_constants_params_from_LLM
# Covers all 6 remainder classes plus the len=0 boundary (7 cases total).
# Fails (not skips) when OPENAI_API_KEY is absent.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def require_openai_key_sim_integration() -> None:
    """Fail the entire session if OPENAI_API_KEY is not set.

    Raises:
        pytest.fail: If ``OPENAI_API_KEY`` is absent from the environment.

    Pre-conditions:
        Test session is starting.
    Post-conditions:
        All tests in this module may proceed (key is present).
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.fail(
            "OPENAI_API_KEY is not set. Set it in .env or environment before "
            "running this test."
        )


# (prompt_length, expected_n) — all 6 remainder classes + len=0 boundary
_LLM_PARAM_CASES = [
    pytest.param(0,  1, id="len=0_n=1_boundary"),
    pytest.param(6,  1, id="len=6_n=1_remainder0"),
    pytest.param(7,  2, id="len=7_n=2_remainder1"),
    pytest.param(8,  3, id="len=8_n=3_remainder2"),
    pytest.param(9,  4, id="len=9_n=4_remainder3"),
    pytest.param(10, 5, id="len=10_n=5_remainder4"),
    pytest.param(5,  6, id="len=5_n=6_remainder5"),
]


@pytest.mark.parametrize("prompt_length,expected_n", _LLM_PARAM_CASES)
def test_get_elastic_constants_params_from_LLM_real_openai(
    prompt_length: int, expected_n: int
) -> None:
    """_get_elastic_constants_params_from_LLM returns the correct ordered prefix via real OpenAI.

    Constructs a synthetic prompt of ``prompt_length`` characters, calls
    ``_get_elastic_constants_params_from_LLM`` directly, and verifies that the
    LLM returns exactly the first ``expected_n`` entries of ``_PREDEFINED_SIM_CALLS``
    in order.

    Args:
        prompt_length: Length of synthetic prompt ``"x" * prompt_length``.
        expected_n: Expected number of pairs returned, ``prompt_length % 6 + 1``.

    Pre-conditions:
        - ``OPENAI_API_KEY`` is set (enforced by session fixture).
        - ``SimulationAgent(provider="openai")`` is constructable.
    Post-conditions:
        - Return value length == ``expected_n``.
        - Each pair equals ``_PREDEFINED_SIM_CALLS[i]`` (correct ordered prefix).
        - Both elements of every pair are ``str``.

    Complexity:
        One real OpenAI Responses API call per invocation.
    """
    agent = SimulationAgent(provider="openai")
    prompt = "x" * prompt_length

    result = agent._get_elastic_constants_params_from_LLM(prompt)

    assert len(result) == expected_n, (
        f"Expected {expected_n} pairs for prompt length {prompt_length}, got {len(result)}"
    )
    for i, pair in enumerate(result):
        assert pair == _PREDEFINED_SIM_CALLS[i], (
            f"Pair {i} mismatch: expected {_PREDEFINED_SIM_CALLS[i]!r}, got {pair!r}"
        )
        assert isinstance(pair[0], str), f"composition at index {i} must be str"
        assert isinstance(pair[1], str), f"supercell_size at index {i} must be str"
