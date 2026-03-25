"""Level 3 integration tests — real Docker + LAMMPS elastic constant calculations.

These tests call ``compute_elastic_constants_tool`` against the live Docker image
``elastic-lammps-tool:latest`` and assert three layers of validation per element:

    Layer 1 — status and schema:
        ``status == "ok"``, C11/C12/C44 are floats, ``runtime_seconds`` is positive.

    Layer 2 — Born stability criteria:
        Physical constraints that must hold for any valid cubic crystal.

    Layer 3 — EAM potential-specific ranges:
        Tight ±5% bounds derived from the Mishin/Mendelev/Zhou potentials used.

All tests fail when the Docker image is not present. Build the image before
running this file::

    docker build -t elastic-lammps-tool:latest src/tools/elastic_constants_lammps/
    py -m pytest tests/test_integration_lammps.py -v

Preconditions:
    - Docker daemon must be running.
    - Image ``elastic-lammps-tool:latest`` must be built.

Postconditions:
    - All assertions hold; test suite exits 0.
"""

import json
import os
import subprocess

import pytest

from src.multi.sim.agent import SimulationAgent, _PREDEFINED_SIM_CALLS
from src.tools.elastic_constants_lammps.host_wrapper import (
    compute_elastic_constants_tool,
)


# ---------------------------------------------------------------------------
# Single source of truth for EAM potential-specific C11/C12/C44 ranges (±5%).
# Both the per-element tests and the end-to-end parametrized test reference
# this dict so range changes only need a single edit.
# ---------------------------------------------------------------------------

_ELEMENT_RANGES: dict[str, tuple] = {
    "Al": ((108, 122), (60, 70), (30, 34)),
    "Cu": ((165, 178), (118, 128), (72, 80)),
    "Ni": ((244, 258), (140, 155), (117, 129)),
    "Fe": ((231, 256), (138, 150), (110, 122)),
    "W": ((516, 536), (195, 208), (152, 168)),
    "Mo": ((451, 474), (162, 172), (107, 119)),
}


# ---------------------------------------------------------------------------
# Docker availability guard
# ---------------------------------------------------------------------------


def _docker_image_exists(tag: str) -> bool:
    """Return True if *tag* is present in the local Docker image store."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", tag],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


@pytest.fixture(scope="session", autouse=True)
def require_docker_image() -> None:
    """Fail the entire session if the Docker image is not present.

    Raises:
        pytest.fail: If ``elastic-lammps-tool:latest`` is not in the local
            Docker image store or if the Docker CLI is unavailable.

    Preconditions:
        - Docker daemon is running.
        - Image ``elastic-lammps-tool:latest`` has been built.
    """
    if not _docker_image_exists("elastic-lammps-tool:latest"):
        pytest.fail(
            "Docker image elastic-lammps-tool:latest is not built. "
            "Run: docker build -t elastic-lammps-tool:latest "
            "src/tools/elastic_constants_lammps/"
        )


@pytest.fixture(scope="session", autouse=True)
def require_openai_key() -> None:
    """Fail the entire session if OPENAI_API_KEY is not set.

    Required for any test in this file that calls ``perform_real_simulation``
    via a real OpenAI Responses API call.

    Raises:
        pytest.fail: If ``OPENAI_API_KEY`` is absent from the environment.

    Pre-conditions:
        Test session is starting.
    Post-conditions:
        All tests in this module that need an API key may proceed.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.fail(
            "OPENAI_API_KEY is not set. Set it in .env or environment before "
            "running this test."
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_layer1(result: dict) -> None:
    """Layer 1: status and schema validation."""
    assert result.get("status") == "ok", f"Expected status 'ok', got: {result}"
    for key in ("C11", "C12", "C44", "runtime_seconds"):
        assert key in result, f"Missing key '{key}' in result: {result}"
    assert isinstance(result["C11"], float), "C11 must be a float"
    assert isinstance(result["C12"], float), "C12 must be a float"
    assert isinstance(result["C44"], float), "C44 must be a float"
    assert result["runtime_seconds"] > 0, "runtime_seconds must be positive"


def _assert_layer2(result: dict) -> None:
    """Layer 2: Born stability criteria for cubic crystals."""
    c11, c12, c44 = result["C11"], result["C12"], result["C44"]
    assert c44 > 0, f"C44={c44} must be positive"
    assert c11 > 0, f"C11={c11} must be positive"
    assert c12 > 0, f"C12={c12} must be positive"
    assert c11 > c12, f"C11={c11} must exceed C12={c12}"
    assert c11 - c12 > 0, "Tetragonal shear stability: C11 - C12 > 0"
    assert c11 + 2 * c12 > 0, "Bulk modulus: C11 + 2*C12 > 0"


def _assert_layer3(result: dict, c11_range, c12_range, c44_range) -> None:
    """Layer 3: EAM potential-specific tight ranges for C11, C12, and C44 (±5% of known EAM values)."""
    assert c11_range[0] <= result["C11"] <= c11_range[1], (
        f"C11={result['C11']:.1f} GPa out of expected range {c11_range}"
    )
    assert c12_range[0] <= result["C12"] <= c12_range[1], (
        f"C12={result['C12']:.1f} GPa out of expected range {c12_range}"
    )
    assert c44_range[0] <= result["C44"] <= c44_range[1], (
        f"C44={result['C44']:.1f} GPa out of expected range {c44_range}"
    )


# ---------------------------------------------------------------------------
# Per-element tests (supercell_size=3 for speed, still physically valid)
# ---------------------------------------------------------------------------


def test_elastic_Al():
    """Al (FCC, Mishin 1999): C11≈114, C12≈62, C44≈32 GPa."""
    result = compute_elastic_constants_tool(composition="Al", supercell_size=3)
    _assert_layer1(result)
    _assert_layer2(result)
    _assert_layer3(result, *_ELEMENT_RANGES["Al"])


def test_elastic_Cu():
    """Cu (FCC, Mishin 2001): C11≈170, C12≈122, C44≈76 GPa."""
    result = compute_elastic_constants_tool(composition="Cu", supercell_size=3)
    _assert_layer1(result)
    _assert_layer2(result)
    _assert_layer3(result, *_ELEMENT_RANGES["Cu"])


def test_elastic_Ni():
    """Ni (FCC, Mishin 1999): C11≈251, C12≈148, C44≈123 GPa."""
    result = compute_elastic_constants_tool(composition="Ni", supercell_size=4)
    _assert_layer1(result)
    _assert_layer2(result)
    _assert_layer3(result, *_ELEMENT_RANGES["Ni"])


def test_elastic_Fe():
    """Fe (BCC, Mendelev 2003): C11≈243, C12≈145, C44≈116 GPa."""
    result = compute_elastic_constants_tool(composition="Fe", supercell_size=4)
    _assert_layer1(result)
    _assert_layer2(result)
    _assert_layer3(result, *_ELEMENT_RANGES["Fe"])


def test_elastic_W():
    """W (BCC, Zhou 2004): C11≈523, C12≈204, C44≈160 GPa."""
    result = compute_elastic_constants_tool(composition="W", supercell_size=3)
    _assert_layer1(result)
    _assert_layer2(result)
    _assert_layer3(result, *_ELEMENT_RANGES["W"])


def test_elastic_Mo():
    """Mo (BCC, Zhou 2004): C11≈457, C12≈166, C44≈113 GPa."""
    result = compute_elastic_constants_tool(composition="Mo", supercell_size=5)
    _assert_layer1(result)
    _assert_layer2(result)
    _assert_layer3(result, *_ELEMENT_RANGES["Mo"])


# ---------------------------------------------------------------------------
# End-to-end pipeline test: perform_real_simulation across all remainder classes
# ---------------------------------------------------------------------------

# (prompt_length, expected_n) — all 6 remainder classes + len=0 boundary (7 rows)
_PERF_SIM_CASES = [
    pytest.param(0, 1, id="len=0_n=1_boundary"),
    pytest.param(6, 1, id="len=6_n=1_remainder0"),
    pytest.param(7, 2, id="len=7_n=2_remainder1"),
    pytest.param(8, 3, id="len=8_n=3_remainder2"),
    pytest.param(9, 4, id="len=9_n=4_remainder3"),
    pytest.param(10, 5, id="len=10_n=5_remainder4"),
    pytest.param(5, 6, id="len=5_n=6_remainder5"),
]


@pytest.mark.parametrize("prompt_length,expected_n", _PERF_SIM_CASES)
def test_perform_real_simulation_all_remainder_classes(
    prompt_length: int, expected_n: int
) -> None:
    """Full pipeline: perform_real_simulation returns correct results for every remainder class.

    Exercises the complete call chain
    ``perform_real_simulation → _get_elastic_constants_params_from_LLM → Docker``
    for each of the 7 synthetic prompt lengths (6 remainder classes + len=0
    boundary).  Applies all three validation layers to every result entry and
    asserts the compositions match the expected ordered prefix of
    ``_PREDEFINED_SIM_CALLS``.

    Args:
        prompt_length: Length of synthetic prompt ``"x" * prompt_length``.
        expected_n: Expected number of simulation results, ``prompt_length % 6 + 1``.

    Pre-conditions:
        - ``elastic-lammps-tool:latest`` Docker image is built (enforced by
          ``require_docker_image`` session fixture).
        - ``OPENAI_API_KEY`` is set (enforced by ``require_openai_key`` session
          fixture).
        - ``SimulationAgent`` is constructed fresh inside this function body.
    Post-conditions:
        - ``result`` is a ``list[str]`` of length ``expected_n``.
        - Every entry passes Layer 1 (status/schema), Layer 2 (Born stability),
          and Layer 3 (EAM potential-specific ranges from ``_ELEMENT_RANGES``).
        - Compositions match the ordered prefix of ``_PREDEFINED_SIM_CALLS[:n]``.

    Complexity:
        ``expected_n`` Docker container invocations plus one OpenAI Responses
        API call per parametrized invocation.
    """
    agent = SimulationAgent(provider="openai")
    prompt = "x" * prompt_length

    result = agent.perform_real_simulation(prompt)

    assert isinstance(result, list), "perform_real_simulation must return a list"
    assert len(result) == expected_n, (
        f"Expected {expected_n} results for prompt length {prompt_length}, "
        f"got {len(result)}"
    )

    parsed_results = [json.loads(entry) for entry in result]

    for entry in parsed_results:
        _assert_layer1(entry)
        _assert_layer2(entry)

        composition = entry.get("composition")
        if composition not in _ELEMENT_RANGES:
            pytest.fail(
                f"Unknown composition '{composition}' returned by LLM — not in "
                f"_ELEMENT_RANGES. _PREDEFINED_SIM_CALLS must have drifted from "
                f"_ELEMENT_RANGES."
            )
        _assert_layer3(entry, *_ELEMENT_RANGES[composition])

    actual_compositions = [e["composition"] for e in parsed_results]
    expected_compositions = [p[0] for p in _PREDEFINED_SIM_CALLS[:expected_n]]
    assert actual_compositions == expected_compositions, (
        f"Composition prefix mismatch: expected {expected_compositions}, "
        f"got {actual_compositions}"
    )

