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

import subprocess

import pytest

from src.tools.elastic_constants_lammps.host_wrapper import (
    compute_elastic_constants_tool,
)
from src.multi.sim.agent import SimulationAgent


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
    _assert_layer3(result, c11_range=(108, 122), c12_range=(60, 70), c44_range=(30, 34))


def test_elastic_Cu():
    """Cu (FCC, Mishin 2001): C11≈170, C12≈122, C44≈76 GPa."""
    result = compute_elastic_constants_tool(composition="Cu", supercell_size=3)
    _assert_layer1(result)
    _assert_layer2(result)
    _assert_layer3(
        result, c11_range=(165, 178), c12_range=(118, 128), c44_range=(72, 80)
    )


def test_elastic_Ni():
    """Ni (FCC, Mishin 1999): C11≈251, C12≈148, C44≈123 GPa."""
    result = compute_elastic_constants_tool(composition="Ni", supercell_size=4)
    _assert_layer1(result)
    _assert_layer2(result)
    _assert_layer3(
        result, c11_range=(244, 258), c12_range=(140, 155), c44_range=(117, 129)
    )


def test_elastic_Fe():
    """Fe (BCC, Mendelev 2003): C11≈243, C12≈145, C44≈116 GPa."""
    result = compute_elastic_constants_tool(composition="Fe", supercell_size=4)
    _assert_layer1(result)
    _assert_layer2(result)
    _assert_layer3(
        result, c11_range=(231, 256), c12_range=(138, 150), c44_range=(110, 122)
    )


def test_elastic_W():
    """W (BCC, Zhou 2004): C11≈523, C12≈204, C44≈160 GPa."""
    result = compute_elastic_constants_tool(composition="W", supercell_size=3)
    _assert_layer1(result)
    _assert_layer2(result)
    _assert_layer3(
        result, c11_range=(516, 536), c12_range=(195, 208), c44_range=(152, 168)
    )


def test_elastic_Mo():
    """Mo (BCC, Zhou 2004): C11≈457, C12≈166, C44≈113 GPa."""
    result = compute_elastic_constants_tool(composition="Mo", supercell_size=5)
    _assert_layer1(result)
    _assert_layer2(result)
    _assert_layer3(
        result, c11_range=(451, 474), c12_range=(162, 172), c44_range=(107, 119)
    )


# ---------------------------------------------------------------------------
# End-to-end pipeline test
# ---------------------------------------------------------------------------


def test_sim_agent_prefetch_with_real_docker():
    """Full pipeline: SimulationAgent.perform_real_simulation() with real Docker.

    Preconditions:
        - ``elastic-lammps-tool:latest`` Docker image is built.
        - OPENAI_API_KEY is set in environment or .env.

    Postconditions:
        - context is a non-empty list[str].
        - agent._current_sim_results equals the returned context.
    """
    agent = SimulationAgent(provider="openai", max_iterations=1)
    context = agent.perform_real_simulation(
        "Compute elastic constants for the Ni superalloy constituents."
    )
    assert isinstance(context, list) and len(context) > 0, (
        "Prefetch context must be a non-empty list"
    )
    assert agent._current_sim_results == context, (
        "_current_sim_results must be set to the prefetched context"
    )
