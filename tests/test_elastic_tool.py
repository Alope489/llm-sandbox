"""Unit tests for src/tools/elastic_constants_lammps/elastic_tool.py.

Tests cover pure Python logic only — element lookup, argparse, linear
regression math, JSON output schema, and error handling.  LAMMPS internals
are mocked so no Docker or LAMMPS installation is needed.
"""

import json
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import src.tools.elastic_constants_lammps.elastic_tool as elastic_tool


# ---------------------------------------------------------------------------
# Element data lookup
# ---------------------------------------------------------------------------

def test_element_lookup_fcc():
    elem = elastic_tool.ELEMENT_DATA["Al"]
    assert elem["structure"] == "fcc"
    assert elem["a0"] == pytest.approx(4.05)


def test_element_lookup_bcc():
    elem = elastic_tool.ELEMENT_DATA["Fe"]
    assert elem["structure"] == "bcc"
    assert elem["a0"] == pytest.approx(2.87)


def test_unknown_element_raises():
    with pytest.raises(KeyError):
        elastic_tool.compute_elastic_constants("Zr", "/fake/path.alloy", 4)


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def test_argparse_defaults():
    args = elastic_tool.parse_args(
        ["--composition", "Al", "--potential", "/app/potentials/Al.eam.alloy"]
    )
    assert args.composition == "Al"
    assert args.potential == "/app/potentials/Al.eam.alloy"
    assert args.supercell_size == 4


@pytest.mark.parametrize("argv", [
    ["--potential", "/app/potentials/Al.eam.alloy"],  # missing --composition
    ["--composition", "Al"],                          # missing --potential
    [],                                               # both missing
])
def test_argparse_required_args(argv):
    with pytest.raises(SystemExit):
        elastic_tool.parse_args(argv)


# ---------------------------------------------------------------------------
# Linear regression math
# ---------------------------------------------------------------------------

def test_linear_regression_c11():
    C11_expected_GPa = 100.0
    C11_expected_bar = C11_expected_GPa / elastic_tool.BAR_TO_GPA
    strains = np.array(elastic_tool.STRAIN_VALUES)
    pxx_values = [-C11_expected_bar * e for e in strains]
    slope = np.polyfit(strains, [-p for p in pxx_values], 1)[0]
    C11_recovered = slope * elastic_tool.BAR_TO_GPA
    assert C11_recovered == pytest.approx(C11_expected_GPa, abs=0.001)


def test_linear_regression_c44():
    C44_expected_GPa = 50.0
    C44_expected_bar = C44_expected_GPa / elastic_tool.BAR_TO_GPA
    strains = np.array(elastic_tool.STRAIN_VALUES)
    pxy_values = [-C44_expected_bar * e for e in strains]
    slope = np.polyfit(strains, [-p for p in pxy_values], 1)[0]
    C44_recovered = slope * elastic_tool.BAR_TO_GPA
    assert C44_recovered == pytest.approx(C44_expected_GPa, abs=0.001)


# ---------------------------------------------------------------------------
# JSON output schema
# ---------------------------------------------------------------------------

def _make_mock_lammps(lx_eq=16.2, pxx=-1e5, pyy=-5e4, pxy=-3e4):
    lmp = MagicMock()
    lmp.get_thermo.side_effect = lambda key: {
        "lx": lx_eq, "pxx": pxx, "pyy": pyy, "pxy": pxy
    }.get(key, 0.0)
    return lmp


def test_json_output_schema():
    with patch("src.tools.elastic_constants_lammps.elastic_tool._build_lammps",
               return_value=_make_mock_lammps()), \
         patch("src.tools.elastic_constants_lammps.elastic_tool._relax_box",
               return_value=16.2), \
         patch("src.tools.elastic_constants_lammps.elastic_tool._stress_for_strain",
               return_value={"pxx": -1e5, "pyy": -5e4, "pxy": -3e4}):
        result = elastic_tool.compute_elastic_constants("Al", "/fake.alloy", 4)

    assert "C11" in result
    assert "C12" in result
    assert "C44" in result
    assert isinstance(result["C11"], float)
    assert isinstance(result["C12"], float)
    assert isinstance(result["C44"], float)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_error_json_on_unknown_element(capsys):
    test_argv = [
        "--composition", "Zr",
        "--potential", "/app/potentials/Zr.eam.alloy",
    ]
    with pytest.raises(SystemExit) as exc_info:
        elastic_tool.main(test_argv)
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert output["status"] == "error"
    assert output["C11"] is None
