"""Unit tests for src/tools/elastic_constants_lammps/elastic_tool.py.

Tests cover pure Python logic only — element lookup, argparse, patch helpers,
log parser, subprocess wrapper, and JSON output schema.  No Docker, no LAMMPS
installation, and no subprocess execution is required; all external I/O is
either done against temporary files or mocked with unittest.mock.
"""

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, call, patch

import pytest

import src.tools.elastic_constants_lammps.elastic_tool as elastic_tool


# ---------------------------------------------------------------------------
# Element data lookup
# ---------------------------------------------------------------------------

def test_element_lookup_fcc():
    """FCC element has correct structure, a0, and mass."""
    elem = elastic_tool.ELEMENT_DATA["Al"]
    assert elem["structure"] == "fcc"
    assert elem["a0"] == pytest.approx(4.05)
    assert elem["mass"] == pytest.approx(26.982)


def test_element_lookup_bcc():
    """BCC element has correct structure, a0, and mass."""
    elem = elastic_tool.ELEMENT_DATA["Fe"]
    assert elem["structure"] == "bcc"
    assert elem["a0"] == pytest.approx(2.87)
    assert elem["mass"] == pytest.approx(55.845)


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
# _patch_init_mod
# ---------------------------------------------------------------------------

def test_patch_init_mod_replaces_all_four_lines():
    """All 4 Si-specific lines are replaced correctly for an FCC element."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "init.mod"), "w") as f:
            f.write(
                "variable a equal 5.43\n"
                "lattice         diamond $a\n"
                "region\t\tbox prism 0 2.0 0 3.0 0 4.0 0.0 0.0 0.0\n"
                "mass 1 1.0e-20\n"
            )
        elastic_tool._patch_init_mod(tmpdir, "Al", 3)
        with open(os.path.join(tmpdir, "init.mod")) as f:
            content = f.read()
    assert "variable a equal 4.05" in content
    assert "lattice fcc $a" in content
    assert "region box prism 0 3 0 3 0 3 0.0 0.0 0.0" in content
    assert "mass 1 26.982" in content
    assert "5.43" not in content
    assert "diamond" not in content


def test_patch_init_mod_bcc_element():
    """BCC element receives correct structure and mass."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "init.mod"), "w") as f:
            f.write(
                "variable a equal 5.43\n"
                "lattice         diamond $a\n"
                "region\t\tbox prism 0 2.0 0 3.0 0 4.0 0.0 0.0 0.0\n"
                "mass 1 1.0e-20\n"
            )
        elastic_tool._patch_init_mod(tmpdir, "Fe", 4)
        with open(os.path.join(tmpdir, "init.mod")) as f:
            content = f.read()
    assert "lattice bcc $a" in content
    assert "mass 1 55.845" in content


def test_patch_init_mod_raises_on_missing_string():
    """RuntimeError is raised when a target string is absent from init.mod."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "init.mod"), "w") as f:
            f.write("variable a equal 5.43\n")  # other 3 lines absent
        with pytest.raises(RuntimeError, match="init.mod"):
            elastic_tool._patch_init_mod(tmpdir, "Al", 3)


# ---------------------------------------------------------------------------
# _patch_potential_files
# ---------------------------------------------------------------------------

def _write_minimal_potential_mod(tmpdir: str) -> None:
    with open(os.path.join(tmpdir, "potential.mod"), "w") as f:
        f.write("pair_style\tsw\npair_coeff * * Si.sw Si\n")


def test_patch_potential_files_eam_alloy():
    """pair_style eam/alloy and correct pair_coeff are written; potential.inp is created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_minimal_potential_mod(tmpdir)
        elastic_tool._patch_potential_files(tmpdir, "Al", "/app/potentials/Al.eam.alloy")
        with open(os.path.join(tmpdir, "potential.mod")) as f:
            mod = f.read()
        with open(os.path.join(tmpdir, "potential.inp")) as f:
            inp = f.read()
    assert "pair_style eam/alloy" in mod
    assert "pair_coeff * * /app/potentials/Al.eam.alloy Al" in mod
    assert mod == inp  # shutil.copy guarantees identical content


def test_patch_potential_files_eam_fs():
    """pair_style eam/fs is selected for a .eam.fs potential file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_minimal_potential_mod(tmpdir)
        elastic_tool._patch_potential_files(tmpdir, "Fe", "/app/potentials/Fe.eam.fs")
        with open(os.path.join(tmpdir, "potential.mod")) as f:
            mod = f.read()
    assert "pair_style eam/fs" in mod
    assert "pair_coeff * * /app/potentials/Fe.eam.fs Fe" in mod


def test_patch_potential_files_raises_on_missing_string():
    """RuntimeError is raised when a target string is absent from potential.mod."""
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "potential.mod"), "w") as f:
            f.write("pair_style\tsw\n")  # pair_coeff line absent
        with pytest.raises(RuntimeError, match="potential.mod"):
            elastic_tool._patch_potential_files(tmpdir, "Al", "/app/potentials/Al.eam.alloy")


# ---------------------------------------------------------------------------
# _patch_in_elastic
# ---------------------------------------------------------------------------

def test_patch_in_elastic_appends_three_print_lines():
    """3 print statements are appended; original content is preserved."""
    original = "# existing content\nsome lammps commands\n"
    with tempfile.TemporaryDirectory() as tmpdir:
        with open(os.path.join(tmpdir, "in.elastic"), "w") as f:
            f.write(original)
        elastic_tool._patch_in_elastic(tmpdir)
        with open(os.path.join(tmpdir, "in.elastic")) as f:
            content = f.read()
    assert content.startswith(original)
    assert 'print "C11cubic = ${C11cubic} ${cunits}"' in content
    assert 'print "C12cubic = ${C12cubic} ${cunits}"' in content
    assert 'print "C44cubic = ${C44cubic} ${cunits}"' in content


# ---------------------------------------------------------------------------
# _parse_log_lammps
# ---------------------------------------------------------------------------

def _write_log(tmpdir: str, text: str) -> str:
    path = os.path.join(tmpdir, "log.lammps")
    with open(path, "w") as f:
        f.write(text)
    return path


def test_parse_log_lammps_extracts_correct_values():
    """C11, C12, C44 are correctly parsed and rounded."""
    log = (
        "... lots of lammps output ...\n"
        "C11cubic = 114.3 GPa\n"
        "C12cubic = 62.1 GPa\n"
        "C44cubic = 32.7 GPa\n"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        result = elastic_tool._parse_log_lammps(_write_log(tmpdir, log))
    assert result == {"C11": 114.3, "C12": 62.1, "C44": 32.7}


def test_parse_log_lammps_raises_on_missing_key():
    """RuntimeError is raised when a cubic constant is absent from the log."""
    log = "C11cubic = 114.3 GPa\nC12cubic = 62.1 GPa\n"  # C44cubic absent
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(RuntimeError, match="C44cubic"):
            elastic_tool._parse_log_lammps(_write_log(tmpdir, log))


def test_parse_log_lammps_raises_on_negative_value():
    """RuntimeError is raised when a parsed value is outside (0, 2000) GPa."""
    log = "C11cubic = -5.0 GPa\nC12cubic = 62.1 GPa\nC44cubic = 32.7 GPa\n"
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(RuntimeError, match="physical range"):
            elastic_tool._parse_log_lammps(_write_log(tmpdir, log))


def test_parse_log_lammps_raises_on_zero_value():
    """RuntimeError is raised when a parsed value is exactly 0."""
    log = "C11cubic = 0.0 GPa\nC12cubic = 62.1 GPa\nC44cubic = 32.7 GPa\n"
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(RuntimeError, match="physical range"):
            elastic_tool._parse_log_lammps(_write_log(tmpdir, log))


# ---------------------------------------------------------------------------
# _run_lammps
# ---------------------------------------------------------------------------

def test_run_lammps_calls_correct_command():
    """subprocess.run receives ['lmp', '-in', 'in.elastic'] with correct cwd and timeout."""
    with patch("src.tools.elastic_constants_lammps.elastic_tool.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        elastic_tool._run_lammps("/fake/tmpdir")
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    assert args[0] == ["lmp", "-in", "in.elastic"]
    assert kwargs["cwd"] == "/fake/tmpdir"
    assert kwargs["timeout"] == elastic_tool._LAMMPS_TIMEOUT_S


def test_run_lammps_raises_on_nonzero_exit_no_log():
    """RuntimeError is raised on failure when no log.lammps file exists."""
    with patch("src.tools.elastic_constants_lammps.elastic_tool.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="bad error")
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(RuntimeError, match="LAMMPS exited"):
                elastic_tool._run_lammps(tmpdir)


def test_run_lammps_includes_log_tail_on_failure():
    """RuntimeError message includes the last lines of log.lammps when present."""
    with patch("src.tools.elastic_constants_lammps.elastic_tool.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=1, stderr="")
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "log.lammps"), "w") as f:
                f.write("line one\nline two\nTHE_TAIL_LINE\n")
            with pytest.raises(RuntimeError, match="THE_TAIL_LINE"):
                elastic_tool._run_lammps(tmpdir)


# ---------------------------------------------------------------------------
# _copy_scripts
# ---------------------------------------------------------------------------

def test_copy_scripts_copies_all_five():
    """shutil.copy is called exactly once for each of the 5 script files."""
    with patch("src.tools.elastic_constants_lammps.elastic_tool.shutil.copy") as mock_copy:
        elastic_tool._copy_scripts("/fake/tmpdir")
    copied_names = {os.path.basename(c.args[0]) for c in mock_copy.call_args_list}
    assert copied_names == {
        "in.elastic", "displace.mod", "init.mod", "potential.mod", "compliance.py"
    }
    assert mock_copy.call_count == 5


# ---------------------------------------------------------------------------
# JSON output schema (end-to-end, sub-functions mocked)
# ---------------------------------------------------------------------------

def test_json_output_schema():
    """compute_elastic_constants returns a dict with float C11, C12, C44."""
    with patch.object(elastic_tool, "_copy_scripts"), \
         patch.object(elastic_tool, "_patch_all"), \
         patch.object(elastic_tool, "_run_lammps"), \
         patch.object(elastic_tool, "_parse_log_lammps",
                      return_value={"C11": 114.3, "C12": 62.1, "C44": 32.7}):
        result = elastic_tool.compute_elastic_constants("Al", "/fake.alloy", 4)
    assert isinstance(result["C11"], float)
    assert isinstance(result["C12"], float)
    assert isinstance(result["C44"], float)


# ---------------------------------------------------------------------------
# Error handling (main CLI)
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
