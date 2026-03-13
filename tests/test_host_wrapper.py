"""Unit tests for src/tools/elastic_constants_lammps/host_wrapper.py.

All tests mock subprocess.run so no Docker daemon or container image is
required.  Tests cover:

- Docker command structure (no volume mount, correct composition flag)
- Automatic potential mapping for known and unknown elements
- Explicit potential override bypasses auto-mapping
- Default supercell_size
- Successful JSON parsing from container stdout
- Error dict on non-zero exit code
- Error dict on subprocess.TimeoutExpired
- Error dict on FileNotFoundError (Docker CLI missing)
- Error dict on non-JSON stdout
- ELASTIC_IMAGE environment variable override
- OPENAI_TOOL_SCHEMA structure
- ANTHROPIC_TOOL_SCHEMA structure
"""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from src.tools.elastic_constants_lammps.host_wrapper import (
    ANTHROPIC_TOOL_SCHEMA,
    OPENAI_TOOL_SCHEMA,
    _DEFAULT_POTENTIALS,
    _resolve_potential,
    compute_elastic_constants_tool,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SUCCESS_PAYLOAD = {
    "composition": "Al",
    "C11": 114.3,
    "C12": 61.8,
    "C44": 31.6,
    "runtime_seconds": 52.1,
    "status": "ok",
}


def _make_completed_process(stdout: str, returncode: int = 0) -> MagicMock:
    proc = MagicMock(spec=subprocess.CompletedProcess)
    proc.stdout = stdout
    proc.stderr = ""
    proc.returncode = returncode
    return proc


# ---------------------------------------------------------------------------
# _resolve_potential unit tests
# ---------------------------------------------------------------------------

def test_resolve_potential_explicit_override():
    assert _resolve_potential("Al", "/custom/path.alloy") == "/custom/path.alloy"


def test_resolve_potential_known_element_al():
    assert _resolve_potential("Al", None) == "/app/potentials/Al.eam.alloy"


def test_resolve_potential_known_element_cu():
    assert _resolve_potential("Cu", None) == "/app/potentials/Cu.eam.alloy"


def test_resolve_potential_unknown_element_fallback():
    assert _resolve_potential("Zr", None) == "/app/potentials/Zr.eam.alloy"


# ---------------------------------------------------------------------------
# Docker command structure tests
# ---------------------------------------------------------------------------

def test_docker_command_no_volume_mount():
    with patch("subprocess.run", return_value=_make_completed_process(json.dumps(_SUCCESS_PAYLOAD))) as mock_run:
        compute_elastic_constants_tool("Al")
        cmd = mock_run.call_args[0][0]
        assert "-v" not in cmd


def test_docker_command_uses_composition():
    with patch("subprocess.run", return_value=_make_completed_process(json.dumps(_SUCCESS_PAYLOAD))) as mock_run:
        compute_elastic_constants_tool("Al")
        cmd = mock_run.call_args[0][0]
        assert "--composition" in cmd
        assert "Al" in cmd


def test_potential_auto_mapped_al():
    with patch("subprocess.run", return_value=_make_completed_process(json.dumps(_SUCCESS_PAYLOAD))) as mock_run:
        compute_elastic_constants_tool("Al")
        cmd = mock_run.call_args[0][0]
        potential_idx = cmd.index("--potential") + 1
        assert cmd[potential_idx] == "/app/potentials/Al.eam.alloy"


def test_potential_auto_mapped_cu():
    cu_payload = {**_SUCCESS_PAYLOAD, "composition": "Cu"}
    with patch("subprocess.run", return_value=_make_completed_process(json.dumps(cu_payload))) as mock_run:
        compute_elastic_constants_tool("Cu")
        cmd = mock_run.call_args[0][0]
        potential_idx = cmd.index("--potential") + 1
        assert cmd[potential_idx] == "/app/potentials/Cu.eam.alloy"


def test_potential_auto_mapped_unknown_fallback():
    zr_payload = {**_SUCCESS_PAYLOAD, "composition": "Zr"}
    with patch("subprocess.run", return_value=_make_completed_process(json.dumps(zr_payload))) as mock_run:
        compute_elastic_constants_tool("Zr")
        cmd = mock_run.call_args[0][0]
        potential_idx = cmd.index("--potential") + 1
        assert cmd[potential_idx] == "/app/potentials/Zr.eam.alloy"


def test_potential_explicit_override():
    with patch("subprocess.run", return_value=_make_completed_process(json.dumps(_SUCCESS_PAYLOAD))) as mock_run:
        compute_elastic_constants_tool("Al", potential="/custom/Al_v2.eam.alloy")
        cmd = mock_run.call_args[0][0]
        potential_idx = cmd.index("--potential") + 1
        assert cmd[potential_idx] == "/custom/Al_v2.eam.alloy"


def test_docker_command_default_supercell():
    with patch("subprocess.run", return_value=_make_completed_process(json.dumps(_SUCCESS_PAYLOAD))) as mock_run:
        compute_elastic_constants_tool("Al")
        cmd = mock_run.call_args[0][0]
        supercell_idx = cmd.index("--supercell_size") + 1
        assert cmd[supercell_idx] == "4"


# ---------------------------------------------------------------------------
# Return value tests
# ---------------------------------------------------------------------------

def test_returns_parsed_json_on_success():
    with patch("subprocess.run", return_value=_make_completed_process(json.dumps(_SUCCESS_PAYLOAD))):
        result = compute_elastic_constants_tool("Al")
    assert result["status"] == "ok"
    assert result["C11"] == pytest.approx(114.3)
    assert result["C12"] == pytest.approx(61.8)
    assert result["C44"] == pytest.approx(31.6)


def test_returns_error_dict_on_nonzero_exit():
    error_payload = {
        "composition": "Al",
        "C11": None,
        "C12": None,
        "C44": None,
        "runtime_seconds": None,
        "status": "error",
        "error": "element not found",
    }
    with patch("subprocess.run", return_value=_make_completed_process(json.dumps(error_payload), returncode=1)):
        result = compute_elastic_constants_tool("Al")
    assert result["status"] == "error"


def test_returns_error_dict_on_nonzero_exit_no_json():
    proc = _make_completed_process("fatal error\n", returncode=1)
    proc.stderr = "docker: error"
    with patch("subprocess.run", return_value=proc):
        result = compute_elastic_constants_tool("Al")
    assert result["status"] == "error"
    assert result["C11"] is None


def test_returns_error_dict_on_timeout():
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="docker", timeout=300)):
        result = compute_elastic_constants_tool("Al")
    assert result["status"] == "error"
    assert "timed out" in result["error"].lower()


def test_returns_error_dict_on_docker_not_found():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        result = compute_elastic_constants_tool("Al")
    assert result["status"] == "error"
    assert "docker" in result["error"].lower()


def test_returns_error_dict_on_bad_json():
    with patch("subprocess.run", return_value=_make_completed_process("this is not json")):
        result = compute_elastic_constants_tool("Al")
    assert result["status"] == "error"
    assert result["C11"] is None


# ---------------------------------------------------------------------------
# Environment variable override
# ---------------------------------------------------------------------------

def test_elastic_image_env_override(monkeypatch):
    monkeypatch.setenv("ELASTIC_IMAGE", "my-image:v2")
    with patch("subprocess.run", return_value=_make_completed_process(json.dumps(_SUCCESS_PAYLOAD))) as mock_run:
        compute_elastic_constants_tool("Al")
        cmd = mock_run.call_args[0][0]
    assert "my-image:v2" in cmd


# ---------------------------------------------------------------------------
# Schema structure tests
# ---------------------------------------------------------------------------

def test_openai_schema_structure():
    assert OPENAI_TOOL_SCHEMA["type"] == "function"
    fn = OPENAI_TOOL_SCHEMA["function"]
    assert fn["name"] == "compute_elastic_constants_tool"
    assert "composition" in fn["parameters"]["required"]
    assert "composition" in fn["parameters"]["properties"]
    assert "potential" in fn["parameters"]["properties"]
    assert "supercell_size" in fn["parameters"]["properties"]


def test_anthropic_schema_structure():
    assert ANTHROPIC_TOOL_SCHEMA["name"] == "compute_elastic_constants_tool"
    assert "input_schema" in ANTHROPIC_TOOL_SCHEMA
    schema = ANTHROPIC_TOOL_SCHEMA["input_schema"]
    assert "composition" in schema["required"]
    assert "composition" in schema["properties"]
    assert "potential" in schema["properties"]
    assert "supercell_size" in schema["properties"]
