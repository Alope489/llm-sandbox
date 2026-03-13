"""Portable host-side wrapper for the elastic-lammps-tool Docker container.

This module is the entry point for LLM pipelines and other Python code running
on the host machine. It starts the elastic-lammps-tool container via
subprocess, passes the requested parameters, captures JSON output from
elastic_tool.py, and returns the results as a Python dict.

The caller does not need a Docker SDK — only the Docker CLI must be installed
and the image must have been built locally:

    docker build -t elastic-lammps-tool:latest \\
        src/tools/elastic_constants_lammps/

Typical runtime is 30–120 s depending on supercell size.

Public API
----------
compute_elastic_constants_tool(composition, potential, supercell_size) -> dict
    Run the container and return the elastic constant results.

Exported schemas
----------------
OPENAI_TOOL_SCHEMA      Ready-to-use OpenAI function-calling schema dict.
ANTHROPIC_TOOL_SCHEMA   Ready-to-use Anthropic tool-use schema dict.

Environment variables (optional overrides)
------------------------------------------
ELASTIC_IMAGE : Docker image name (default: elastic-lammps-tool:latest)
"""

import json
import os
import subprocess

# ---------------------------------------------------------------------------
# Automatic potential mapping
# ---------------------------------------------------------------------------

_DEFAULT_POTENTIALS = {
    "Al": "/app/potentials/Al.eam.alloy",
    "Cu": "/app/potentials/Cu.eam.alloy",
    "Ni": "/app/potentials/Ni.eam.alloy",
    "Fe": "/app/potentials/Fe.eam.alloy",
    "W":  "/app/potentials/W.eam.alloy",
    "Mo": "/app/potentials/Mo.eam.alloy",
}


def _resolve_potential(composition: str, potential: "str | None") -> str:
    """Return the resolved potential path.

    Resolution order:
    1. If *potential* is explicitly provided (not None), use it as-is.
    2. Look up *composition* in ``_DEFAULT_POTENTIALS``.
    3. Fall back to ``/app/potentials/{composition}.eam.alloy``.

    Args:
        composition: Element symbol, e.g. ``"Al"``.
        potential: Caller-supplied path override, or ``None``.

    Returns:
        Absolute potential path to pass to the container.
    """
    if potential is not None:
        return potential
    return _DEFAULT_POTENTIALS.get(composition, f"/app/potentials/{composition}.eam.alloy")


# ---------------------------------------------------------------------------
# Main wrapper function
# ---------------------------------------------------------------------------

def compute_elastic_constants_tool(
    composition: str,
    potential: "str | None" = None,
    supercell_size: int = 4,
) -> dict:
    """Compute elastic constants C11, C12, C44 (GPa) via the LAMMPS container.

    Launches ``elastic-lammps-tool:latest`` (or the image named by the
    ``ELASTIC_IMAGE`` environment variable) using ``subprocess.run``.  No
    Docker SDK is needed — only the Docker CLI must be on PATH.  The potential
    file is selected automatically from *composition* unless *potential* is
    supplied explicitly.

    Typical runtime is 30–120 s depending on *supercell_size*.

    Args:
        composition: Element symbol.  Built-in auto-mapping covers
            ``Al``, ``Cu``, ``Ni``, ``Fe``, ``W``, ``Mo``.
        potential: Optional override — absolute path to an EAM/alloy
            potential *inside* the container.  When ``None`` the path is
            resolved from ``_DEFAULT_POTENTIALS`` (falling back to
            ``/app/potentials/{composition}.eam.alloy`` for unknown elements).
        supercell_size: Unit cells per axis.  Default 4 gives 256 atoms
            for an FCC lattice.

    Returns:
        Parsed JSON dict from ``elastic_tool.py``.  On success the shape is::

            {
                "composition": "Al",
                "C11": 114.3,
                "C12": 61.8,
                "C44": 31.6,
                "runtime_seconds": 52.1,
                "status": "ok"
            }

        On any failure the wrapper returns an error dict instead of raising::

            {
                "composition": "<composition>",
                "C11": null,
                "C12": null,
                "C44": null,
                "runtime_seconds": null,
                "status": "error",
                "error": "<message>"
            }

    Preconditions:
        - Docker CLI is installed and on PATH.
        - The image ``elastic-lammps-tool:latest`` (or ``ELASTIC_IMAGE``) has
          been built from ``src/tools/elastic_constants_lammps/``.
        - The chosen potential file exists inside the container image.

    Postconditions:
        - Returns a dict; never raises.
        - When ``status == "ok"``, ``C11``, ``C12``, and ``C44`` are floats
          in GPa.
    """
    resolved_potential = _resolve_potential(composition, potential)
    image = os.environ.get("ELASTIC_IMAGE", "elastic-lammps-tool:latest")
    cmd = [
        "docker", "run", "--rm",
        image,
        "--composition", composition,
        "--potential", resolved_potential,
        "--supercell_size", str(supercell_size),
    ]

    def _error(msg: str) -> dict:
        return {
            "composition": composition,
            "C11": None,
            "C12": None,
            "C44": None,
            "runtime_seconds": None,
            "status": "error",
            "error": msg,
        }

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return _error("Docker container timed out after 300 seconds.")
    except FileNotFoundError:
        return _error(
            "Docker CLI not found. Ensure Docker is installed and on PATH."
        )
    except Exception as exc:
        return _error(f"Unexpected error launching Docker: {exc}")

    if result.returncode != 0:
        stderr_hint = result.stderr.strip()
        try:
            return json.loads(result.stdout.strip())
        except (json.JSONDecodeError, ValueError):
            return _error(
                f"Container exited with code {result.returncode}. "
                + (f"stderr: {stderr_hint}" if stderr_hint else "No stderr output.")
            )

    try:
        return json.loads(result.stdout.strip())
    except (json.JSONDecodeError, ValueError):
        return _error(
            f"Container returned non-JSON output: {result.stdout.strip()!r}"
        )


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_TOOL_DESCRIPTION = (
    "Run a high-fidelity LAMMPS molecular dynamics simulation to compute the three "
    "independent elastic constants C11, C12, and C44 (all in GPa) for a pure cubic "
    "(FCC or BCC) metallic element using EAM interatomic potentials.\n\n"
    "Physical meaning:\n"
    "  C11 — longitudinal stiffness; resistance to uniaxial compression/extension along "
    "a crystal axis. Higher C11 means a stiffer material.\n"
    "  C12 — cross-stiffness; transverse stress induced by axial strain. Together with "
    "C11 it determines bulk modulus: K = (C11 + 2*C12) / 3.\n"
    "  C44 — shear stiffness; resistance to shear deformation. Higher C44 means more "
    "resistance to slip and plastic deformation.\n\n"
    "Call this tool when:\n"
    "  - You need simulation-derived elastic moduli rather than approximate textbook values.\n"
    "  - You want to reason about mechanical stiffness, anisotropy, or deformation "
    "resistance of a constituent element in an alloy.\n"
    "  - You need C11, C12, C44 to compute derived quantities such as bulk modulus K, "
    "shear modulus G = C44, or Zener anisotropy A = 2*C44 / (C11 - C12).\n"
    "  - The prompt mentions elastic constants, stiffness, moduli, or mechanical properties.\n\n"
    "Supported elements with auto-mapped EAM potentials: Al (FCC), Cu (FCC), Ni (FCC), "
    "Fe (BCC), W (BCC), Mo (BCC). Runtime: 30-120 seconds."
)

_COMPOSITION_DESCRIPTION = (
    "Element symbol for the metal to simulate. "
    "Supported with auto-mapped potentials: Al, Cu, Ni, Fe, W, Mo. "
    "For other elements supply the 'potential' argument explicitly."
)

_POTENTIAL_DESCRIPTION = (
    "Optional override: absolute path to an EAM/alloy potential file inside the container "
    "(e.g. '/app/potentials/MyElement.eam.alloy'). "
    "Leave unset to use the automatically mapped potential for the given element symbol. "
    "Only supply this when using a custom or non-standard potential."
)

_SUPERCELL_DESCRIPTION = (
    "Number of unit cells per axis for the simulation supercell. "
    "Default 4 produces 256 atoms for FCC or 128 for BCC. "
    "Increase to 6 or 8 for higher accuracy at the cost of longer runtime (up to 120 s). "
    "Decrease to 3 for a quick smoke-test (~10 s) with lower accuracy."
)

OPENAI_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "compute_elastic_constants_tool",
        "description": _TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "composition": {
                    "type": "string",
                    "description": _COMPOSITION_DESCRIPTION,
                },
                "potential": {
                    "type": "string",
                    "description": _POTENTIAL_DESCRIPTION,
                },
                "supercell_size": {
                    "type": "integer",
                    "description": _SUPERCELL_DESCRIPTION,
                },
            },
            "required": ["composition"],
        },
    },
}

ANTHROPIC_TOOL_SCHEMA = {
    "name": "compute_elastic_constants_tool",
    "description": _TOOL_DESCRIPTION,
    "input_schema": {
        "type": "object",
        "properties": {
            "composition": {
                "type": "string",
                "description": _COMPOSITION_DESCRIPTION,
            },
            "potential": {
                "type": "string",
                "description": _POTENTIAL_DESCRIPTION,
            },
            "supercell_size": {
                "type": "integer",
                "description": _SUPERCELL_DESCRIPTION,
            },
        },
        "required": ["composition"],
    },
}
