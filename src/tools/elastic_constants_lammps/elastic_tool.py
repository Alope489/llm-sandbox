"""In-container LAMMPS script for computing elastic constants of FCC/BCC metals.

Method:
1. Box relaxation (CG minimiser + fix box/relax iso) to find equilibrium geometry.
2. Strain loop over ε ∈ {-0.01, -0.005, +0.005, +0.01} — fresh LAMMPS instance
   per strain state:
   - Uniaxial e11 strain on standard [100] crystal → P_xx (→ C11), P_yy (→ C12).
   - Uniaxial x-strain on [110]-rotated crystal → P_xx (→ E_110).
     C44 = E_110 - (C11+C12)/2  (from cubic rotation: C'_11 = (C11+C12)/2 + C44).
3. Linear regression (numpy.polyfit) on stress vs strain; slope × (-1 bar→GPa).

The [110] rotation method is used for C44 to avoid the off-diagonal virial (pxy)
which is unreliable for EAM potentials in this LAMMPS build. Only diagonal
stresses (pxx, pyy) are read throughout.

CLI
---
python elastic_tool.py --composition Al --potential /app/potentials/Al.eam.alloy
                       [--supercell_size 4]

Output (stdout, JSON)
---------------------
{"composition": "Al", "C11": 114.3, "C12": 61.8, "C44": 31.6,
 "runtime_seconds": 52.1, "status": "ok"}

On error:
{"composition": "Al", "C11": null, "C12": null, "C44": null,
 "runtime_seconds": null, "status": "error", "error": "<message>"}
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BAR_TO_GPA = 1e-4

STRAIN_VALUES = [-0.01, -0.005, 0.005, 0.01]

ELEMENT_DATA = {
    "Al": {"structure": "fcc", "a0": 4.05},
    "Cu": {"structure": "fcc", "a0": 3.615},
    "Ni": {"structure": "fcc", "a0": 3.52},
    "Fe": {"structure": "bcc", "a0": 2.87},
    "W":  {"structure": "bcc", "a0": 3.165},
    "Mo": {"structure": "bcc", "a0": 3.147},
}

# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace with composition, potential, supercell_size.
    """
    parser = argparse.ArgumentParser(
        description="Compute elastic constants via LAMMPS EAM simulation."
    )
    parser.add_argument("--composition", required=True,
                        help="Element symbol (Al, Cu, Ni, Fe, W, Mo).")
    parser.add_argument("--potential", required=True,
                        help="Absolute path to EAM/alloy potential file.")
    parser.add_argument("--supercell_size", type=int, default=4,
                        help="Unit cells per axis (default: 4).")
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# LAMMPS helpers
# ---------------------------------------------------------------------------

def _build_lammps(composition: str, pot_path: str, supercell_size: int):
    """Instantiate and configure a LAMMPS object.

    Preconditions:
        - ``lammps`` Python package is importable (inside container).
        - ``pot_path`` points to a valid EAM/alloy potential file.

    Args:
        composition: Element symbol.
        pot_path: Path to the EAM/alloy potential file.
        supercell_size: Number of unit cells per axis.

    Returns:
        Configured ``lammps.lammps`` instance.
    """
    from lammps import lammps  # type: ignore

    elem = ELEMENT_DATA[composition]
    structure = elem["structure"]
    a0 = elem["a0"]
    n = supercell_size

    lmp = lammps(cmdargs=["-screen", "none", "-log", "none"])
    lmp.commands_list([
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        f"lattice {structure} {a0}",
        f"region box block 0 {n} 0 {n} 0 {n}",
        "create_box 1 box",
        f"create_atoms 1 box",
        f"pair_style {'eam/fs' if pot_path.endswith('.eam.fs') else 'eam/alloy'}",
        f"pair_coeff * * {pot_path} {composition}",
        "mass 1 1.0",
        "neighbor 2.0 bin",
        "neigh_modify every 1 delay 0 check yes",
    ])
    return lmp


def _relax_box(lmp) -> float:
    """Run FIRE box relaxation and return equilibrium lx.

    Args:
        lmp: Configured LAMMPS instance (atoms already created).

    Returns:
        Equilibrium box length lx_eq in Angstroms.

    Postconditions:
        - The LAMMPS box is at its equilibrium geometry.
        - Atoms are at their minimum-energy positions.
    """
    lmp.commands_list([
        "min_style cg",
        "fix 1 all box/relax iso 0.0 vmax 0.001",
        "minimize 1e-10 1e-10 10000 100000",
        "unfix 1",
        "change_box all triclinic",
    ])
    return lmp.get_thermo("lx")


def _stress_for_strain(composition: str, pot_path: str, supercell_size: int,
                       strain_type: str, strain_value: float) -> dict:
    """Build a fresh LAMMPS instance, relax it, apply one strain, and return stresses.

    Each call is fully self-contained: a new LAMMPS object is created, the box
    is relaxed to equilibrium, exactly one strain state is applied, stresses are
    read, and the instance is immediately closed.  No box restoration is needed
    because the instance is discarded after a single measurement.

    Using a fresh instance per strain state is the community-standard approach
    (used in the original AtomAgents code and most published LAMMPS elastic
    scripts) and avoids neighbour-list or triclinic-state corruption that
    accumulates when a single instance is reused across multiple strain states.

    For uniaxial e11 strain: atoms are relaxed (CG) after the box is scaled to
    relieve transverse forces while keeping the applied axial strain.

    For shear e12 strain: stresses are read directly after affine deformation
    without atomic relaxation. For a monatomic Bravais lattice (FCC/BCC) all
    atoms are equivalent so there are no internal degrees of freedom to relax;
    minimisation would cause atomic shuffling and corrupt C44.

    Args:
        composition: Element symbol (must be a key in ``ELEMENT_DATA``).
        pot_path: Absolute path to the EAM/alloy or EAM/fs potential file.
        supercell_size: Number of unit cells per axis.
        strain_type: ``"e11"`` for uniaxial or ``"e12"`` for engineering shear.
        strain_value: Dimensionless strain magnitude ε.

    Returns:
        Dict with stress components (bar): ``pxx``, ``pyy``, ``pxy``.

    Postconditions:
        - The LAMMPS instance is closed before returning.
    """
    lmp = _build_lammps(composition, pot_path, supercell_size)
    lx_eq = _relax_box(lmp)

    if strain_type == "e11":
        lmp.command(f"change_box all x scale {1.0 + strain_value} remap")
        lmp.commands_list([
            "min_style cg",
            "minimize 1e-8 1e-8 5000 50000",
        ])
    else:
        lmp.command(f"change_box all xy final {strain_value * lx_eq} remap")
        lmp.command("run 0")

    result = {
        "pxx": lmp.get_thermo("pxx"),
        "pyy": lmp.get_thermo("pyy"),
        "pxy": lmp.get_thermo("pxy"),
    }
    lmp.close()
    return result


def _c44_stress_110(composition: str, pot_path: str, supercell_size: int,
                    strain_value: float) -> float:
    """Build a fresh LAMMPS instance with [110] orientation, apply x-strain, return pxx.

    For a cubic crystal rotated so x‖[110], y‖[-110], z‖[001], the rotated
    stiffness component C'_11 = (C11 + C12)/2 + C44.  A uniaxial x-strain ε
    therefore gives pxx = -C'_11 * ε, so C44 = -pxx/ε - (C11 + C12)/2.

    This approach reads only the diagonal stress pxx and never uses pxy,
    avoiding the off-diagonal virial issue seen with the xy-shear method.

    Args:
        composition: Element symbol (must be a key in ``ELEMENT_DATA``).
        pot_path: Absolute path to the EAM/alloy or EAM/fs potential file.
        supercell_size: Number of unit cells per axis.
        strain_value: Dimensionless uniaxial strain magnitude ε.

    Returns:
        pxx stress component in bar after x-strain in the [110] frame.

    Postconditions:
        - The LAMMPS instance is closed before returning.
    """
    from lammps import lammps  # type: ignore

    elem = ELEMENT_DATA[composition]
    n = supercell_size
    lmp = lammps(cmdargs=["-screen", "none", "-log", "none"])
    lmp.commands_list([
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        f"lattice {elem['structure']} {elem['a0']}"
        " orient x 1 1 0 orient y -1 1 0 orient z 0 0 1",
        f"region box block 0 {n} 0 {n} 0 {n}",
        "create_box 1 box",
        "create_atoms 1 box",
        f"pair_style {'eam/fs' if pot_path.endswith('.eam.fs') else 'eam/alloy'}",
        f"pair_coeff * * {pot_path} {composition}",
        "mass 1 1.0",
        "neighbor 2.0 bin",
        "neigh_modify every 1 delay 0 check yes",
    ])
    lmp.commands_list([
        "min_style cg",
        "fix 1 all box/relax iso 0.0 vmax 0.001",
        "minimize 1e-10 1e-10 10000 100000",
        "unfix 1",
    ])
    lmp.command(f"change_box all x scale {1.0 + strain_value} remap")
    lmp.commands_list([
        "min_style cg",
        "minimize 1e-8 1e-8 5000 50000",
    ])
    pxx = lmp.get_thermo("pxx")
    lmp.close()
    return pxx


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_elastic_constants(composition: str, pot_path: str,
                               supercell_size: int) -> dict:
    """Compute C11, C12, C44 (GPa) for the given element via EAM LAMMPS.

    Implements the Ghafarollahi & Buehler AtomAgents stress-strain method.

    Args:
        composition: Element symbol (must be in ``ELEMENT_DATA``).
        pot_path: Absolute path to the EAM/alloy potential file.
        supercell_size: Number of unit cells per axis.

    Returns:
        Dict with keys ``C11``, ``C12``, ``C44`` (all floats in GPa).

    Raises:
        KeyError: If ``composition`` is not in ``ELEMENT_DATA``.
        RuntimeError: If LAMMPS encounters an error.

    Preconditions:
        - ``composition`` is a key in ``ELEMENT_DATA``.
        - ``pot_path`` is a valid, readable EAM/alloy potential file.

    Postconditions:
        - LAMMPS instance is closed/cleaned up.
        - Returned values are in GPa.
    """
    if composition not in ELEMENT_DATA:
        raise KeyError(
            f"Unknown element '{composition}'. "
            f"Supported: {list(ELEMENT_DATA.keys())}"
        )

    pxx_e11, pyy_e11, pxx_110 = [], [], []

    for eps in STRAIN_VALUES:
        s = _stress_for_strain(composition, pot_path, supercell_size, "e11", eps)
        pxx_e11.append(s["pxx"])
        pyy_e11.append(s["pyy"])

        pxx_110.append(_c44_stress_110(composition, pot_path, supercell_size, eps))

    strains = np.array(STRAIN_VALUES)
    # σ = −P → elastic constant = slope of (−P) vs ε, converted bar → GPa
    C11 = float(np.polyfit(strains, [-p for p in pxx_e11], 1)[0] * BAR_TO_GPA)
    C12 = float(np.polyfit(strains, [-p for p in pyy_e11], 1)[0] * BAR_TO_GPA)
    # C'_11 ([110] frame) = (C11+C12)/2 + C44  →  C44 = E_110 - (C11+C12)/2
    E_110 = float(np.polyfit(strains, [-p for p in pxx_110], 1)[0] * BAR_TO_GPA)
    C44 = E_110 - (C11 + C12) / 2

    return {"C11": round(C11, 4), "C12": round(C12, 4), "C44": round(C44, 4)}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    """CLI entry point — parse args, run calculation, print JSON to stdout.

    Preconditions:
        - Called from within the Docker container where LAMMPS is installed.
        - ``--composition`` and ``--potential`` are provided.

    Postconditions:
        - Prints a single JSON line to stdout.
        - Exits 0 on success, 1 on error.
    """
    args = parse_args(argv)
    t0 = time.perf_counter()
    try:
        results = compute_elastic_constants(
            args.composition, args.potential, args.supercell_size
        )
        output = {
            "composition": args.composition,
            "C11": results["C11"],
            "C12": results["C12"],
            "C44": results["C44"],
            "runtime_seconds": round(time.perf_counter() - t0, 2),
            "status": "ok",
        }
        print(json.dumps(output))
        sys.exit(0)
    except Exception as exc:
        output = {
            "composition": args.composition,
            "C11": None,
            "C12": None,
            "C44": None,
            "runtime_seconds": round(time.perf_counter() - t0, 2),
            "status": "error",
            "error": str(exc),
        }
        print(json.dumps(output))
        sys.exit(1)


if __name__ == "__main__":
    main()
