"""Diagnostic: compare C44 from direct pxy shear vs current [110] rotation method.

Run inside the Docker container via volume mount (no rebuild needed):

    docker run --rm \\
      -v "/host/path/tests/c44_pxy_diagnostic.py:/app/c44_pxy_diagnostic.py" \\
      elastic-lammps-tool:latest \\
      python /app/c44_pxy_diagnostic.py

Potentials are baked into the image at /app/potentials/.

Methods compared
----------------
pxy direct  : apply engineering shear xy = γ * Lx_eq via change_box xy final,
              run 0 (no atomic relaxation), read pxy, fit slope of (-pxy) vs γ.
              C44 = slope_bar * 1e-4  [GPa]

[110] method : uniaxial x-strain on [110]-oriented crystal, read pxx, fit slope.
              C44 = E_110 - (C11 + C12) / 2  (current production formula)
"""
from __future__ import annotations

import sys

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ELEMENTS = {
    "Al": {"structure": "fcc", "a0": 4.05, "pot": "/app/potentials/Al.eam.alloy"},
    "Ni": {"structure": "fcc", "a0": 3.52, "pot": "/app/potentials/Ni.eam.alloy"},
}

SUPERCELL = 4
STRAIN_VALUES = [-0.01, -0.005, 0.005, 0.01]
BAR_TO_GPA = 1e-4

EXPECTED_C44 = {
    "Al": "28–36 GPa",
    "Ni": "118–130 GPa",
}

# ---------------------------------------------------------------------------
# LAMMPS helpers (self-contained, no imports from elastic_tool.py)
# ---------------------------------------------------------------------------

def _build_lammps(composition: str, pot_path: str, supercell_size: int,
                   orient_x=None, orient_y=None, orient_z=None):
    """Instantiate and configure a LAMMPS object for the given composition.

    Args:
        composition: Element symbol (must be a key in ``ELEMENTS``).
        pot_path: Absolute path to the EAM/alloy or EAM/fs potential file.
        supercell_size: Number of unit cells per axis.
        orient_x: Optional lattice orientation string for x-axis (e.g. ``"1 1 0"``).
        orient_y: Optional lattice orientation string for y-axis.
        orient_z: Optional lattice orientation string for z-axis.

    Returns:
        Configured ``lammps.lammps`` instance ready for box relaxation.
    """
    from lammps import lammps  # type: ignore

    elem = ELEMENTS[composition]
    n = supercell_size
    lmp = lammps(cmdargs=["-screen", "none", "-log", "none"])

    lattice_cmd = f"lattice {elem['structure']} {elem['a0']}"
    if orient_x:
        lattice_cmd += f" orient x {orient_x} orient y {orient_y} orient z {orient_z}"

    lmp.commands_list([
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        lattice_cmd,
        f"region box block 0 {n} 0 {n} 0 {n}",
        "create_box 1 box",
        "create_atoms 1 box",
        f"pair_style {'eam/fs' if pot_path.endswith('.eam.fs') else 'eam/alloy'}",
        f"pair_coeff * * {pot_path} {composition}",
        "mass 1 1.0",
        "neighbor 2.0 bin",
        "neigh_modify every 1 delay 0 check yes",
    ])
    return lmp


def _relax_and_get_lx(lmp) -> float:
    """Run CG box relaxation and return the equilibrium x box length.

    Args:
        lmp: Configured LAMMPS instance (atoms already created).

    Returns:
        Equilibrium box length lx in Angstroms after ISO pressure relaxation.

    Postconditions:
        - Box is converted to triclinic and at its minimum-energy geometry.
    """
    lmp.commands_list([
        "min_style cg",
        "fix 1 all box/relax iso 0.0 vmax 0.001",
        "minimize 1e-10 1e-10 10000 100000",
        "unfix 1",
        "change_box all triclinic",
    ])
    return lmp.get_thermo("lx")


# ---------------------------------------------------------------------------
# pxy direct method
# ---------------------------------------------------------------------------

def c44_via_pxy(composition: str) -> float:
    """Compute C44 (GPa) via direct pxy shear — fresh instance per strain."""
    pot_path = ELEMENTS[composition]["pot"]
    pxy_list = []

    for gamma in STRAIN_VALUES:
        lmp = _build_lammps(composition, pot_path, SUPERCELL)
        lx_eq = _relax_and_get_lx(lmp)
        lmp.command(f"change_box all xy final {gamma * lx_eq} remap")
        lmp.command("run 0")
        pxy_list.append(lmp.get_thermo("pxy"))
        lmp.close()

    strains = np.array(STRAIN_VALUES)
    slope_bar = np.polyfit(strains, [-p for p in pxy_list], 1)[0]
    return float(slope_bar * BAR_TO_GPA)


# ---------------------------------------------------------------------------
# [110] method  (mirrors production _c44_stress_110 + compute_elastic_constants)
# ---------------------------------------------------------------------------

def c44_via_110(composition: str) -> float:
    """Compute C44 (GPa) via [110]-rotation uniaxial method (current production)."""
    pot_path = ELEMENTS[composition]["pot"]

    pxx_e11, pyy_e11, pxx_110 = [], [], []

    for eps in STRAIN_VALUES:
        # e11 → C11, C12
        lmp = _build_lammps(composition, pot_path, SUPERCELL)
        _relax_and_get_lx(lmp)
        lmp.command(f"change_box all x scale {1.0 + eps} remap")
        lmp.commands_list(["min_style cg", "minimize 1e-8 1e-8 5000 50000"])
        pxx_e11.append(lmp.get_thermo("pxx"))
        pyy_e11.append(lmp.get_thermo("pyy"))
        lmp.close()

        # [110] uniaxial → E_110
        lmp = _build_lammps(composition, pot_path, SUPERCELL,
                             orient_x="1 1 0", orient_y="-1 1 0", orient_z="0 0 1")
        lmp.commands_list([
            "min_style cg",
            "fix 1 all box/relax iso 0.0 vmax 0.001",
            "minimize 1e-10 1e-10 10000 100000",
            "unfix 1",
        ])
        lmp.command(f"change_box all x scale {1.0 + eps} remap")
        lmp.commands_list(["min_style cg", "minimize 1e-8 1e-8 5000 50000"])
        pxx_110.append(lmp.get_thermo("pxx"))
        lmp.close()

    strains = np.array(STRAIN_VALUES)
    C11 = float(np.polyfit(strains, [-p for p in pxx_e11], 1)[0] * BAR_TO_GPA)
    C12 = float(np.polyfit(strains, [-p for p in pyy_e11], 1)[0] * BAR_TO_GPA)
    E110 = float(np.polyfit(strains, [-p for p in pxx_110], 1)[0] * BAR_TO_GPA)
    return E110 - (C11 + C12) / 2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nC44 diagnostic: direct pxy shear vs [110] rotation method\n")
    print(f"{'Element':<8} {'pxy method':>12} {'[110] method':>14} {'Expected range':>16}")
    print("-" * 54)

    for comp in ["Al", "Ni"]:
        try:
            c44_pxy = c44_via_pxy(comp)
            pxy_str = f"{c44_pxy:.1f} GPa"
        except Exception as exc:
            pxy_str = f"ERROR: {exc}"
            c44_pxy = None

        try:
            c44_110 = c44_via_110(comp)
            s110_str = f"{c44_110:.1f} GPa"
        except Exception as exc:
            s110_str = f"ERROR: {exc}"

        print(f"{comp:<8} {pxy_str:>12} {s110_str:>14} {EXPECTED_C44[comp]:>16}")

    print()
    sys.exit(0)
