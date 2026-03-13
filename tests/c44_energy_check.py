"""Standalone diagnostic: compute C44 via energy second-derivative for Al and Ni.

Run inside the Docker container via volume mount:

    docker run --rm \
      -v "/host/path/tests/c44_energy_check.py:/app/c44_energy_check.py" \
      elastic-lammps-tool:latest \
      python /app/c44_energy_check.py

No image rebuild needed — potentials are baked in.

Method
------
For engineering shear strain γ applied via change_box xy:

    C44 = 2 * (d²E/dγ²) / V₀

where E is potential energy (eV) and V₀ is equilibrium volume (Å³).
No minimization is performed after shearing — for a monatomic Bravais
lattice (FCC/BCC) the crystal has no internal degrees of freedom, so
minimizing would shuffle atoms back and corrupt the harmonic response.
"""
from __future__ import annotations

import sys

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ELEMENTS = {
    "Al": {"structure": "fcc", "a0": 4.05,  "pot": "/app/potentials/Al.eam.alloy"},
    "Ni": {"structure": "fcc", "a0": 3.52,  "pot": "/app/potentials/Ni.eam.alloy"},
}

SUPERCELL = 3
SHEAR_STRAINS = np.linspace(-0.02, 0.02, 9)  # 9 points, includes γ=0

EV_PER_ANG3_TO_GPA = 160.2174  # 1 eV/Å³ = 160.2174 GPa

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def get_energy_at_shear(gamma: float, composition: str, pot_path: str,
                         supercell_size: int) -> tuple[float, float]:
    """Return (pe_eV, V0_ang3) for one shear strain value.

    Fresh LAMMPS instance per call — no state reuse.
    Box is relaxed at γ=0 first; shear is then applied via change_box xy;
    energy is read after run 0 with NO atomic minimization.
    """
    from lammps import lammps  # type: ignore

    elem = ELEMENTS[composition]
    n = supercell_size

    lmp = lammps(cmdargs=["-screen", "none", "-log", "none"])
    lmp.commands_list([
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        f"lattice {elem['structure']} {elem['a0']}",
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
        "change_box all triclinic",
    ])

    lx = lmp.get_thermo("lx")
    ly = lmp.get_thermo("ly")
    lz = lmp.get_thermo("lz")
    V0 = lx * ly * lz

    if gamma != 0.0:
        lmp.command(f"change_box all xy final {gamma * lx} remap")

    lmp.command("run 0")
    pe = lmp.get_thermo("pe")
    lmp.close()
    return pe, V0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_c44_energy(composition: str) -> float:
    """Compute C44 (GPa) via energy second-derivative for the given element."""
    energies = []
    v0 = None
    for gamma in SHEAR_STRAINS:
        pe, V0 = get_energy_at_shear(float(gamma), composition,
                                      ELEMENTS[composition]["pot"], SUPERCELL)
        energies.append(pe)
        if v0 is None:
            v0 = V0

    energies = np.array(energies)
    e0 = energies[len(energies) // 2]  # γ=0 is the middle point
    delta_e = energies - e0

    coeffs = np.polyfit(SHEAR_STRAINS, delta_e, 2)
    return 2.0 * coeffs[0] / v0 * EV_PER_ANG3_TO_GPA


if __name__ == "__main__":
    stress_110 = {"Al": 32.1, "Ni": 137.5}
    literature  = {"Al": 31.6, "Ni": 123.0}

    print("\nC44 cross-check: energy second-derivative vs [110] stress vs literature\n")
    print(f"{'Element':<8} {'[110] stress':>14} {'Energy 2nd-deriv':>18} {'Literature':>12}")
    print("-" * 56)

    for comp in ["Al", "Ni"]:
        try:
            c44_energy = compute_c44_energy(comp)
            energy_str = f"{c44_energy:.1f} GPa"
        except Exception as exc:
            energy_str = f"ERROR: {exc}"
            c44_energy = None

        print(f"{comp:<8} {stress_110[comp]:>12.1f} GPa "
              f"{energy_str:>18} {literature[comp]:>10.1f} GPa")

    print()
    sys.exit(0)
