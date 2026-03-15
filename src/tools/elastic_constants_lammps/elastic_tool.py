"""In-container LAMMPS script for computing elastic constants of FCC/BCC metals.

Method (AtomAgents Voigt perturbation):
1. Copy the five AtomAgents input scripts (in.elastic, displace.mod, init.mod,
   potential.mod, compliance.py) verbatim from /app/scripts/ into a temporary
   directory.
2. Patch the copies minimally:
   - init.mod: replace 4 Si-specific lines (lattice parameter, lattice type,
     region geometry, atomic mass).
   - potential.mod: replace 2 Si-specific lines (pair_style, pair_coeff).
   - in.elastic: append 3 print statements so LAMMPS emits C11cubic, C12cubic,
     C44cubic to log.lammps (these variables are already computed by in.elastic
     but not printed in the original).
3. Run ``lmp -in in.elastic`` via subprocess.
4. Parse the 3 printed values from log.lammps.

LAMMPS performs all 6 finite-strain perturbations and all cubic averaging
internally.  Python does no physics — it only patches input files and reads
the final output lines.

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
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

# ---------------------------------------------------------------------------
# Element data
# ---------------------------------------------------------------------------

ELEMENT_DATA = {
    "Al": {"structure": "fcc", "a0": 4.05, "mass": 26.982},
    "Cu": {"structure": "fcc", "a0": 3.615, "mass": 63.546},
    "Ni": {"structure": "fcc", "a0": 3.52, "mass": 58.693},
    "Fe": {"structure": "bcc", "a0": 2.87, "mass": 55.845},
    "W": {"structure": "bcc", "a0": 3.165, "mass": 183.840},
    "Mo": {"structure": "bcc", "a0": 3.147, "mass": 95.960},
}

# ---------------------------------------------------------------------------
# AtomAgents patch targets — update here if scripts/0_codes/ files change
# ---------------------------------------------------------------------------

_LAMMPS_TIMEOUT_S = 600  # seconds before subprocess.run raises TimeoutExpired
_GPa_PRECISION = 4  # decimal places for returned GPa values
_LOG_TAIL_LINES = 50  # lines of log.lammps included in error messages

_SCRIPT_NAMES = (
    "in.elastic",
    "displace.mod",
    "init.mod",
    "potential.mod",
    "compliance.py",
)

# Each entry: (exact string in source file, replacement template for .format(**subs))
_INIT_MOD_PATCHES: list[tuple[str, str]] = [
    ("variable a equal 5.43", "variable a equal {a0}"),
    ("lattice         diamond $a", "lattice {structure} $a"),
    (
        "region\t\tbox prism 0 2.0 0 3.0 0 4.0 0.0 0.0 0.0",
        "region box prism 0 {n} 0 {n} 0 {n} 0.0 0.0 0.0",
    ),
    ("mass 1 1.0e-20", "mass 1 {mass}"),
]

_POTENTIAL_MOD_PATCHES: list[tuple[str, str]] = [
    ("pair_style\tsw", "pair_style {pair_style}"),
    ("pair_coeff * * Si.sw Si", "pair_coeff * * {pot_path} {composition}"),
]

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
    parser.add_argument(
        "--composition", required=True, help="Element symbol (Al, Cu, Ni, Fe, W, Mo)."
    )
    parser.add_argument(
        "--potential", required=True, help="Absolute path to EAM/alloy potential file."
    )
    parser.add_argument(
        "--supercell_size",
        type=int,
        default=4,
        help="Unit cells per axis (default: 4).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# AtomAgents script helpers
# ---------------------------------------------------------------------------


def _copy_scripts(tmpdir: str) -> None:
    """Copy all 5 AtomAgents scripts verbatim from /app/scripts/ into tmpdir.

    Args:
        tmpdir: Absolute path to the temporary working directory.
    """
    for fname in _SCRIPT_NAMES:
        shutil.copy(os.path.join("/app/scripts", fname), os.path.join(tmpdir, fname))


def _patch_init_mod(tmpdir: str, composition: str, supercell_size: int) -> None:
    """Patch the verbatim init.mod copy for the given element.

    Iterates over _INIT_MOD_PATCHES. Each old string is asserted present
    before replacement so drift from the original is caught immediately.
    Only 4 Si-specific lines change; all other content is kept verbatim.

    Note:
        The region replacement forces a cubic block geometry. This is correct
        for all FCC and BCC metals in ELEMENT_DATA but would need adjustment
        for non-cubic crystal systems.

    Args:
        tmpdir: Absolute path to the temp working directory containing init.mod.
        composition: Element symbol (must be a key in ``ELEMENT_DATA``).
        supercell_size: Number of unit cells per axis (n × n × n box).

    Raises:
        RuntimeError: If any expected original string is not found in init.mod,
            indicating the file in scripts/0_codes/ has changed unexpectedly.
    """
    elem = ELEMENT_DATA[composition]
    subs = {**elem, "n": supercell_size}
    path = os.path.join(tmpdir, "init.mod")
    with open(path) as f:
        content = f.read()
    for old, new_template in _INIT_MOD_PATCHES:
        new = new_template.format(**subs)
        if old not in content:
            raise RuntimeError(
                f"_patch_init_mod: expected string not found in init.mod: {old!r}\n"
                "Check that scripts/0_codes/init.mod has not been modified."
            )
        content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)


def _patch_potential_files(tmpdir: str, composition: str, pot_path: str) -> None:
    """Patch the verbatim potential.mod copy and produce potential.inp.

    Iterates over _POTENTIAL_MOD_PATCHES. Each old string is asserted present
    before replacement so drift from the original is caught immediately.
    potential.inp is produced via shutil.copy from the already-patched
    potential.mod, so both files are guaranteed identical and only one
    write operation occurs.

    Args:
        tmpdir: Absolute path to the temp working directory containing potential.mod.
        composition: Element symbol (e.g. ``"Al"``).
        pot_path: Absolute path to the EAM/alloy or EAM/fs potential file.

    Raises:
        RuntimeError: If any expected original string is not found in potential.mod,
            indicating the file in scripts/0_codes/ has changed unexpectedly.
    """
    pair_style = "eam/fs" if pot_path.endswith(".eam.fs") else "eam/alloy"
    subs = {"pair_style": pair_style, "pot_path": pot_path, "composition": composition}
    path = os.path.join(tmpdir, "potential.mod")
    with open(path) as f:
        content = f.read()
    for old, new_template in _POTENTIAL_MOD_PATCHES:
        new = new_template.format(**subs)
        if old not in content:
            raise RuntimeError(
                f"_patch_potential_files: expected string not found in potential.mod: {old!r}\n"
                "Check that scripts/0_codes/potential.mod has not been modified."
            )
        content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    # shutil.copy instead of a second open/write — guarantees identical content
    shutil.copy(path, os.path.join(tmpdir, "potential.inp"))


def _patch_in_elastic(tmpdir: str) -> None:
    """Append C11cubic/C12cubic/C44cubic print lines to the in.elastic copy.

    in.elastic already computes C11cubic, C12cubic, C44cubic as LAMMPS
    variables (lines 105-107) but never prints them. This append makes them
    available in log.lammps for parsing. All existing content is kept verbatim.

    Args:
        tmpdir: Absolute path to the temp working directory containing in.elastic.
    """
    path = os.path.join(tmpdir, "in.elastic")
    with open(path, "a") as f:
        f.write('\nprint "C11cubic = ${C11cubic} ${cunits}"\n')
        f.write('print "C12cubic = ${C12cubic} ${cunits}"\n')
        f.write('print "C44cubic = ${C44cubic} ${cunits}"\n')


def _patch_all(
    tmpdir: str, composition: str, pot_path: str, supercell_size: int
) -> None:
    """Apply all patches to the tmpdir copies in the correct order.

    Args:
        tmpdir: Absolute path to the temp working directory.
        composition: Element symbol (must be a key in ``ELEMENT_DATA``).
        pot_path: Absolute path to the EAM potential file.
        supercell_size: Number of unit cells per axis.
    """
    _patch_init_mod(tmpdir, composition, supercell_size)
    _patch_potential_files(tmpdir, composition, pot_path)
    _patch_in_elastic(tmpdir)


def _run_lammps(tmpdir: str) -> None:
    """Execute ``lmp -in in.elastic`` in tmpdir; raise RuntimeError on failure.

    On non-zero exit, appends the last _LOG_TAIL_LINES lines of log.lammps
    (if it exists) to the error message to aid debugging.

    Args:
        tmpdir: Absolute path to directory containing the patched input scripts.

    Raises:
        RuntimeError: If LAMMPS exits with a non-zero return code.
    """
    result = subprocess.run(
        ["lmp", "-in", "in.elastic"],
        cwd=tmpdir,
        capture_output=True,
        text=True,
        timeout=_LAMMPS_TIMEOUT_S,
    )
    if result.returncode != 0:
        log_tail = ""
        log_path = os.path.join(tmpdir, "log.lammps")
        if os.path.exists(log_path):
            with open(log_path) as f:
                lines = f.readlines()
            log_tail = "".join(lines[-_LOG_TAIL_LINES:])
        raise RuntimeError(
            f"LAMMPS exited with code {result.returncode}.\n"
            f"stderr: {result.stderr[:500]}\n"
            f"log.lammps (last {_LOG_TAIL_LINES} lines):\n{log_tail}"
        )


def _parse_log_lammps(log_path: str) -> dict:
    """Parse C11cubic, C12cubic, C44cubic from log.lammps.

    Looks for lines printed by the 3 appended print statements:
        C11cubic = <value> GPa
        C12cubic = <value> GPa
        C44cubic = <value> GPa

    The regex anchors to the literal `` GPa`` suffix so a changed unit string
    or a malformed number fails loudly rather than producing a silent garbage
    value. Each parsed value is also bounds-checked against the physical range
    (0, 2000) GPa — any real cubic metal falls well within this range.

    Args:
        log_path: Absolute path to the log.lammps file.

    Returns:
        Dict with keys ``C11``, ``C12``, ``C44`` (floats in GPa).

    Raises:
        RuntimeError: If any value is not found in the log or outside (0, 2000) GPa.
    """
    with open(log_path) as f:
        text = f.read()
    vals = {}
    for key in ("C11cubic", "C12cubic", "C44cubic"):
        m = re.search(rf"{key} = ([\d.eE+\-]+) GPa", text)
        if not m:
            raise RuntimeError(
                f"log.lammps: '{key}' not found — check LAMMPS run completed successfully"
            )
        val = float(m.group(1))
        if not (0 < val < 2000):
            raise RuntimeError(
                f"log.lammps: {key}={val} GPa is outside physical range (0, 2000). "
                "Check LAMMPS run for numerical instability or wrong potential."
            )
        vals[key] = val
    return {
        "C11": round(vals["C11cubic"], _GPa_PRECISION),
        "C12": round(vals["C12cubic"], _GPa_PRECISION),
        "C44": round(vals["C44cubic"], _GPa_PRECISION),
    }


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def compute_elastic_constants(
    composition: str, pot_path: str, supercell_size: int
) -> dict:
    """Compute C11, C12, C44 (GPa) for the given element via AtomAgents LAMMPS scripts.

    Copies the five AtomAgents input scripts verbatim into a temporary directory,
    applies minimal targeted patches (lattice, potential, 3 print lines), runs
    LAMMPS, and parses C11cubic/C12cubic/C44cubic from log.lammps. All physics
    — the 6 finite-strain perturbations and the cubic averaging — is performed
    entirely inside LAMMPS.

    Args:
        composition: Element symbol (must be in ``ELEMENT_DATA``).
        pot_path: Absolute path to the EAM/alloy or EAM/fs potential file.
        supercell_size: Number of unit cells per axis.

    Returns:
        Dict with keys ``C11``, ``C12``, ``C44`` (all floats in GPa).

    Raises:
        KeyError: If ``composition`` is not in ``ELEMENT_DATA``.
        RuntimeError: If LAMMPS exits non-zero or parsed values are unphysical.
    """
    if composition not in ELEMENT_DATA:
        raise KeyError(
            f"Unknown element '{composition}'. Supported: {list(ELEMENT_DATA.keys())}"
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        _copy_scripts(tmpdir)
        _patch_all(tmpdir, composition, pot_path, supercell_size)
        _run_lammps(tmpdir)
        return _parse_log_lammps(os.path.join(tmpdir, "log.lammps"))


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
