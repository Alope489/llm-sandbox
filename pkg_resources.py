"""
Local compatibility shim for environments where ``pkg_resources`` is missing.

ChemCrow imports ``pkg_resources`` and uses only ``pkg_resources.resource_filename``
to locate package data files (e.g. ``chemcrow/data/chem_wep.csv``).

This shim implements just enough of that API for ChemCrow to run.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path


def resource_filename(package_or_requirement: str, resource_name: str) -> str:
    """
    Resolve an installed package's resource path to an absolute filesystem path.
    """
    spec = find_spec(package_or_requirement)
    if spec is None:
        raise ModuleNotFoundError(f"Cannot find package: {package_or_requirement}")

    # For packages, submodule_search_locations exists and points to the directory.
    if spec.submodule_search_locations:
        base_dir = Path(list(spec.submodule_search_locations)[0])
    elif spec.origin:
        base_dir = Path(spec.origin).parent
    else:
        raise FileNotFoundError(f"Cannot resolve package directory for: {package_or_requirement}")

    return str(base_dir / resource_name)

