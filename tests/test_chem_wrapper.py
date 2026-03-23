"""Unit tests for chemistry wrapper routing behavior."""

from src.multi.chemistry import chem_wrapper
from src.multi.chemistry.chem_driver import ChemistryResult, ChemistryRunResult


def test_looks_like_smiles_router_check():
    """Router detects compact SMILES-like strings and rejects names."""
    assert chem_wrapper._looks_like_smiles("CCO") is True
    assert chem_wrapper._looks_like_smiles("c1ccccc1") is True
    assert chem_wrapper._looks_like_smiles("sodium chloride") is False


def test_run_chemistry_task_marks_smiles_route(monkeypatch):
    """Wrapper labels routed SMILES tasks."""

    class FakeDriver:
        def run_chemistry_sim(self, _query):
            return ChemistryRunResult(
                output=ChemistryResult(
                    query="CCO",
                    route="smiles",
                    cid=None,
                    molecular_weight=None,
                    molecular_formula=None,
                    is_valid_smiles=True,
                    found=True,
                ),
                step_start=1.0,
                step_end=2.0,
                latency_seconds=1.0,
            )

    monkeypatch.setattr(chem_wrapper, "_get_driver", lambda: FakeDriver())
    result = chem_wrapper.run_chemistry_task("CCO")
    assert result.route == "smiles"
    assert result.is_valid_smiles is True


def test_run_chemistry_task_marks_name_lookup_route(monkeypatch):
    """Wrapper labels routed common-name lookup tasks."""

    class FakeDriver:
        def run_chemistry_sim(self, _query):
            return ChemistryRunResult(
                output=ChemistryResult(
                    query="acetone",
                    route="name_lookup",
                    cid=180,
                    molecular_weight=58.08,
                    molecular_formula="C3H6O",
                    is_valid_smiles=False,
                    found=True,
                ),
                step_start=1.0,
                step_end=2.0,
                latency_seconds=1.0,
            )

    monkeypatch.setattr(chem_wrapper, "_get_driver", lambda: FakeDriver())
    result = chem_wrapper.run_chemistry_task("acetone")
    assert result.route == "name_lookup"
    assert result.cid == 180
    assert result.molecular_formula == "C3H6O"
    assert result.molecular_weight == 58.08
