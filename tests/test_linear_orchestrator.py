"""Tests for src.linear.orchestrator run() with mocked extract, process, summarize."""
import os
import sys

import pytest
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
load_dotenv()


def _minimal_extraction_dict():
    return {
        "material_system": {"material_name": "Ni", "composition": [], "phase_type": None, "microstructure": {"grain_size_nm": None, "porosity_percent": None, "crystal_structure": None}},
        "processing_conditions": {"synthesis_method": None, "heat_treatment": {"temperature_K": None, "duration_hours": None, "cooling_rate_K_per_min": None}, "pressure_GPa": None},
        "simulation_parameters": {"temperature_range_K": {"min": None, "max": None, "step": None}, "strain_rate_s_inverse": None, "boundary_conditions": None, "model_type": None},
        "computed_properties": {"thermal_conductivity_W_per_mK": None, "yield_strength_MPa": None, "youngs_modulus_GPa": None, "poissons_ratio": None, "thermal_expansion_coefficient_per_K": None, "specific_heat_J_per_kgK": None, "electrical_conductivity_S_per_m": None, "density_kg_per_m3": None},
        "uncertainty_estimates": {"property_uncertainty_percent": None, "model_confidence_level": None},
    }


def test_run_returns_summary_extraction_processing(monkeypatch):
    from src.linear import orchestrator

    extraction = _minimal_extraction_dict()
    monkeypatch.setattr(orchestrator, "extract", lambda t: extraction)
    monkeypatch.setattr(orchestrator, "process", lambda d, task: {task: "result"})
    monkeypatch.setattr(orchestrator, "summarize", lambda inp, ext, pr: "Human summary.")
    result = orchestrator.run("Simulate alloy.", tasks=["schema_validation"])
    assert "summary" in result
    assert "extraction" in result
    assert "processing" in result
    assert result["summary"] == "Human summary."
    assert result["extraction"] == extraction
    assert result["processing"] == {"schema_validation": {"schema_validation": "result"}}


def test_run_calls_extract_once_with_input(monkeypatch):
    from src.linear import orchestrator

    extraction = _minimal_extraction_dict()
    extract_calls = []

    def track_extract(t):
        extract_calls.append(t)
        return extraction

    monkeypatch.setattr(orchestrator, "extract", track_extract)
    monkeypatch.setattr(orchestrator, "process", lambda d, task: {})
    monkeypatch.setattr(orchestrator, "summarize", lambda i, e, p: "")
    orchestrator.run("My input text", tasks=["schema_validation"])
    assert extract_calls == ["My input text"]


def test_run_calls_process_per_task_with_extraction(monkeypatch):
    from src.linear import orchestrator

    extraction = _minimal_extraction_dict()
    process_calls = []

    def track_process(data, task):
        process_calls.append((data, task))
        return {"task": task}

    monkeypatch.setattr(orchestrator, "extract", lambda t: extraction)
    monkeypatch.setattr(orchestrator, "process", track_process)
    monkeypatch.setattr(orchestrator, "summarize", lambda i, e, p: "")
    orchestrator.run("x", tasks=["schema_validation", "constraint_verification"])
    assert len(process_calls) == 2
    assert process_calls[0] == (extraction, "schema_validation")
    assert process_calls[1] == (extraction, "constraint_verification")


def test_run_calls_summarize_with_input_extraction_and_processing(monkeypatch):
    from src.linear import orchestrator

    extraction = _minimal_extraction_dict()
    summarize_args = []

    def track_summarize(orig, ext, proc):
        summarize_args.append((orig, ext, proc))
        return "Summary"

    monkeypatch.setattr(orchestrator, "extract", lambda t: extraction)
    monkeypatch.setattr(orchestrator, "process", lambda d, task: {"key": task})
    monkeypatch.setattr(orchestrator, "summarize", track_summarize)
    orchestrator.run("Input", tasks=["normalization"])
    assert len(summarize_args) == 1
    assert summarize_args[0][0] == "Input"
    assert summarize_args[0][1] == extraction
    assert summarize_args[0][2] == {"normalization": {"key": "normalization"}}


def test_run_with_no_tasks_uses_all_tasks(monkeypatch):
    from src.linear import orchestrator
    from src.linear.processor import TASKS

    extraction = _minimal_extraction_dict()
    process_calls = []

    def track_process(data, task):
        process_calls.append(task)
        return {}

    monkeypatch.setattr(orchestrator, "extract", lambda t: extraction)
    monkeypatch.setattr(orchestrator, "process", track_process)
    monkeypatch.setattr(orchestrator, "summarize", lambda i, e, p: "")
    orchestrator.run("x", tasks=None)
    assert len(process_calls) == len(TASKS)
    for t in TASKS:
        assert t in process_calls
