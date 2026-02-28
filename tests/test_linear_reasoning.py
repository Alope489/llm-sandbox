"""Tests for src.linear.reasoning summarize() with mocked wrapper."""
import json
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
load_dotenv()


def _minimal_extraction_dict():
    return {
        "material_system": {"material_name": "Ni superalloy", "composition": [], "phase_type": None, "microstructure": {"grain_size_nm": None, "porosity_percent": None, "crystal_structure": None}},
        "processing_conditions": {"synthesis_method": None, "heat_treatment": {"temperature_K": None, "duration_hours": None, "cooling_rate_K_per_min": None}, "pressure_GPa": None},
        "simulation_parameters": {"temperature_range_K": {"min": None, "max": None, "step": None}, "strain_rate_s_inverse": None, "boundary_conditions": None, "model_type": None},
        "computed_properties": {"thermal_conductivity_W_per_mK": None, "yield_strength_MPa": None, "youngs_modulus_GPa": None, "poissons_ratio": None, "thermal_expansion_coefficient_per_K": None, "specific_heat_J_per_kgK": None, "electrical_conductivity_S_per_m": None, "density_kg_per_m3": None},
        "uncertainty_estimates": {"property_uncertainty_percent": None, "model_confidence_level": None},
    }


def test_summarize_returns_string(monkeypatch):
    from src.linear import reasoning

    monkeypatch.setattr(reasoning, "complete", lambda msgs: "Summary: extraction and two tasks run.")
    result = reasoning.summarize(
        "Simulate Ni alloy.",
        _minimal_extraction_dict(),
        {"schema_validation": {"valid": True, "issues": []}, "constraint_verification": {"plausible": True, "warnings": []}},
    )
    assert isinstance(result, str)
    assert "Summary" in result


def test_summarize_passes_input_extraction_and_processing_to_complete(monkeypatch):
    from src.linear import reasoning

    seen = []

    def capture(msgs):
        seen.append(msgs)
        return "Done."

    monkeypatch.setattr(reasoning, "complete", capture)
    extraction = _minimal_extraction_dict()
    processing_results = {"schema_validation": {"valid": False, "issues": ["Missing composition"]}}
    reasoning.summarize("Original task text", extraction, processing_results)
    assert len(seen) == 1
    user_content = next((m["content"] for m in seen[0] if m.get("role") == "user"), "")
    payload = json.loads(user_content)
    assert payload["original_input"] == "Original task text"
    assert payload["extraction"] == extraction
    assert payload["processing_results"] == processing_results


def test_summarize_system_prompt_describes_linear_structure(monkeypatch):
    from src.linear import reasoning

    seen = []

    def capture(msgs):
        seen.append(msgs)
        return "Done."

    monkeypatch.setattr(reasoning, "complete", capture)
    reasoning.summarize("x", _minimal_extraction_dict(), {})
    system_content = next((m["content"] for m in seen[0] if m.get("role") == "system"), "")
    assert "extractor" in system_content.lower()
    assert "processor" in system_content.lower()
    assert "schema_validation" in system_content or "extraction" in system_content
