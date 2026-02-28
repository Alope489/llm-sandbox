"""Tests for src.linear.processor process() with mocked wrapper and integration tests per task."""
import json
import os
import sys

import pytest
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
load_dotenv()


def _minimal_extraction_dict():
    return {
        "material_system": {
            "material_name": "Ni-based superalloy",
            "composition": [{"element": "Ni", "percentage": 60}, {"element": "Cr", "percentage": 20}],
            "phase_type": "dual-phase",
            "microstructure": {"grain_size_nm": 850, "porosity_percent": 1.5, "crystal_structure": "gamma/gamma-prime"},
        },
        "processing_conditions": {
            "synthesis_method": "vacuum arc remelting",
            "heat_treatment": {"temperature_K": 1200, "duration_hours": 4, "cooling_rate_K_per_min": 15},
            "pressure_GPa": None,
        },
        "simulation_parameters": {
            "temperature_range_K": {"min": 500, "max": 1300, "step": 200},
            "strain_rate_s_inverse": 1e-3,
            "boundary_conditions": "periodic",
            "model_type": "crystal plasticity",
        },
        "computed_properties": {
            "thermal_conductivity_W_per_mK": None,
            "yield_strength_MPa": None,
            "youngs_modulus_GPa": None,
            "poissons_ratio": None,
            "thermal_expansion_coefficient_per_K": None,
            "specific_heat_J_per_kgK": None,
            "electrical_conductivity_S_per_m": None,
            "density_kg_per_m3": None,
        },
        "uncertainty_estimates": {"property_uncertainty_percent": 5, "model_confidence_level": None},
    }


def test_process_unknown_task_raises():
    from src.linear import processor

    with pytest.raises(ValueError, match="Unknown task"):
        processor.process(_minimal_extraction_dict(), "unknown_task")


def test_process_schema_validation_returns_valid_and_issues(monkeypatch):
    from src.linear import processor

    monkeypatch.setattr(
        processor,
        "complete",
        lambda msgs: json.dumps({"valid": False, "issues": ["Composition sums to 80%, not 100%"]}),
    )
    result = processor.process(_minimal_extraction_dict(), processor.TASK_SCHEMA_VALIDATION)
    assert "valid" in result
    assert "issues" in result
    assert result["valid"] is False
    assert isinstance(result["issues"], list)
    assert "80%" in result["issues"][0]


def test_process_schema_validation_passes_data_in_messages(monkeypatch):
    from src.linear import processor

    seen = []

    def capture(m):
        seen.append(m)
        return json.dumps({"valid": True, "issues": []})

    monkeypatch.setattr(processor, "complete", capture)
    processor.process(_minimal_extraction_dict(), processor.TASK_SCHEMA_VALIDATION)
    assert len(seen) == 1
    user_content = next((m["content"] for m in seen[0] if m.get("role") == "user"), "")
    assert "Ni-based superalloy" in user_content
    assert "60" in user_content


def test_process_constraint_verification_returns_plausible_and_warnings(monkeypatch):
    from src.linear import processor

    monkeypatch.setattr(
        processor,
        "complete",
        lambda msgs: json.dumps({"plausible": True, "warnings": []}),
    )
    result = processor.process(_minimal_extraction_dict(), processor.TASK_CONSTRAINT_VERIFICATION)
    assert "plausible" in result
    assert "warnings" in result
    assert result["plausible"] is True
    assert isinstance(result["warnings"], list)


def test_process_constraint_verification_passes_data_in_messages(monkeypatch):
    from src.linear import processor

    seen = []

    def capture(m):
        seen.append(m)
        return json.dumps({"plausible": True, "warnings": []})

    monkeypatch.setattr(processor, "complete", capture)
    processor.process(_minimal_extraction_dict(), processor.TASK_CONSTRAINT_VERIFICATION)
    assert len(seen) == 1
    user_content = next((m["content"] for m in seen[0] if m.get("role") == "user"), "")
    assert "1200" in user_content
    assert "crystal plasticity" in user_content


def test_process_feature_extraction_returns_expected_keys(monkeypatch):
    from src.linear import processor

    monkeypatch.setattr(
        processor,
        "complete",
        lambda msgs: json.dumps({
            "alloy_class": "superalloy",
            "functional_category": "structural",
            "dominant_mechanism": "dislocation",
            "dimensionality": "bulk",
        }),
    )
    result = processor.process(_minimal_extraction_dict(), processor.TASK_FEATURE_EXTRACTION)
    assert result["alloy_class"] == "superalloy"
    assert result["functional_category"] == "structural"
    assert result["dominant_mechanism"] == "dislocation"
    assert result["dimensionality"] == "bulk"


def test_process_feature_extraction_passes_data_in_messages(monkeypatch):
    from src.linear import processor

    seen = []

    def capture(m):
        seen.append(m)
        return json.dumps({
            "alloy_class": "superalloy",
            "functional_category": "structural",
            "dominant_mechanism": "dislocation",
            "dimensionality": "bulk",
        })

    monkeypatch.setattr(processor, "complete", capture)
    processor.process(_minimal_extraction_dict(), processor.TASK_FEATURE_EXTRACTION)
    assert len(seen) == 1
    user_content = next((m["content"] for m in seen[0] if m.get("role") == "user"), "")
    assert "material_system" in user_content


def test_process_normalization_returns_dict_with_top_level_keys(monkeypatch):
    from src.linear import processor

    monkeypatch.setattr(
        processor,
        "complete",
        lambda msgs: json.dumps({
            "material_system": {"material_name": "Ni-based superalloy", "composition": [{"element": "Ni", "fraction": 0.6}, {"element": "Cr", "fraction": 0.2}]},
            "processing_conditions": {},
            "simulation_parameters": {"temperatures_K": [500, 700, 900, 1100, 1300]},
            "computed_properties": {},
            "uncertainty_estimates": {},
        }),
    )
    result = processor.process(_minimal_extraction_dict(), processor.TASK_NORMALIZATION)
    assert "material_system" in result
    assert "processing_conditions" in result
    assert "simulation_parameters" in result
    assert "computed_properties" in result
    assert "uncertainty_estimates" in result
    assert result["material_system"]["composition"][0]["fraction"] == 0.6


def test_process_normalization_passes_data_in_messages(monkeypatch):
    from src.linear import processor

    seen = []

    def capture(m):
        seen.append(m)
        return json.dumps({
            "material_system": _minimal_extraction_dict()["material_system"],
            "processing_conditions": _minimal_extraction_dict()["processing_conditions"],
            "simulation_parameters": _minimal_extraction_dict()["simulation_parameters"],
            "computed_properties": _minimal_extraction_dict()["computed_properties"],
            "uncertainty_estimates": _minimal_extraction_dict()["uncertainty_estimates"],
        })

    monkeypatch.setattr(processor, "complete", capture)
    processor.process(_minimal_extraction_dict(), processor.TASK_NORMALIZATION)
    assert len(seen) == 1
    user_content = next((m["content"] for m in seen[0] if m.get("role") == "user"), "")
    assert "percentage" in user_content


def test_process_risk_ranking_returns_rankings(monkeypatch):
    from src.linear import processor

    monkeypatch.setattr(
        processor,
        "complete",
        lambda msgs: json.dumps({
            "property_ranking": ["yield_strength_MPa", "thermal_conductivity_W_per_mK"],
            "processing_ranking": ["heat_treatment", "synthesis_method"],
        }),
    )
    result = processor.process(_minimal_extraction_dict(), processor.TASK_RISK_RANKING)
    assert "property_ranking" in result
    assert "processing_ranking" in result
    assert isinstance(result["property_ranking"], list)
    assert isinstance(result["processing_ranking"], list)


def test_process_risk_ranking_passes_data_in_messages(monkeypatch):
    from src.linear import processor

    seen = []

    def capture(m):
        seen.append(m)
        return json.dumps({"property_ranking": [], "processing_ranking": []})

    monkeypatch.setattr(processor, "complete", capture)
    processor.process(_minimal_extraction_dict(), processor.TASK_RISK_RANKING)
    assert len(seen) == 1
    user_content = next((m["content"] for m in seen[0] if m.get("role") == "user"), "")
    assert "uncertainty" in user_content or "property_uncertainty" in user_content or "5" in user_content


def test_process_parses_json_inside_markdown_fence(monkeypatch):
    from src.linear import processor

    monkeypatch.setattr(
        processor,
        "complete",
        lambda msgs: "```json\n{\"valid\": true, \"issues\": []}\n```",
    )
    result = processor.process(_minimal_extraction_dict(), processor.TASK_SCHEMA_VALIDATION)
    assert result["valid"] is True
    assert result["issues"] == []


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"),
    reason="No API key set; set OPENAI_API_KEY or ANTHROPIC_API_KEY to run integration test",
)
def test_process_integration_schema_validation():
    from src.linear import processor

    result = processor.process(_minimal_extraction_dict(), processor.TASK_SCHEMA_VALIDATION)
    assert isinstance(result, dict)
    assert "valid" in result
    assert "issues" in result
    assert isinstance(result["issues"], list)


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"),
    reason="No API key set; set OPENAI_API_KEY or ANTHROPIC_API_KEY to run integration test",
)
def test_process_integration_constraint_verification():
    from src.linear import processor

    result = processor.process(_minimal_extraction_dict(), processor.TASK_CONSTRAINT_VERIFICATION)
    assert isinstance(result, dict)
    assert "plausible" in result
    assert "warnings" in result
    assert isinstance(result["warnings"], list)


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"),
    reason="No API key set; set OPENAI_API_KEY or ANTHROPIC_API_KEY to run integration test",
)
def test_process_integration_feature_extraction():
    from src.linear import processor

    result = processor.process(_minimal_extraction_dict(), processor.TASK_FEATURE_EXTRACTION)
    assert isinstance(result, dict)
    assert "alloy_class" in result
    assert "functional_category" in result
    assert "dominant_mechanism" in result
    assert "dimensionality" in result


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"),
    reason="No API key set; set OPENAI_API_KEY or ANTHROPIC_API_KEY to run integration test",
)
def test_process_integration_normalization():
    from src.linear import processor

    result = processor.process(_minimal_extraction_dict(), processor.TASK_NORMALIZATION)
    assert isinstance(result, dict)
    assert "material_system" in result
    assert "processing_conditions" in result
    assert "simulation_parameters" in result
    assert "computed_properties" in result
    assert "uncertainty_estimates" in result


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"),
    reason="No API key set; set OPENAI_API_KEY or ANTHROPIC_API_KEY to run integration test",
)
def test_process_integration_risk_ranking():
    from src.linear import processor

    result = processor.process(_minimal_extraction_dict(), processor.TASK_RISK_RANKING)
    assert isinstance(result, dict)
    assert "property_ranking" in result
    assert "processing_ranking" in result
    assert isinstance(result["property_ranking"], list)
    assert isinstance(result["processing_ranking"], list)
