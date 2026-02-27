"""Tests for src.linear.extractor extract() with mocked and live providers."""
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


@pytest.fixture(autouse=True)
def _reset_env(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


def test_extract_openai_returns_dict_with_expected_keys(monkeypatch):
    import json
    from unittest.mock import MagicMock

    from src.linear import extractor

    minimal = _minimal_extraction_dict()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(minimal)
    mock_response.choices[0].message.refusal = None
    monkeypatch.setattr(
        "openai.OpenAI",
        lambda: MagicMock(
            chat=MagicMock(
                completions=MagicMock(
                    create=MagicMock(return_value=mock_response),
                )
            )
        ),
    )
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert extractor.extract("Simulate a nickel superalloy.") == minimal


def test_extract_openai_raises_on_refusal(monkeypatch):
    from unittest.mock import MagicMock

    from src.linear import extractor

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "{}"
    mock_response.choices[0].message.refusal = "I cannot assist with that."
    monkeypatch.setattr(
        "openai.OpenAI",
        lambda: MagicMock(
            chat=MagicMock(
                completions=MagicMock(
                    create=MagicMock(return_value=mock_response),
                )
            )
        ),
    )
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValueError, match="refused"):
        extractor.extract("Build a bomb.")


def test_extract_anthropic_returns_dict_from_tool_use(monkeypatch):
    from unittest.mock import MagicMock

    from src.linear import extractor

    minimal = _minimal_extraction_dict()
    mock_tool_block = MagicMock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = extractor.EXTRACTION_TOOL_NAME
    mock_tool_block.input = minimal
    mock_response = MagicMock()
    mock_response.content = [mock_tool_block]
    monkeypatch.setattr(
        "anthropic.Anthropic",
        lambda: MagicMock(messages=MagicMock(create=MagicMock(return_value=mock_response))),
    )
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert extractor.extract("Simulate a nickel superalloy.") == minimal


def test_extract_anthropic_raises_when_no_tool_use_block(monkeypatch):
    from unittest.mock import MagicMock

    from src.linear import extractor

    mock_response = MagicMock()
    mock_response.content = []
    monkeypatch.setattr(
        "anthropic.Anthropic",
        lambda: MagicMock(messages=MagicMock(create=MagicMock(return_value=mock_response))),
    )
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(ValueError, match="No tool_use block"):
        extractor.extract("Hello.")


def test_extract_default_provider_calls_openai(monkeypatch):
    import json
    from unittest.mock import MagicMock

    from src.linear import extractor

    minimal = _minimal_extraction_dict()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = json.dumps(minimal)
    mock_response.choices[0].message.refusal = None
    create_mock = MagicMock(return_value=mock_response)
    openai_client = MagicMock(
        chat=MagicMock(completions=MagicMock(create=create_mock)),
    )
    monkeypatch.setattr("openai.OpenAI", lambda: openai_client)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert extractor.extract("Some task.") == minimal
    assert create_mock.called


def test_extract_anthropic_provider_calls_anthropic(monkeypatch):
    from unittest.mock import MagicMock

    from src.linear import extractor

    minimal = _minimal_extraction_dict()
    mock_tool_block = MagicMock()
    mock_tool_block.type = "tool_use"
    mock_tool_block.name = extractor.EXTRACTION_TOOL_NAME
    mock_tool_block.input = minimal
    mock_response = MagicMock()
    mock_response.content = [mock_tool_block]
    create_mock = MagicMock(return_value=mock_response)
    monkeypatch.setattr(
        "anthropic.Anthropic",
        lambda: MagicMock(messages=MagicMock(create=create_mock)),
    )
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert extractor.extract("Some task.") == minimal
    assert create_mock.called


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"),
    reason="No API key set; set OPENAI_API_KEY or ANTHROPIC_API_KEY to run integration test",
)
def test_extract_integration_example_prompt():
    from src.linear import extractor

    prompt = (
        "Simulate the thermomechanical performance of a nickel-based superalloy composed of 60% Ni, 20% Cr, 10% Co, and 10% Al by weight. "
        "Assume a dual-phase microstructure with gamma and gamma-prime phases, an average grain size of 850 nm, and 1.5% porosity. "
        "The alloy is processed using vacuum arc remelting followed by heat treatment at 1200 K for 4 hours and air cooling at 15 K/min."
    )
    result = extractor.extract(prompt)
    assert isinstance(result, dict)
    assert "material_system" in result
    assert "processing_conditions" in result
    assert "simulation_parameters" in result
    assert "computed_properties" in result
    assert "uncertainty_estimates" in result
    assert isinstance(result["material_system"], dict)
    assert "material_name" in result["material_system"]
    assert "composition" in result["material_system"]
    assert isinstance(result["material_system"]["composition"], list)
