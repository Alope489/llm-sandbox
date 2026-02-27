"""Extract structured material/simulation data from task descriptions via LLM (OpenAI or Anthropic)."""
import json
import os

from dotenv import load_dotenv

load_dotenv()

EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "material_system": {
            "type": "object",
            "properties": {
                "material_name": {"type": ["string", "null"]},
                "composition": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "element": {"type": "string"},
                            "percentage": {"type": "number"},
                        },
                        "required": ["element", "percentage"],
                        "additionalProperties": False,
                    },
                },
                "phase_type": {"type": ["string", "null"]},
                "microstructure": {
                    "type": "object",
                    "properties": {
                        "grain_size_nm": {"type": ["number", "null"]},
                        "porosity_percent": {"type": ["number", "null"]},
                        "crystal_structure": {"type": ["string", "null"]},
                    },
                    "required": ["grain_size_nm", "porosity_percent", "crystal_structure"],
                    "additionalProperties": False,
                },
            },
            "required": ["material_name", "composition", "phase_type", "microstructure"],
            "additionalProperties": False,
        },
        "processing_conditions": {
            "type": "object",
            "properties": {
                "synthesis_method": {"type": ["string", "null"]},
                "heat_treatment": {
                    "type": "object",
                    "properties": {
                        "temperature_K": {"type": ["number", "null"]},
                        "duration_hours": {"type": ["number", "null"]},
                        "cooling_rate_K_per_min": {"type": ["number", "null"]},
                    },
                    "required": ["temperature_K", "duration_hours", "cooling_rate_K_per_min"],
                    "additionalProperties": False,
                },
                "pressure_GPa": {"type": ["number", "null"]},
            },
            "required": ["synthesis_method", "heat_treatment", "pressure_GPa"],
            "additionalProperties": False,
        },
        "simulation_parameters": {
            "type": "object",
            "properties": {
                "temperature_range_K": {
                    "type": "object",
                    "properties": {
                        "min": {"type": ["number", "null"]},
                        "max": {"type": ["number", "null"]},
                        "step": {"type": ["number", "null"]},
                    },
                    "required": ["min", "max", "step"],
                    "additionalProperties": False,
                },
                "strain_rate_s_inverse": {"type": ["number", "null"]},
                "boundary_conditions": {"type": ["string", "null"]},
                "model_type": {"type": ["string", "null"]},
            },
            "required": ["temperature_range_K", "strain_rate_s_inverse", "boundary_conditions", "model_type"],
            "additionalProperties": False,
        },
        "computed_properties": {
            "type": "object",
            "properties": {
                "thermal_conductivity_W_per_mK": {"type": ["number", "null"]},
                "yield_strength_MPa": {"type": ["number", "null"]},
                "youngs_modulus_GPa": {"type": ["number", "null"]},
                "poissons_ratio": {"type": ["number", "null"]},
                "thermal_expansion_coefficient_per_K": {"type": ["number", "null"]},
                "specific_heat_J_per_kgK": {"type": ["number", "null"]},
                "electrical_conductivity_S_per_m": {"type": ["number", "null"]},
                "density_kg_per_m3": {"type": ["number", "null"]},
            },
            "required": [
                "thermal_conductivity_W_per_mK",
                "yield_strength_MPa",
                "youngs_modulus_GPa",
                "poissons_ratio",
                "thermal_expansion_coefficient_per_K",
                "specific_heat_J_per_kgK",
                "electrical_conductivity_S_per_m",
                "density_kg_per_m3",
            ],
            "additionalProperties": False,
        },
        "uncertainty_estimates": {
            "type": "object",
            "properties": {
                "property_uncertainty_percent": {"type": ["number", "null"]},
                "model_confidence_level": {"type": ["number", "null"]},
            },
            "required": ["property_uncertainty_percent", "model_confidence_level"],
            "additionalProperties": False,
        },
    },
    "required": [
        "material_system",
        "processing_conditions",
        "simulation_parameters",
        "computed_properties",
        "uncertainty_estimates",
    ],
    "additionalProperties": False,
}

EXTRACTION_TOOL_NAME = "record_extraction"


def extract(text: str) -> dict:
    """Extract structured material/simulation data from task description. Uses LLM_PROVIDER (openai or anthropic)."""
    if os.environ.get("LLM_PROVIDER", "openai") == "anthropic":
        return _extract_anthropic(text)
    return _extract_openai(text)


def _extract_openai(text: str) -> dict:
    from openai import OpenAI

    response = OpenAI().chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {
                "role": "system",
                "content": "Extract structured data from the task description that follows. Return ONLY valid JSON. Do not include explanations. Do not summarize. Do not restate the task. Do not include markdown. If a value is missing and cannot be reasonably inferred, return null.",
            },
            {"role": "user", "content": text},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "material_simulation_extraction",
                "strict": True,
                "schema": EXTRACTION_SCHEMA,
            },
        },
    )
    msg = response.choices[0].message
    if getattr(msg, "refusal", None):
        raise ValueError(f"Model refused: {msg.refusal}")
    return json.loads(msg.content)


def _extract_anthropic(text: str) -> dict:
    from anthropic import Anthropic

    response = Anthropic().messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
        max_tokens=int(os.environ.get("MAX_TOKENS", "1024")),
        tools=[
            {
                "name": EXTRACTION_TOOL_NAME,
                "description": "Record the extracted material and simulation parameters from the task description.",
                "input_schema": EXTRACTION_SCHEMA,
                "strict": True,
            }
        ],
        tool_choice={"type": "tool", "name": EXTRACTION_TOOL_NAME},
        messages=[
            {
                "role": "user",
                "content": "Extract structured data from the task description that follows. Return ONLY valid JSON. Do not include explanations. Do not summarize. Do not restate the task. Do not include markdown. If a value is missing and cannot be reasonably inferred, return null.\n\n"
                + text,
            }
        ],
    )
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == EXTRACTION_TOOL_NAME:
            return block.input
    raise ValueError("No tool_use block found in Anthropic response")
