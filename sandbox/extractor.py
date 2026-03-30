"""Sandbox extractor component (independent, OpenAI-only).

This module is intentionally self-contained so it can be tested independently
inside `sandbox/` without importing code from `src/`.

It performs one OpenAI LLM call and returns an Output dict that
includes:
  - extraction: parsed extraction dict (schema-constrained)
  - output_prompt: the model's raw output text (JSON string)
  - elapsed_ms: wall-clock time spent waiting for the LLM call
  - input_tokens: provider-reported input token count
  - output_tokens: provider-reported output token count
"""

import json
import os
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_RESULTS_PATH = "test-results.md"
_DEFAULT_PROCESSOR_PROMPTS_PATH = os.path.join("sandbox", "prompts", "processor.txt")
_DEFAULT_REASONING_INPUT_PATH = os.path.join("sandbox", "prompts", "reasoning_input.txt")

EXTRACTION_SCHEMA: dict[str, Any] = {
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

_SYSTEM_PROMPT = (
    "Extract structured data from the task description that follows. "
    "Return ONLY valid JSON. Do not include explanations. Do not summarize. "
    "Do not restate the task. Do not include markdown. "
    "If a value is missing and cannot be reasonably inferred, return null."
)

def _append_test_results(*, component: str, input_prompt: str, output: dict[str, Any]) -> None:
    path = os.environ.get("SANDBOX_RESULTS_PATH", _DEFAULT_RESULTS_PATH)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    block = (
        f"\n\n## {component} ({stamp})\n\n"
        f"### Input\n\n```\n{input_prompt}\n```\n\n"
        f"### Output\n\n"
        f"- elapsed_ms: {output.get('elapsed_ms')}\n"
        f"- input_tokens: {output.get('input_tokens')}\n"
        f"- output_tokens: {output.get('output_tokens')}\n\n"
        f"```json\n{json.dumps(output, ensure_ascii=False, indent=2)}\n```\n"
    )
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("# LLM sandbox test results\n")
    with open(path, "a", encoding="utf-8") as f:
        f.write(block)


def _append_processor_prompt(*, extraction: dict[str, Any]) -> None:
    path = os.environ.get("SANDBOX_PROCESSOR_PROMPTS_PATH", _DEFAULT_PROCESSOR_PROMPTS_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"producer_file": "extractor.py", "extraction": extraction}, ensure_ascii=False))
        f.write("\n---\n")


def _append_reasoning_input(*, original_input: str, extraction: dict[str, Any]) -> None:
    path = os.environ.get("SANDBOX_REASONING_INPUT_PATH", _DEFAULT_REASONING_INPUT_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "producer_file": "extractor.py",
                    "original_input": original_input,
                    "extraction": extraction,
                },
                ensure_ascii=False,
            )
        )
        f.write("\n---\n")


def extract(text: str, *, emit_artifacts: bool = True) -> dict[str, Any]:
    """Run the sandbox extractor.

    Environment:
        - OPENAI_MODEL: default "gpt-4o-mini"

    Returns:
        Output dict with keys: extraction, output_prompt, elapsed_ms, input_tokens, output_tokens.
    """
    output = _extract_openai(text)
    if os.environ.get("SANDBOX_PRINT", "1") == "1":
        print(f"[sandbox.extractor] elapsed_ms={output['elapsed_ms']} input_tokens={output['input_tokens']} output_tokens={output['output_tokens']}")
    if emit_artifacts:
        _append_test_results(component="extractor", input_prompt=text, output=output)
        _append_processor_prompt(extraction=output["extraction"])
        _append_reasoning_input(original_input=text, extraction=output["extraction"])
    return output


def _extract_openai(text: str) -> dict[str, Any]:
    from openai import OpenAI

    t0 = time.perf_counter()
    raw = OpenAI().with_raw_response.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": text}],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "material_simulation_extraction", "strict": True, "schema": EXTRACTION_SCHEMA},
        },
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    parsed = raw.parse()
    msg = parsed.choices[0].message
    if getattr(msg, "refusal", None):
        raise ValueError(f"Model refused: {msg.refusal}")
    output_prompt = msg.content
    usage = parsed.usage
    return {
        "extraction": json.loads(output_prompt),
        "output_prompt": output_prompt,
        "elapsed_ms": round(elapsed_ms, 3),
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
    }

