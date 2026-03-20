"""Extract structured material/simulation data from task descriptions via LLM (OpenAI or Anthropic).

Telemetry instrumentation:
    ``extract``, ``_extract_openai``, and ``_extract_anthropic`` all accept an
    optional ``ctx`` parameter (``CallContext``).  When provided, a
    ``llm_call`` record is appended to ``ctx.records`` with timing and token
    counts.  The caller should supply a ``dataclasses.replace()`` snapshot with
    ``agent`` and ``span`` already set.

Dependencies:
    json, os, dataclasses, time, datetime, dotenv, openai, anthropic,
    src.llm_pipeline_telemetry.

Pillar compliance:
    - Pillar 1: Functional parity with pre-telemetry code.
    - Pillar 4: No hardcoding; model from env vars.
    - Pillar 7: try/except with error-path telemetry records.
"""
import dataclasses
import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

from src.llm_pipeline_telemetry import (
    CallContext,
    get_anthropic_client,
    get_openai_client,
    log_llm_call,
)

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


def extract(text: str, ctx: Optional[CallContext] = None) -> dict:
    """Extract structured material/simulation data from task description. Uses LLM_PROVIDER (openai or anthropic).

    Args:
        text: Raw task description string.
        ctx: Optional ``CallContext`` snapshot.  When provided, a ``llm_call``
            record is emitted.  The caller should have already stamped
            ``agent="extractor"`` and ``span="extract"`` via
            ``dataclasses.replace()``.

    Returns:
        A dict matching the extraction schema.

    Raises:
        ValueError: If the model refuses the request.

    Complexity:
        O(1) — single API call.
    """
    if os.environ.get("LLM_PROVIDER", "openai") == "anthropic":
        return _extract_anthropic(text, ctx=ctx)
    return _extract_openai(text, ctx=ctx)


def _extract_openai(text: str, ctx: Optional[CallContext] = None) -> dict:
    """Call OpenAI with structured JSON output schema and return the parsed dict.

    Args:
        text: Raw task description.
        ctx: Optional ``CallContext`` snapshot.

    Returns:
        Parsed extraction dict.

    Raises:
        ValueError: If the model issues a refusal.
        openai.APIError: On API failure; error record emitted when ctx set.

    Postconditions:
        - If ctx is not None, exactly one ``llm_call`` record is appended.

    Complexity:
        O(1).
    """
    client = get_openai_client()
    call_ctx = dataclasses.replace(ctx, agent="extractor", span="extract") if ctx is not None else None
    call_start_ts = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    try:
        raw = client.with_raw_response.chat.completions.create(
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
        client_elapsed_ms = (time.perf_counter() - t0) * 1000
        call_end_ts = datetime.now(timezone.utc)
        if call_ctx is not None:
            usage = raw.parse().usage
            raw_ms = raw.headers.get("openai-processing-ms")
            server_ms = int(raw_ms) if raw_ms and int(raw_ms) > 0 else None
            log_llm_call(
                call_ctx,
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                provider_server_latency_ms=server_ms,
                client_elapsed_ms=client_elapsed_ms,
                call_start_ts=call_start_ts.isoformat(),
                call_end_ts=call_end_ts.isoformat(),
                status="ok",
            )
        msg = raw.parse().choices[0].message
        if getattr(msg, "refusal", None):
            raise ValueError(f"Model refused: {msg.refusal}")
        return json.loads(msg.content)
    except Exception:
        client_elapsed_ms = (time.perf_counter() - t0) * 1000
        call_end_ts = datetime.now(timezone.utc)
        if call_ctx is not None:
            log_llm_call(
                call_ctx,
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                input_tokens=0,
                output_tokens=0,
                provider_server_latency_ms=None,
                client_elapsed_ms=client_elapsed_ms,
                call_start_ts=call_start_ts.isoformat(),
                call_end_ts=call_end_ts.isoformat(),
                status="error",
            )
        raise


def _extract_anthropic(text: str, ctx: Optional[CallContext] = None) -> dict:
    """Call Anthropic with a forced tool_use response and return the parsed dict.

    Args:
        text: Raw task description.
        ctx: Optional ``CallContext`` snapshot.

    Returns:
        Parsed extraction dict.

    Raises:
        ValueError: If no ``tool_use`` block is found in the response.
        anthropic.APIError: On API failure; error record emitted when ctx set.

    Postconditions:
        - If ctx is not None, exactly one ``llm_call`` record is appended.
        - ``provider_server_latency_ms`` is always ``None`` (Anthropic).

    Complexity:
        O(1).
    """
    client = get_anthropic_client()
    call_ctx = dataclasses.replace(ctx, agent="extractor", span="extract") if ctx is not None else None
    call_start_ts = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    try:
        raw = client.with_raw_response.messages.create(
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
        client_elapsed_ms = (time.perf_counter() - t0) * 1000
        call_end_ts = datetime.now(timezone.utc)
        if call_ctx is not None:
            usage = raw.parse().usage
            log_llm_call(
                call_ctx,
                model=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                provider_server_latency_ms=None,
                client_elapsed_ms=client_elapsed_ms,
                call_start_ts=call_start_ts.isoformat(),
                call_end_ts=call_end_ts.isoformat(),
                status="ok",
            )
        for block in raw.parse().content:
            if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == EXTRACTION_TOOL_NAME:
                return block.input
        raise ValueError("No tool_use block found in Anthropic response")
    except Exception:
        client_elapsed_ms = (time.perf_counter() - t0) * 1000
        call_end_ts = datetime.now(timezone.utc)
        if call_ctx is not None:
            log_llm_call(
                call_ctx,
                model=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
                input_tokens=0,
                output_tokens=0,
                provider_server_latency_ms=None,
                client_elapsed_ms=client_elapsed_ms,
                call_start_ts=call_start_ts.isoformat(),
                call_end_ts=call_end_ts.isoformat(),
                status="error",
            )
        raise
