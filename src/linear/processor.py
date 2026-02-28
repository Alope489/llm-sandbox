"""LLM-based symbolic reasoning on extracted material/simulation data. Uses wrapper.complete only."""
import json
import re

from src.wrapper import complete

TASK_SCHEMA_VALIDATION = "schema_validation"
TASK_CONSTRAINT_VERIFICATION = "constraint_verification"
TASK_FEATURE_EXTRACTION = "feature_extraction"
TASK_NORMALIZATION = "normalization"
TASK_RISK_RANKING = "risk_ranking"

TASKS = (
    TASK_SCHEMA_VALIDATION,
    TASK_CONSTRAINT_VERIFICATION,
    TASK_FEATURE_EXTRACTION,
    TASK_NORMALIZATION,
    TASK_RISK_RANKING,
)


def _parse_json_reply(text: str) -> dict:
    s = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    if m:
        s = m.group(1).strip()
    return json.loads(s)


def process(data: dict, task: str) -> dict:
    """Run one of schema_validation, constraint_verification, feature_extraction, normalization, risk_ranking on extraction-shaped data."""
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    if task == TASK_SCHEMA_VALIDATION:
        return _run_schema_validation(data)
    if task == TASK_CONSTRAINT_VERIFICATION:
        return _run_constraint_verification(data)
    if task == TASK_FEATURE_EXTRACTION:
        return _run_feature_extraction(data)
    if task == TASK_NORMALIZATION:
        return _run_normalization(data)
    return _run_risk_ranking(data)


def _run_schema_validation(data: dict) -> dict:
    reply = complete(
        [
            {
                "role": "system",
                "content": "You validate material/simulation extraction data. Check: composition percentages sum to ~100% (or note if missing); missing required fields; unit plausibility; contradictory fields (e.g. porosity 0% vs 'highly porous'). Reply with ONLY a JSON object: {\"valid\": boolean, \"issues\": [list of strings]}. No markdown, no explanation.",
            },
            {"role": "user", "content": json.dumps(data, indent=2)},
        ]
    )
    return _parse_json_reply(reply)


def _run_constraint_verification(data: dict) -> dict:
    reply = complete(
        [
            {
                "role": "system",
                "content": "You verify physics/constraint plausibility of material/simulation data. Consider: temperature vs melting point; realistic strain rate; model vs scale (e.g. DFT for macroscopic grain is inconsistent). Reply with ONLY a JSON object: {\"plausible\": boolean, \"warnings\": [list of strings]}. No markdown, no explanation.",
            },
            {"role": "user", "content": json.dumps(data, indent=2)},
        ]
    )
    return _parse_json_reply(reply)


def _run_feature_extraction(data: dict) -> dict:
    reply = complete(
        [
            {
                "role": "system",
                "content": "You classify the material/simulation from the extraction data. Infer: alloy_class (e.g. superalloy, composite, cathode); functional_category (e.g. structural, energy material); dominant_mechanism (e.g. dislocation, diffusion, phonon scattering); dimensionality (e.g. bulk, layered). Reply with ONLY a JSON object with keys: alloy_class, functional_category, dominant_mechanism, dimensionality (strings). No markdown, no explanation.",
            },
            {"role": "user", "content": json.dumps(data, indent=2)},
        ]
    )
    return _parse_json_reply(reply)


def _run_normalization(data: dict) -> dict:
    reply = complete(
        [
            {
                "role": "system",
                "content": "You normalize/reformat the extraction data: convert composition percentages to fractions (e.g. 60 -> 0.6); expand temperature_range_K {min, max, step} into an array of temperatures; keep units standardized. Return a single JSON object with the same top-level keys (material_system, processing_conditions, simulation_parameters, computed_properties, uncertainty_estimates) and normalized values. For composition use a list of {element, fraction}. For temperature range include a temperatures_K array. No markdown, no explanation.",
            },
            {"role": "user", "content": json.dumps(data, indent=2)},
        ]
    )
    return _parse_json_reply(reply)


def _run_risk_ranking(data: dict) -> dict:
    reply = complete(
        [
            {
                "role": "system",
                "content": "You rank by sensitivity/impact. From the extraction data: (1) Rank which computed properties are most sensitive to compositional variation (list property names from most to least sensitive). (2) Rank processing parameters by expected impact (list parameter names). Reply with ONLY a JSON object: {\"property_ranking\": [strings], \"processing_ranking\": [strings]}. No markdown, no explanation.",
            },
            {"role": "user", "content": json.dumps(data, indent=2)},
        ]
    )
    return _parse_json_reply(reply)
