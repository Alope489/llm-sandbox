"""Sandbox processor component (independent, OpenAI-only).

This is a rebuild of `src/linear/processor.py` for the LLM sandbox. It runs a
single OpenAI call per invocation and returns an Output dict including:
  - result: parsed JSON dict produced by the model
  - output_prompt: the model's raw output text
  - elapsed_ms: wall-clock time spent waiting for the LLM call
  - input_tokens: provider-reported input token count
  - output_tokens: provider-reported output token count

It also prints a one-line summary to the terminal and appends details to the
central `test-results.md` file.
"""

import json
import os
import re
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_RESULTS_PATH = "test-results.md"
_DEFAULT_REASONING_INPUT_PATH = os.path.join("sandbox", "prompts", "reasoning_input.txt")

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


def _parse_json_reply(text: str) -> dict[str, Any]:
    s = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s)
    return json.loads(m.group(1).strip() if m else s)


def process(data: dict[str, Any], task: str, *, emit_artifacts: bool = True) -> dict[str, Any]:
    """Run a processor task on an extraction-shaped dict."""
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")
    output = _process_openai(data, task)
    if os.environ.get("SANDBOX_PRINT", "1") == "1":
        print(f"[sandbox.processor:{task}] elapsed_ms={output['elapsed_ms']} input_tokens={output['input_tokens']} output_tokens={output['output_tokens']}")
    if emit_artifacts:
        _append_test_results(component=f"processor.{task}", input_prompt=output["input_prompt"], output=output)
        _append_reasoning_input(output)
    return output


def _append_reasoning_input(output: dict[str, Any]) -> None:
    path = os.environ.get("SANDBOX_REASONING_INPUT_PATH", _DEFAULT_REASONING_INPUT_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "producer_file": "processor.py",
                    "task": output.get("task"),
                    "input_prompt": output.get("input_prompt"),
                    "result": output.get("result"),
                    "output_prompt": output.get("output_prompt"),
                    "elapsed_ms": output.get("elapsed_ms"),
                    "input_tokens": output.get("input_tokens"),
                    "output_tokens": output.get("output_tokens"),
                },
                ensure_ascii=False,
            )
        )
        f.write("\n---\n")


def _process_openai(data: dict[str, Any], task: str) -> dict[str, Any]:
    from openai import OpenAI

    system_prompt = {
        TASK_SCHEMA_VALIDATION: (
            "You validate material/simulation extraction data. Check: composition percentages sum to ~100% (or note if missing); "
            "missing required fields; unit plausibility; contradictory fields (e.g. porosity 0% vs 'highly porous'). "
            'Reply with ONLY a JSON object: {"valid": boolean, "issues": [list of strings]}. No markdown, no explanation.'
        ),
        TASK_CONSTRAINT_VERIFICATION: (
            "You verify physics/constraint plausibility of material/simulation data. Consider: temperature vs melting point; realistic strain rate; "
            'model vs scale (e.g. DFT for macroscopic grain is inconsistent). Reply with ONLY a JSON object: {"plausible": boolean, "warnings": [list of strings]}. '
            "No markdown, no explanation."
        ),
        TASK_FEATURE_EXTRACTION: (
            "You classify the material/simulation from the extraction data. Infer: alloy_class (e.g. superalloy, composite, cathode); functional_category (e.g. structural, energy material); "
            'dominant_mechanism (e.g. dislocation, diffusion, phonon scattering); dimensionality (e.g. bulk, layered). Reply with ONLY a JSON object with keys: alloy_class, functional_category, '
            "dominant_mechanism, dimensionality (strings). No markdown, no explanation."
        ),
        TASK_NORMALIZATION: (
            "You normalize/reformat the extraction data: convert composition percentages to fractions (e.g. 60 -> 0.6); expand temperature_range_K {min, max, step} into an array of temperatures; "
            "keep units standardized. Return a single JSON object with the same top-level keys (material_system, processing_conditions, simulation_parameters, computed_properties, uncertainty_estimates) "
            "and normalized values. For composition use a list of {element, fraction}. For temperature range include a temperatures_K array. No markdown, no explanation."
        ),
        TASK_RISK_RANKING: (
            "You rank by sensitivity/impact. From the extraction data: (1) Rank which computed properties are most sensitive to compositional variation (list property names from most to least sensitive). "
            '(2) Rank processing parameters by expected impact (list parameter names). Reply with ONLY a JSON object: {"property_ranking": [strings], "processing_ranking": [strings]}. No markdown, no explanation.'
        ),
    }[task]

    input_prompt = json.dumps(data, indent=2, ensure_ascii=False)
    t0 = time.perf_counter()
    raw = OpenAI().with_raw_response.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": input_prompt}],
        response_format={"type": "json_object"},
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    parsed = raw.parse()
    msg = parsed.choices[0].message
    if getattr(msg, "refusal", None):
        raise ValueError(f"Model refused: {msg.refusal}")
    usage = parsed.usage
    output_prompt = msg.content
    return {
        "task": task,
        "input_prompt": input_prompt,
        "result": _parse_json_reply(output_prompt),
        "output_prompt": output_prompt,
        "elapsed_ms": round(elapsed_ms, 3),
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
    }

