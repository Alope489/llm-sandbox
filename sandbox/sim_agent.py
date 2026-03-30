"""Sandbox simulation agent (OpenAI-only, single-file regular + real modes)."""

import json
import math
import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_DEFAULT_RESULTS_PATH = "test-results.md"
_DEFAULT_REASONING_INPUT_PATH = os.path.join("sandbox", "prompts", "reasoning_input.txt")
_DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
_DEFAULT_PREDEFINED_SIM_CALLS = (("Al", "3"), ("Cu", "3"), ("Ni", "4"), ("Fe", "4"), ("W", "3"), ("Mo", "5"))


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
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("# LLM sandbox test results\n")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(block)


def _append_reasoning_input(payload: dict[str, Any]) -> None:
    path = os.environ.get("SANDBOX_REASONING_INPUT_PATH", _DEFAULT_REASONING_INPUT_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n---\n")


def _parse_number(text: str) -> float | None:
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text.strip() if isinstance(text, str) else "")
    if not match:
        return None
    try:
        return float(match.group())
    except ValueError:
        return None


def _run_material_simulation(cooling_rate_K_per_min: float, duration_hours: float = 4.0, num_steps: int = 20) -> dict[str, Any]:
    grain_size_nm = 850.0
    porosity_percent = 1.5
    for step in range(num_steps):
        grain_size_nm = max(50.0, min(850.0, 850.0 / (1.0 + 0.03 * (step + 1) * math.log1p(min(2.0, max(0.2, cooling_rate_K_per_min / 15.0))))))
        porosity_percent = max(
            0.0,
            min(10.0, porosity_percent + (0.08 * math.log1p(max(0.0, cooling_rate_K_per_min - 10.0)) + 0.02 * abs(duration_hours - 4.0)) / num_steps),
        )
        if porosity_percent > 5.0:
            break
    return {
        "yield_strength_MPa": 200.0 + 5000.0 / math.sqrt(grain_size_nm),
        "porosity_percent": porosity_percent,
        "success": porosity_percent <= 5.0,
        "grain_size_nm": grain_size_nm,
    }


def _llm_next_cooling_rate(history: list[dict[str, Any]]) -> dict[str, Any]:
    input_prompt = json.dumps({"history": history}, ensure_ascii=False)
    t0 = time.perf_counter()
    raw = OpenAI().with_raw_response.responses.create(
        model=_DEFAULT_MODEL,
        input=(
            "You optimize a nickel-based superalloy heat-treatment simulation. "
            "Goal: maximize yield_strength_MPa while keeping porosity_percent <= 5. "
            "Given history, output only one numeric cooling_rate_K_per_min."
            f"\n\n{input_prompt}"
        ),
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    parsed = raw.parse()
    usage = getattr(parsed, "usage", None)
    output_prompt = parsed.output_text or ""
    return {
        "output_prompt": output_prompt,
        "elapsed_ms": round(elapsed_ms, 3),
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "cooling_rate_K_per_min": max(0.1, min(100.0, _parse_number(output_prompt) if _parse_number(output_prompt) is not None else 12.0)),
    }


def run_regular_simulation(
    initial_cooling_rate_K_per_min: float = 15.0,
    max_iterations: int = 5,
    duration_hours: float = 4.0,
    *,
    emit_artifacts: bool = True,
) -> dict[str, Any]:
    """Run regular simulation optimization with one OpenAI call per iteration."""
    history: list[dict[str, Any]] = []
    llm_calls: list[dict[str, Any]] = []
    current_rate = initial_cooling_rate_K_per_min
    for iteration in range(1, max_iterations + 1):
        simulation = _run_material_simulation(current_rate, duration_hours=duration_hours)
        history.append({"iteration": iteration, "cooling_rate_K_per_min": round(current_rate, 6), **simulation})
        llm_result = _llm_next_cooling_rate(history)
        llm_calls.append(
            {
                "iteration": iteration,
                "output_prompt": llm_result["output_prompt"],
                "elapsed_ms": llm_result["elapsed_ms"],
                "input_tokens": llm_result["input_tokens"],
                "output_tokens": llm_result["output_tokens"],
                "suggested_cooling_rate_K_per_min": llm_result["cooling_rate_K_per_min"],
            }
        )
        current_rate = llm_result["cooling_rate_K_per_min"]
    best_success = max((item for item in history if item["success"]), key=lambda item: item["yield_strength_MPa"], default=None)
    output = {
        "mode": "regular",
        "history": history,
        "best_success": best_success,
        "output_prompt": json.dumps(llm_calls[-1], ensure_ascii=False) if llm_calls else "",
        "elapsed_ms": round(sum(item["elapsed_ms"] for item in llm_calls), 3),
        "input_tokens": int(sum(item["input_tokens"] for item in llm_calls)),
        "output_tokens": int(sum(item["output_tokens"] for item in llm_calls)),
        "llm_calls": llm_calls,
    }
    if emit_artifacts:
        _append_test_results(
            component="sim_agent.regular",
            input_prompt=json.dumps(
                {"initial_cooling_rate_K_per_min": initial_cooling_rate_K_per_min, "max_iterations": max_iterations, "duration_hours": duration_hours},
                ensure_ascii=False,
            ),
            output=output,
        )
        _append_reasoning_input({"producer_file": "sim_agent.py", "event": "regular_simulation", **output})
    if os.environ.get("SANDBOX_PRINT", "1") == "1":
        print(
            "[sandbox.sim_agent:regular] "
            f"elapsed_ms={output['elapsed_ms']} input_tokens={output['input_tokens']} "
            f"output_tokens={output['output_tokens']} iterations={max_iterations}"
        )
    return output


def _select_real_sim_pairs(original_prompt: str) -> dict[str, Any]:
    number_sims_to_run = len(original_prompt) % 6 + 1
    input_payload = json.dumps({"original_prompt": original_prompt, "pairs": _DEFAULT_PREDEFINED_SIM_CALLS, "n": number_sims_to_run}, ensure_ascii=False)
    t0 = time.perf_counter()
    raw = OpenAI().with_raw_response.responses.create(
        model=_DEFAULT_MODEL,
        input=(
            "Return JSON only with schema {\"selected_pairs\": [[\"<composition>\", \"<supercell_size>\"]]}. "
            f"Select exactly the first n pairs from the list in this payload: {input_payload}"
        ),
        text={"format": {"type": "json_object"}},
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    parsed = raw.parse()
    usage = getattr(parsed, "usage", None)
    output_prompt = parsed.output_text or ""
    parsed_json = json.loads(output_prompt)
    return {
        "selected_pairs": parsed_json.get("selected_pairs", [list(pair) for pair in _DEFAULT_PREDEFINED_SIM_CALLS[:number_sims_to_run]])[:number_sims_to_run],
        "output_prompt": output_prompt,
        "elapsed_ms": round(elapsed_ms, 3),
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
    }


def _compute_real_sim_result(composition: str, supercell_size: int) -> dict[str, Any]:
    base = {"Al": 70.0, "Cu": 140.0, "Ni": 200.0, "Fe": 180.0, "W": 410.0, "Mo": 320.0}.get(composition, 150.0)
    return {
        "composition": composition,
        "supercell_size": supercell_size,
        "status": "ok",
        "c11_GPa": round(base + 4.5 * supercell_size, 3),
        "c12_GPa": round(base * 0.55 + 2.5 * supercell_size, 3),
        "c44_GPa": round(base * 0.42 + 1.8 * supercell_size, 3),
    }


def run_real_simulation(original_prompt: str, *, emit_artifacts: bool = True) -> dict[str, Any]:
    """Run real simulation mode with one OpenAI selection call + deterministic simulation outputs."""
    llm_select = _select_real_sim_pairs(original_prompt)
    simulation_results = [_compute_real_sim_result(item[0], int(item[1])) for item in llm_select["selected_pairs"]]
    output = {
        "mode": "real",
        "original_prompt": original_prompt,
        "selected_pairs": llm_select["selected_pairs"],
        "simulation_results": simulation_results,
        "output_prompt": llm_select["output_prompt"],
        "elapsed_ms": llm_select["elapsed_ms"],
        "input_tokens": llm_select["input_tokens"],
        "output_tokens": llm_select["output_tokens"],
    }
    if emit_artifacts:
        _append_test_results(component="sim_agent.real", input_prompt=original_prompt, output=output)
        _append_reasoning_input({"producer_file": "sim_agent.py", "event": "real_simulation", **output})
    if os.environ.get("SANDBOX_PRINT", "1") == "1":
        print(
            "[sandbox.sim_agent:real] "
            f"elapsed_ms={output['elapsed_ms']} input_tokens={output['input_tokens']} "
            f"output_tokens={output['output_tokens']} runs={len(simulation_results)}"
        )
    return output

