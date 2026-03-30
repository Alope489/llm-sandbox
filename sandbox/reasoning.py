"""Sandbox reasoning component (independent, OpenAI-only).

Rebuild of `src/linear/reasoning.py` for the LLM sandbox. This module produces a
human-readable summary from:
  - original input prompt
  - extractor output (structured extraction)
  - downstream agent outputs (e.g. processor tasks) including provenance

Return value is an Output dict that can be consumed downstream and includes:
  - summary: the human-readable summary text
  - output_prompt: model raw output text (same as summary here)
  - elapsed_ms, input_tokens, output_tokens

It also prints a one-line timing/tokens summary and appends a detailed record to
`test-results.md`.
"""

import json
import os
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_RESULTS_PATH = "test-results.md"

LINEAR_STRUCTURE = """
The pipeline has stages:
1. Extractor: turns raw task descriptions into structured data with keys:
   material_system, processing_conditions, simulation_parameters, computed_properties, uncertainty_estimates.
2. Processor: runs tasks on the extraction:
   schema_validation, constraint_verification, feature_extraction, normalization, risk_ranking.
""".strip()


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


def summarize(
    *,
    original_input: str,
    extraction: dict[str, Any],
    agent_outputs: list[dict[str, Any]],
    emit_artifacts: bool = True,
) -> dict[str, Any]:
    """Produce a human-readable summary with provenance-aware inputs."""
    output = _summarize_openai(original_input=original_input, extraction=extraction, agent_outputs=agent_outputs)
    if os.environ.get("SANDBOX_PRINT", "1") == "1":
        print(f"[sandbox.reasoning] elapsed_ms={output['elapsed_ms']} input_tokens={output['input_tokens']} output_tokens={output['output_tokens']}")
    if emit_artifacts:
        _append_test_results(component="reasoning", input_prompt=output["input_prompt"], output=output)
    return output


def _summarize_openai(*, original_input: str, extraction: dict[str, Any], agent_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    from openai import OpenAI

    system_prompt = (
        "You summarize the execution of a material/simulation pipeline.\n\n"
        "You are aware of the pipeline structure:\n"
        f"{LINEAR_STRUCTURE}\n\n"
        "You are also given downstream outputs that include a `producer_file` field. "
        "Use that to attribute results (e.g. 'Processor schema_validation found...').\n\n"
        "Write a concise, human-readable summary that: "
        "(1) states what the original input was; "
        "(2) lists the actions taken (extraction, then each downstream agent output you were given); "
        "(3) states key findings from each step (valid/plausible flags, warnings, rankings, etc.). "
        "Use plain language and short paragraphs or bullet points. No raw JSON in the summary."
    )

    input_prompt = json.dumps(
        {"original_input": original_input, "extraction": extraction, "agent_outputs": agent_outputs},
        indent=2,
        ensure_ascii=False,
    )
    t0 = time.perf_counter()
    raw = OpenAI().with_raw_response.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": input_prompt}],
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    parsed = raw.parse()
    msg = parsed.choices[0].message
    if getattr(msg, "refusal", None):
        raise ValueError(f"Model refused: {msg.refusal}")
    usage = parsed.usage
    summary = msg.content
    return {
        "summary": summary,
        "input_prompt": input_prompt,
        "output_prompt": summary,
        "elapsed_ms": round(elapsed_ms, 3),
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
    }

