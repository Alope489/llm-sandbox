"""Sandbox knowledge-base agent (OpenAI-only, Responses API only, single-file)."""

import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_DEFAULT_RESULTS_PATH = "test-results.md"
_DEFAULT_REASONING_INPUT_PATH = os.path.join("sandbox", "prompts", "reasoning_input.txt")
_DEFAULT_KB_DATA_DIR = os.path.join("sandbox", "kb_data")
_SCALE_DOC_COUNT = {"small": 8, "medium": 40, "large": 120}
_BASE_REFERENCES = (
    "Inconel 718 is a nickel-based superalloy designed for fatigue and creep resistance at elevated temperature. "
    "Published summaries note service relevance up to about 700 C with strong oxidation resistance.",
    "NIST cryogenic property pages for Inconel 718 report thermal conductivity and linear expansion correlations "
    "across low-temperature ranges on the order of 4 K to 300 K.",
    "Ti-6Al-4V is an alpha-beta titanium alloy. Heat-treatment routes that vary cooling rate after solution treatment "
    "change alpha and beta morphology, affecting strength-ductility balance.",
    "Porous alumina ceramics generally show lower thermal conductivity as porosity increases because pores reduce "
    "effective conductive pathways through the solid phase.",
)
_ALLOYS = ("Inconel 718", "Ti-6Al-4V", "CoCrFeMnNi", "Al2O3 ceramic", "NiTi")
_PROCESSES = ("solution treatment", "aging", "arc melting", "sol-gel", "powder metallurgy")
_PROPERTIES = ("yield strength", "youngs modulus", "thermal conductivity", "specific heat", "thermal expansion", "density")
_STATE: dict[str, str | None] = {"vector_store_id": None}


def _legacy_excerpt(idx: int) -> str:
    path = Path(_DEFAULT_KB_DATA_DIR) / "small" / f"materials_{(idx % _SCALE_DOC_COUNT['small']) + 1:04d}.md"
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


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
        f"```json\n{json.dumps(output, ensure_ascii=False, indent=2, default=_json_default)}\n```\n"
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
        handle.write(json.dumps(payload, ensure_ascii=False, default=_json_default))
        handle.write("\n---\n")


def _doc_text(idx: int) -> str:
    legacy = _legacy_excerpt(idx)
    return (
        f"# Materials note {idx + 1}\n\n"
        f"Material family: {_ALLOYS[idx % len(_ALLOYS)]}\n"
        f"Processing focus: {_PROCESSES[idx % len(_PROCESSES)]}\n"
        f"Priority property: {_PROPERTIES[idx % len(_PROPERTIES)]}\n\n"
        f"Reference context: {_BASE_REFERENCES[idx % len(_BASE_REFERENCES)]}\n\n"
        "Extended context block: This note is intentionally verbose for retrieval testing. It couples material identity, "
        "processing intent, and property-centric questions into one document so vector search can disambiguate between "
        "closely related metallurgy topics. Include temperature dependence, microstructure evolution, and uncertainty "
        "communication in every downstream simulation plan so model output remains decision-useful across changing "
        "boundary conditions and sparse empirical anchors.\n\n"
        "Scenario narrative: Treat this entry as a pre-brief for a simulation run where operators must compare at least "
        "three candidate process windows. One window should bias toward conservative thermal gradients, one toward "
        "throughput, and one toward property uniformity. Record assumptions that link process controls to constitutive "
        "model parameters, and explicitly name what evidence would invalidate those assumptions in future iterations.\n\n"
        "Data quality and assumptions: Prioritize traceability over compactness. If a property trend is extrapolated "
        "outside measured ranges, write the extrapolation method, expected failure mode, and a confidence qualifier. "
        "If multiple publications disagree, capture both values and reason about test conditions rather than averaging "
        "blindly. When uncertainty propagates through the model, provide a concise explanation of whether uncertainty "
        "is dominated by input measurement error, model form error, or calibration sparsity.\n\n"
        f"Generated test guidance: For {_ALLOYS[idx % len(_ALLOYS)]}, design a simulation matrix that sweeps temperature "
        "and at least one processing parameter. Track uncertainty bands, boundary condition assumptions, mesh strategy, "
        "and calibration checkpoints. Include a pass-fail rubric for physical plausibility and another rubric for "
        "operational usefulness so reviewers can separate mathematically stable results from engineering-relevant results.\n\n"
        "Retrieval tags: materials-science, process-window, uncertainty-quantification, thermo-mechanical-modeling, "
        "decision-support, validation-plan\n\n"
        f"Legacy source excerpt:\n\n{legacy if legacy else 'No legacy note was available for this index.'}\n"
    )


def _build_materials_corpus(scale: str) -> dict[str, Any]:
    if scale.strip().lower() not in _SCALE_DOC_COUNT:
        raise ValueError("scale must be one of: small, medium, large")
    root = Path(_DEFAULT_KB_DATA_DIR) / scale.strip().lower()
    root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for idx in range(_SCALE_DOC_COUNT[scale.strip().lower()]):
        (root / f"materials_{idx + 1:04d}.md").write_text(_doc_text(idx), encoding="utf-8")
        paths.append(str(root / f"materials_{idx + 1:04d}.md"))
    return {
        "scale": scale.strip().lower(),
        "document_count": _SCALE_DOC_COUNT[scale.strip().lower()],
        "paths": paths,
        "local_store_bytes": sum(os.path.getsize(path) for path in paths),
    }


def clear_vector_store_reference() -> None:
    """Clear in-process vector store reference only."""
    _STATE["vector_store_id"] = None


def preload(scale: str = "small", *, emit_artifacts: bool = True) -> dict[str, Any]:
    """Preload scalable materials corpus into OpenAI vector store."""
    corpus = _build_materials_corpus(scale=scale)
    client = OpenAI()
    if _STATE["vector_store_id"] is None:
        _STATE["vector_store_id"] = client.vector_stores.create(name="sandbox-materials-kb").id
    before = client.vector_stores.retrieve(_STATE["vector_store_id"])
    t0 = time.perf_counter()
    streams = [open(path, "rb") for path in corpus["paths"]]
    try:
        batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=_STATE["vector_store_id"], files=streams
        )
    finally:
        for stream in streams:
            stream.close()
    output = {
        "scale": corpus["scale"],
        "document_count": corpus["document_count"],
        "local_store_bytes": corpus["local_store_bytes"],
        "vector_store_id": _STATE["vector_store_id"],
        "vector_store_usage_bytes_before": getattr(before, "usage_bytes", 0) or 0,
        "vector_store_usage_bytes_after": getattr(client.vector_stores.retrieve(_STATE["vector_store_id"]), "usage_bytes", 0) or 0,
        "file_counts": _json_default(getattr(batch, "file_counts", None)),
        "status": getattr(batch, "status", "unknown"),
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 3),
        "input_tokens": 0,
        "output_tokens": 0,
    }
    if emit_artifacts:
        _append_test_results(component="kb_agent.preload", input_prompt=json.dumps({"scale": scale}), output=output)
        _append_reasoning_input({"producer_file": "kb_agent.py", "event": "preload", "output": output})
    if os.environ.get("SANDBOX_PRINT", "1") == "1":
        print(
            "[sandbox.kb_agent:preload] "
            f"scale={output['scale']} docs={output['document_count']} "
            f"local_bytes={output['local_store_bytes']} usage_before={output['vector_store_usage_bytes_before']} "
            f"usage_after={output['vector_store_usage_bytes_after']} elapsed_ms={output['elapsed_ms']}"
        )
    return output


def _citation_file_ids(response: Any) -> list[str]:
    if not getattr(response, "output", None):
        return []
    ids: list[str] = []
    for item in response.output:
        if getattr(item, "type", None) == "message":
            for block in getattr(item, "content", []):
                if getattr(block, "type", None) == "output_text":
                    for annotation in getattr(block, "annotations", []):
                        if getattr(annotation, "type", None) == "file_citation" and getattr(annotation, "file_id", None):
                            ids.append(annotation.file_id)
    return sorted(set(ids))


def ask(query: str, *, emit_artifacts: bool = True) -> dict[str, Any]:
    """Answer a query from the preloaded OpenAI file-search knowledge base."""
    if _STATE["vector_store_id"] is None:
        raise ValueError("Vector store is not initialized. Call preload first.")
    t0 = time.perf_counter()
    raw = OpenAI().with_raw_response.responses.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        input=query,
        tools=[{"type": "file_search", "vector_store_ids": [_STATE["vector_store_id"]]}],
    )
    parsed = raw.parse()
    usage = getattr(parsed, "usage", None)
    output = {
        "query": query,
        "answer": parsed.output_text,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 3),
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "citation_file_ids": _citation_file_ids(parsed),
        "vector_store_id": _STATE["vector_store_id"],
    }
    if emit_artifacts:
        _append_test_results(component="kb_agent.ask", input_prompt=query, output=output)
        _append_reasoning_input({"producer_file": "kb_agent.py", "event": "ask", **output})
    if os.environ.get("SANDBOX_PRINT", "1") == "1":
        print(
            "[sandbox.kb_agent] "
            f"elapsed_ms={output['elapsed_ms']} input_tokens={output['input_tokens']} "
            f"output_tokens={output['output_tokens']} citations={len(output['citation_file_ids'])}"
        )
    return output

