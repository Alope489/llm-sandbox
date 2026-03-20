"""LLM agent that produces a human-readable summary of linear pipeline actions and results.

Telemetry instrumentation:
    ``summarize`` accepts an optional ``ctx`` parameter (``CallContext``).  It
    stamps ``agent="reasoning"`` and ``span="summarize"`` via
    ``dataclasses.replace()`` before forwarding ``ctx`` to ``complete()``.

Dependencies:
    json, dataclasses, src.wrapper, src.llm_pipeline_telemetry.

Pillar compliance:
    - Pillar 4: No hardcoding; all labels are literal constants here.
    - Pillar 7: All exceptions propagate from ``complete()`` to the caller.
"""
import dataclasses
import json
from typing import Optional

from src.llm_pipeline_telemetry import CallContext
from src.wrapper import complete

LINEAR_STRUCTURE = """
The linear pipeline in src/linear/ has two stages:
1. Extractor (extractor.py): extract(text) parses raw task descriptions into structured data with keys:
   material_system, processing_conditions, simulation_parameters, computed_properties, uncertainty_estimates.
2. Processor (processor.py): process(data, task) runs one of these tasks on the extraction:
   schema_validation (valid, issues), constraint_verification (plausible, warnings),
   feature_extraction (alloy_class, functional_category, dominant_mechanism, dimensionality),
   normalization (same keys with normalized values: fractions, temperatures_K array),
   risk_ranking (property_ranking, processing_ranking).
"""


def summarize(
    original_input: str,
    extraction: dict,
    processing_results: dict,
    ctx: Optional[CallContext] = None,
) -> str:
    """Produce a human-readable summary of actions taken and results obtained.

    Args:
        original_input: Raw task description passed to the pipeline.
        extraction: Output of ``extract()``.
        processing_results: Dict mapping task name -> output of
            ``process(data, task)``.
        ctx: Optional ``CallContext``.  When provided, a snapshot with
            ``agent="reasoning"`` and ``span="summarize"`` is forwarded to
            ``complete()``.

    Returns:
        A human-readable plain-text summary string.

    Raises:
        openai.APIError / anthropic.APIError: Propagated from ``complete()``.

    Postconditions:
        - If ctx is not None, exactly one ``llm_call`` record is appended.

    Complexity:
        O(1) — single LLM call.
    """
    call_ctx = (
        dataclasses.replace(ctx, agent="reasoning", span="summarize")
        if ctx is not None
        else None
    )
    return complete(
        [
            {
                "role": "system",
                "content": (
                    "You summarize the execution of a material/simulation pipeline. You are aware of the pipeline structure:\n"
                    + LINEAR_STRUCTURE
                    + "\nWrite a concise, human-readable summary that: (1) states what the original input was; "
                    "(2) lists the actions taken (extraction, then each processing task that was run); "
                    "(3) states what was obtained from each step (key findings, valid/plausible flags, rankings, etc.). "
                    "Use plain language and short paragraphs or bullet points. No raw JSON in the summary."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "original_input": original_input,
                        "extraction": extraction,
                        "processing_results": processing_results,
                    },
                    indent=2,
                ),
            },
        ],
        ctx=call_ctx,
    )
