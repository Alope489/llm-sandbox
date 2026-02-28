"""LLM agent that produces a human-readable summary of linear pipeline actions and results."""
import json

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


def summarize(original_input: str, extraction: dict, processing_results: dict) -> str:
    """Produce a human-readable summary of actions taken and results obtained.
    original_input: raw task description passed to the pipeline.
    extraction: output of extract().
    processing_results: dict mapping task name -> output of process(data, task).
    """
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
        ]
    )
