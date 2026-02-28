"""Orchestrates the linear pipeline: extract → process (one or more tasks) → reasoning summary."""
from src.linear.extractor import extract
from src.linear.processor import TASKS, process
from src.linear.reasoning import summarize


def run(input_text: str, tasks: list[str] | None = None) -> dict:
    """Run the full pipeline: extract from input, run processor tasks, produce human-readable summary.
    input_text: raw task description (e.g. material/simulation prompt).
    tasks: processor task names to run (schema_validation, constraint_verification, etc.). If None, all TASKS are run.
    Returns: {"summary": str, "extraction": dict, "processing": dict} with extraction result and per-task results.
    """
    extraction = extract(input_text)
    task_list = tasks if tasks is not None else list(TASKS)
    processing = {t: process(extraction, t) for t in task_list}
    return {
        "summary": summarize(input_text, extraction, processing),
        "extraction": extraction,
        "processing": processing,
    }
