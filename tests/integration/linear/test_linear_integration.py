"""Detailed integration tests for the linear pipeline: extract → process → reasoning → orchestrator.
All tests use live LLM API calls. Marked as integration + linear. Skip when no API key."""
import os

import pytest

from src.linear import (
    extract,
    process,
    run,
    summarize,
    TASK_CONSTRAINT_VERIFICATION,
    TASK_FEATURE_EXTRACTION,
    TASK_NORMALIZATION,
    TASK_RISK_RANKING,
    TASK_SCHEMA_VALIDATION,
)
from src.linear.processor import TASKS

pytestmark = [
    pytest.mark.integration,
    pytest.mark.linear,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY") and not os.environ.get("ANTHROPIC_API_KEY"),
        reason="Set OPENAI_API_KEY or ANTHROPIC_API_KEY to run linear integration tests",
    ),
]

EXAMPLE_PROMPT = (
    "Simulate the thermomechanical performance of a nickel-based superalloy composed of 60% Ni, 20% Cr, 10% Co, and 10% Al by weight. "
    "Assume a dual-phase microstructure with gamma and gamma-prime phases, an average grain size of 850 nm, and 1.5% porosity. "
    "The alloy is processed using vacuum arc remelting followed by heat treatment at 1200 K for 4 hours and air cooling at 15 K/min."
)

REQUIRED_EXTRACTION_KEYS = (
    "material_system",
    "processing_conditions",
    "simulation_parameters",
    "computed_properties",
    "uncertainty_estimates",
)


def test_linear_integration_extract_returns_full_schema():
    """Extract from a detailed prompt; assert all top-level keys and nested structure."""
    result = extract(EXAMPLE_PROMPT)
    assert isinstance(result, dict)
    for key in REQUIRED_EXTRACTION_KEYS:
        assert key in result, f"Missing top-level key: {key}"
    assert isinstance(result["material_system"], dict)
    assert "material_name" in result["material_system"]
    assert "composition" in result["material_system"]
    assert isinstance(result["material_system"]["composition"], list)
    assert isinstance(result["processing_conditions"], dict)
    assert "heat_treatment" in result["processing_conditions"]
    assert isinstance(result["simulation_parameters"], dict)
    assert isinstance(result["computed_properties"], dict)
    assert isinstance(result["uncertainty_estimates"], dict)


def test_linear_integration_extract_composition_reflects_prompt():
    """Extract and verify composition list reflects the stated percentages."""
    result = extract(EXAMPLE_PROMPT)
    comp = result["material_system"]["composition"]
    assert len(comp) >= 4
    elements = {c["element"] for c in comp}
    assert "Ni" in elements
    assert "Cr" in elements


def test_linear_integration_process_schema_validation_shape():
    """Run schema_validation on extracted data; assert valid and issues keys."""
    data = extract(EXAMPLE_PROMPT)
    result = process(data, TASK_SCHEMA_VALIDATION)
    assert isinstance(result, dict)
    assert "valid" in result
    assert "issues" in result
    assert isinstance(result["issues"], list)


def test_linear_integration_process_constraint_verification_shape():
    """Run constraint_verification on extracted data; assert plausible and warnings."""
    data = extract(EXAMPLE_PROMPT)
    result = process(data, TASK_CONSTRAINT_VERIFICATION)
    assert "plausible" in result
    assert "warnings" in result
    assert isinstance(result["warnings"], list)


def test_linear_integration_process_feature_extraction_shape():
    """Run feature_extraction; assert alloy_class, functional_category, mechanism, dimensionality."""
    data = extract(EXAMPLE_PROMPT)
    result = process(data, TASK_FEATURE_EXTRACTION)
    assert "alloy_class" in result
    assert "functional_category" in result
    assert "dominant_mechanism" in result
    assert "dimensionality" in result


def test_linear_integration_process_normalization_shape():
    """Run normalization; assert same top-level keys and normalized-style content."""
    data = extract(EXAMPLE_PROMPT)
    result = process(data, TASK_NORMALIZATION)
    for key in REQUIRED_EXTRACTION_KEYS:
        assert key in result
    assert "material_system" in result
    assert "simulation_parameters" in result


def test_linear_integration_process_risk_ranking_shape():
    """Run risk_ranking; assert property_ranking and processing_ranking lists."""
    data = extract(EXAMPLE_PROMPT)
    result = process(data, TASK_RISK_RANKING)
    assert "property_ranking" in result
    assert "processing_ranking" in result
    assert isinstance(result["property_ranking"], list)
    assert isinstance(result["processing_ranking"], list)


def test_linear_integration_process_all_tasks_sequential():
    """Extract once, then run every processor task in sequence; assert no errors and correct shapes."""
    data = extract(EXAMPLE_PROMPT)
    for task in TASKS:
        result = process(data, task)
        assert isinstance(result, dict)
        assert len(result) > 0


def test_linear_integration_reasoning_summarize_produces_readable_text():
    """Extract, run two tasks, then summarize; assert summary is non-empty and human-readable."""
    data = extract(EXAMPLE_PROMPT)
    processing_results = {
        TASK_SCHEMA_VALIDATION: process(data, TASK_SCHEMA_VALIDATION),
        TASK_CONSTRAINT_VERIFICATION: process(data, TASK_CONSTRAINT_VERIFICATION),
    }
    summary = summarize(EXAMPLE_PROMPT, data, processing_results)
    assert isinstance(summary, str)
    assert len(summary.strip()) > 20


def test_linear_integration_orchestrator_returns_three_keys():
    """run() returns summary, extraction, and processing."""
    result = run("Simulate a nickel superalloy at 500–1300 K.", tasks=[TASK_SCHEMA_VALIDATION])
    assert "summary" in result
    assert "extraction" in result
    assert "processing" in result
    assert isinstance(result["summary"], str)
    assert isinstance(result["extraction"], dict)
    assert isinstance(result["processing"], dict)


def test_linear_integration_orchestrator_subset_tasks():
    """run() with explicit task list runs only those tasks."""
    result = run(
        "Simulate a nickel superalloy.",
        tasks=[TASK_SCHEMA_VALIDATION, TASK_CONSTRAINT_VERIFICATION],
    )
    assert set(result["processing"].keys()) == {TASK_SCHEMA_VALIDATION, TASK_CONSTRAINT_VERIFICATION}
    assert "summary" in result
    assert all(k in result["extraction"] for k in REQUIRED_EXTRACTION_KEYS)


def test_linear_integration_orchestrator_all_tasks():
    """run() with tasks=None runs all five processor tasks."""
    result = run(EXAMPLE_PROMPT, tasks=None)
    assert len(result["processing"]) == len(TASKS)
    for task in TASKS:
        assert task in result["processing"]
    assert len(result["summary"].strip()) > 0
