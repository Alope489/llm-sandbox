"""Unit tests for real chemistry routing and reasoning cost controls."""

from src.multi.chemistry.chem_driver import ChemCrowDriver, ChemistryResult


def test_should_use_reasoning_false_for_simple_lookup():
    """Simple one-token lookup should not trigger Sonnet reasoning."""
    driver = ChemCrowDriver()
    result = ChemistryResult(
        query="acetone",
        route="name_lookup",
        cid=180,
        molecular_weight=58.08,
        molecular_formula="C3H6O",
        is_valid_smiles=False,
        found=True,
    )
    assert driver.should_use_reasoning("acetone", result) is False


def test_should_use_reasoning_true_for_complex_prompt():
    """Long or interpretive prompts should trigger Sonnet reasoning."""
    driver = ChemCrowDriver()
    prompt = "Compare the safety implications and explain why this compound is risky."
    result = ChemistryResult(
        query="acetone",
        route="name_lookup",
        cid=180,
        molecular_weight=58.08,
        molecular_formula="C3H6O",
        is_valid_smiles=False,
        found=True,
    )
    assert driver.should_use_reasoning(prompt, result) is True


def test_run_chemistry_sim_skips_reasoning_for_simple_smiles(monkeypatch):
    """SMILES validation path keeps API costs low for simple tasks."""
    driver = ChemCrowDriver()

    monkeypatch.setattr(driver, "_is_likely_smiles", lambda _text: True)
    monkeypatch.setattr(
        driver,
        "validate_smiles",
        lambda _smiles: ChemistryResult(
            query="CCO",
            route="smiles",
            cid=None,
            molecular_weight=None,
            molecular_formula=None,
            is_valid_smiles=True,
            found=True,
        ),
    )
    monkeypatch.setattr(driver, "should_use_reasoning", lambda _prompt, _result: False)

    result = driver.run_chemistry_sim("CCO")
    assert result.output.is_valid_smiles is True
    assert result.output.reasoning == ""


def test_run_chemistry_sim_calls_reasoning_for_complex_request(monkeypatch):
    """Complex requests should include Sonnet-generated reasoning text."""
    driver = ChemCrowDriver()

    monkeypatch.setattr(driver, "_is_likely_smiles", lambda _text: False)
    monkeypatch.setattr(
        driver,
        "lookup_compound_by_name",
        lambda _name: ChemistryResult(
            query="acetone",
            route="name_lookup",
            cid=180,
            molecular_weight=58.08,
            molecular_formula="C3H6O",
            is_valid_smiles=False,
            found=True,
        ),
    )
    monkeypatch.setattr(driver, "should_use_reasoning", lambda _prompt, _result: True)
    monkeypatch.setattr(driver, "_reason_with_sonnet", lambda _prompt, _result: "Chem reasoning.")

    result = driver.run_chemistry_sim("Explain acetone safety and handling tradeoffs.")
    assert result.output.reasoning == "Chem reasoning."
    assert result.output.molecular_formula == "C3H6O"
