from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv


_REPO_ROOT = Path(__file__).resolve().parents[3]
_ENV_EXAMPLE_PATH = _REPO_ROOT / ".env.example"
_ENV_PATH = _REPO_ROOT / ".env"


def _load_environment() -> None:
    """Load template defaults and local overrides for chemistry integrations."""
    if _ENV_EXAMPLE_PATH.exists():
        load_dotenv(dotenv_path=_ENV_EXAMPLE_PATH, override=False)
    load_dotenv(dotenv_path=_ENV_PATH, override=False)


_load_environment()


@dataclass
class ChemistryResult:
    """Typed chemistry payload consumed by wrapper and reasoning layers."""

    query: str
    route: str
    cid: Optional[int]
    molecular_weight: Optional[float]
    molecular_formula: Optional[str]
    is_valid_smiles: bool
    found: bool
    reasoning: str = ""
    error: Optional[str] = None


@dataclass
class ChemistryRunResult:
    """Typed run envelope with timing metadata and chemistry output."""

    output: ChemistryResult
    step_start: float
    step_end: float
    latency_seconds: float
    tokens_in: int = 0
    tokens_out: int = 0


class ChemCrowDriver:
    """Production chemistry driver backed by RDKit + PubChemPy."""

    def __init__(self, model: str | None = None, temp: float = 0.1) -> None:
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        self.temp = temp

    def _is_likely_smiles(self, text: str) -> bool:
        """Heuristic gate before RDKit parsing for wrapper/router usage."""
        candidate = text.strip()
        if not candidate or " " in candidate:
            return False
        if re.fullmatch(r"[A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+", candidate) is None:
            return False

        # Strong SMILES cues: structural symbols or ring indices.
        if re.search(r"[\[\]\(\)=#@+\-\\/%.0-9]", candidate):
            return True

        # Compact uppercase atom-token strings like CCO or NaCl are SMILES-like.
        uppercase_count = sum(1 for ch in candidate if ch.isupper())
        return uppercase_count >= 2 and len(candidate) <= 12

    def validate_smiles(self, smiles: str) -> ChemistryResult:
        """Validate a SMILES string with RDKit molecule parsing."""
        try:
            # RDKit is the production cheminformatics parser used for SMILES
            # validity checks (replaces previous mock chemistry behavior).
            from rdkit import Chem
        except ImportError as exc:
            return ChemistryResult(
                query=smiles,
                route="smiles",
                cid=None,
                molecular_weight=None,
                molecular_formula=None,
                is_valid_smiles=False,
                found=False,
                error=f"RDKit is not installed. Add the modern 'rdkit' package. ({exc})",
            )

        mol = Chem.MolFromSmiles(smiles)
        return ChemistryResult(
            query=smiles,
            route="smiles",
            cid=None,
            molecular_weight=None,
            molecular_formula=None,
            is_valid_smiles=mol is not None,
            found=mol is not None,
            error=None if mol is not None else "Invalid SMILES.",
        )

    def lookup_compound_by_name(self, name: str) -> ChemistryResult:
        """Fetch formula and molecular weight from PubChem for a common name."""
        try:
            # PubChemPy is the production client used to query PubChem compound
            # records for molecular formula and molecular weight.
            import pubchempy as pcp
        except ImportError as exc:
            return ChemistryResult(
                query=name,
                route="name_lookup",
                cid=None,
                molecular_weight=None,
                molecular_formula=None,
                is_valid_smiles=False,
                found=False,
                error=f"PubChemPy is not installed. Add the 'pubchempy' package. ({exc})",
            )

        compounds = pcp.get_compounds(name, "name")
        if not compounds:
            return ChemistryResult(
                query=name,
                route="name_lookup",
                cid=None,
                molecular_weight=None,
                molecular_formula=None,
                is_valid_smiles=False,
                found=False,
                error="No matching compound found in PubChem.",
            )

        best = compounds[0]
        return ChemistryResult(
            query=name,
            route="name_lookup",
            cid=best.cid,
            molecular_weight=best.molecular_weight,
            molecular_formula=best.molecular_formula,
            is_valid_smiles=False,
            found=True,
        )

    def should_use_reasoning(self, prompt: str, tool_result: ChemistryResult) -> bool:
        """Only invoke Sonnet for multi-step/interpretive chemistry reasoning."""
        if not prompt.strip():
            return False
        if tool_result.route == "name_lookup" and tool_result.found is False:
            return True
        lowered = prompt.lower()
        complex_keywords = (
            "why",
            "compare",
            "interpret",
            "explain",
            "safety",
            "hazard",
            "mechanism",
            "tradeoff",
            "implication",
        )
        return len(prompt.split()) >= 14 or any(keyword in lowered for keyword in complex_keywords)

    def _reason_with_sonnet(self, prompt: str, tool_result: ChemistryResult) -> str:
        """Optionally summarize chemistry results using Anthropic Sonnet 4.6."""
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            return "Reasoning skipped: missing ANTHROPIC_API_KEY."
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            return f"Reasoning skipped: anthropic package unavailable ({exc})."

        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=int(os.environ.get("MAX_TOKENS", "1024")),
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are a chemistry assistant. Explain results briefly.\n\n"
                        f"User request: {prompt}\n"
                        f"Tool result: {tool_result}"
                    ),
                }
            ],
        )
        return response.content[0].text.strip() if response.content else ""

    def run_chemistry_sim(self, prompt: str) -> ChemistryRunResult:
        """Run chemistry task with real tools and optional Sonnet reasoning."""
        step_start = time.time()

        if self._is_likely_smiles(prompt):
            chemistry_result = self.validate_smiles(prompt)
        else:
            chemistry_result = self.lookup_compound_by_name(prompt)

        if self.should_use_reasoning(prompt, chemistry_result):
            chemistry_result.reasoning = self._reason_with_sonnet(prompt, chemistry_result)

        step_end = time.time()
        return ChemistryRunResult(
            output=chemistry_result,
            step_start=step_start,
            step_end=step_end,
            latency_seconds=step_end - step_start,
            tokens_in=0,
            tokens_out=0,
        )