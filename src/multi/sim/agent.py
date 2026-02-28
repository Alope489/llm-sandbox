"""
Optimizer agent: uses an LLM (OpenAI or Anthropic) to suggest cooling_rate_K_per_min
for the nickel-based superalloy material simulation. Goal: maximize yield_strength_MPa
while keeping porosity_percent below 5.0. Schema-aligned variable names.
"""
import os
import re
from typing import List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# History entry: (cooling_rate_K_per_min, yield_strength_MPa, success)
HistoryEntry = Tuple[float, float, bool]

MATERIAL_CONTEXT = (
    "Material system: Nickel-based superalloy, Ni-60, Cr-20, Co-10, Al-10 (dual-phase). "
    "Goal: maximize yield_strength_MPa while keeping porosity_percent below 5.0. "
    "Cooling rate (cooling_rate_K_per_min) affects grain refinement (higher strength at faster cooling) "
    "but very high cooling can increase porosity and cause failure (success=False)."
)

SYSTEM_PROMPT = f"""You are a Materials Informatics Specialist optimizing heat treatment for a nickel-based superalloy.

{MATERIAL_CONTEXT}

You will receive a history of previous attempts: each line gives cooling_rate_K_per_min, yield_strength_MPa, and success (True/False). Use this to suggest the next cooling_rate_K_per_min to try.

Respond with ONLY a single number: the next cooling_rate_K_per_min in K/min (e.g. 15 or 12.5). No units, no explanation, no markdown, no other text. Typical range: about 5 to 50 K/min; going too high risks porosity > 5% and failure."""

DEFAULT_COOLING_RATE = 15.0
COOLDOWN_FALLBACK_RATE = 12.0  # fallback when LLM returns non-numeric
MAX_PARSE_ATTEMPTS = 2


def _parse_cooling_rate_from_response(raw: str) -> Optional[float]:
    """Extract a single number from LLM response; return None if not parseable."""
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
    # Allow optional units like "K/min" or "K/min." at the end
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


class SimulationAgent:
    """
    Agent that runs the material simulation and uses an LLM to suggest the next
    cooling_rate_K_per_min. Maintains history and loops: simulate -> log -> get suggestion -> repeat.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        duration_hours: float = 4.0,
        max_iterations: int = 10,
    ):
        self.provider = (provider or os.environ.get("LLM_PROVIDER", "openai")).lower()
        self.duration_hours = duration_hours
        self.max_iterations = max_iterations
        self.history: List[HistoryEntry] = []

    def get_llm_suggestion(self) -> float:
        """
        Ask the LLM for the next cooling_rate_K_per_min given the current history.
        Returns a float. On non-numeric or empty response, uses a cooldown fallback
        after optional retries.
        """
        if self.provider == "anthropic":
            raw = self._call_anthropic()
        else:
            raw = self._call_openai()

        value = _parse_cooling_rate_from_response(raw)
        if value is not None:
            return max(0.1, min(100.0, value))

        # Cooldown / error handling: LLM gave non-numeric text
        for _ in range(MAX_PARSE_ATTEMPTS - 1):
            if self.provider == "anthropic":
                raw = self._call_anthropic()
            else:
                raw = self._call_openai()
            value = _parse_cooling_rate_from_response(raw)
            if value is not None:
                return max(0.1, min(100.0, value))

        return COOLDOWN_FALLBACK_RATE

    def _call_openai(self) -> str:
        from openai import OpenAI

        user_content = self._format_history_for_prompt()
        response = OpenAI().chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            max_tokens=50,
        )
        msg = response.choices[0].message
        if getattr(msg, "refusal", None):
            return ""
        return (msg.content or "").strip()

    def _call_anthropic(self) -> str:
        from anthropic import Anthropic

        user_content = self._format_history_for_prompt()
        response = Anthropic().messages.create(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            max_tokens=int(os.environ.get("MAX_TOKENS", "128")),
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
        )
        for block in response.content:
            if getattr(block, "type", None) == "text":
                return (getattr(block, "text", "") or "").strip()
        return ""

    def _format_history_for_prompt(self) -> str:
        if not self.history:
            return "No previous attempts. Suggest the first cooling_rate_K_per_min (one number only)."
        lines = [
            f"cooling_rate_K_per_min={rate}, yield_strength_MPa={y:.2f}, success={ok}"
            for rate, y, ok in self.history
        ]
        return "Previous attempts:\n" + "\n".join(lines) + "\n\nNext cooling_rate_K_per_min (reply with one number only):"

    def run_simulation(self, cooling_rate_K_per_min: float) -> Tuple[float, bool]:
        """Run the material simulation for the given cooling_rate_K_per_min. Returns (yield_strength_MPa, success)."""
        from src.multi.sim.simulation import run_material_simulation

        return run_material_simulation(
            cooling_rate_K_per_min=cooling_rate_K_per_min,
            duration_hours=self.duration_hours,
        )

    def run_optimization_loop(
        self,
        initial_cooling_rate_K_per_min: float = DEFAULT_COOLING_RATE,
    ) -> List[HistoryEntry]:
        """
        Run the loop: simulate -> log -> get LLM suggestion -> repeat for max_iterations.
        Returns the full history of (cooling_rate_K_per_min, yield_strength_MPa, success).
        """
        self.history = []
        cooling_rate_K_per_min = initial_cooling_rate_K_per_min

        for _ in range(self.max_iterations):
            yield_strength_MPa, success = self.run_simulation(cooling_rate_K_per_min)
            self.history.append((cooling_rate_K_per_min, yield_strength_MPa, success))

            cooling_rate_K_per_min = self.get_llm_suggestion()

        return self.history
