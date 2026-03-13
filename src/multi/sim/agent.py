"""
Optimizer agent: uses an LLM (OpenAI or Anthropic) to suggest cooling_rate_K_per_min
for the nickel-based superalloy material simulation. Goal: maximize yield_strength_MPa
while keeping porosity_percent below 5.0. Schema-aligned variable names.

Pre-computation phase (Option A)
---------------------------------
When run_and_report or run_optimization_loop is called with use_tools=True, a single
pre-computation phase runs before the optimization loop. The LLM is offered all
registered tools and asked (but not commanded) to gather any material properties it
judges relevant. The result is stored in self._tool_context and injected into the
system prompt for every subsequent cooling rate suggestion.

Per-iteration timing
--------------------
self.timing is populated after each API call with elapsed_seconds, prompt_tokens,
completion_tokens, and tokens_per_second for throughput analysis.
"""
import os
import re
import time
from typing import Callable, List, Optional, Tuple

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

_PREFETCH_PROMPT = (
    "We are about to run a heat treatment optimization for the following alloy:\n\n"
    + MATERIAL_CONTEXT + "\n\n"
    "Before optimization begins, you have access to simulation tools. "
    "If you judge that any material properties (such as elastic constants) would help "
    "inform better cooling rate suggestions, please use the available tools to gather them now. "
    "If no tool data is needed, simply summarize what you already know. "
    "Your response will be injected into the system prompt for the entire optimization run."
)

DEFAULT_COOLING_RATE = 15.0
COOLDOWN_FALLBACK_RATE = 12.0  # fallback when LLM returns non-numeric
MAX_PARSE_ATTEMPTS = 2


def format_simulation_output(
    history: List[HistoryEntry],
    step_lines: Optional[List[str]] = None,
) -> str:
    """
    Format simulation run history as a string suitable for chat display.
    If step_lines is provided (e.g. from on_step), include them; else list history in one line per step.
    Appends a short summary (best successful run if any).
    """
    if not history:
        return "Simulation run: no iterations."
    body = "\n".join(
        step_lines
        if step_lines
        else [
            f"  Iteration {i}: cooling_rate={rate} K/min -> yield_strength={y:.2f} MPa, success={ok}"
            for i, (rate, y, ok) in enumerate(history, start=1)
        ]
    )
    successful = [(r, y) for r, y, ok in history if ok]
    if successful:
        best = max(successful, key=lambda x: x[1])
        summary = f"Best (successful): cooling_rate={best[0]} K/min -> yield_strength={best[1]:.2f} MPa."
    else:
        summary = "No successful runs (all had porosity > 5%)."
    return "Simulation run:\n" + body + "\n" + summary


def _parse_cooling_rate_from_response(raw: str) -> Optional[float]:
    """Extract a single number from LLM response; return None if not parseable."""
    if not raw or not isinstance(raw, str):
        return None
    s = raw.strip()
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

    Attributes:
        provider: Active LLM provider ("openai" or "anthropic").
        duration_hours: Simulation duration passed to run_material_simulation.
        max_iterations: Number of optimization loop iterations.
        history: List of (cooling_rate_K_per_min, yield_strength_MPa, success) tuples.
        timing: List of per-iteration timing dicts populated after each API call.
            Each dict has keys: iteration, elapsed_seconds, prompt_tokens,
            completion_tokens, tokens_per_second.
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
        self.timing: List[dict] = []
        self._tool_context: str = ""

    # ------------------------------------------------------------------
    # Public: tool-augmented queries
    # ------------------------------------------------------------------

    def ask_with_tools(self, query: str) -> str:
        """Ask a material science question using all registered tools.

        The LLM may call any registered tool autonomously if it judges the
        tool relevant to the query.

        Args:
            query: Natural-language question or task description.

        Returns:
            The LLM's final text response as a plain string.
        """
        from src.wrapper import complete_with_tools

        return complete_with_tools(
            [
                {"role": "system", "content": MATERIAL_CONTEXT},
                {"role": "user", "content": query},
            ],
            provider=self.provider,
        )

    # ------------------------------------------------------------------
    # Public: optimization loop
    # ------------------------------------------------------------------

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

        for _ in range(MAX_PARSE_ATTEMPTS - 1):
            if self.provider == "anthropic":
                raw = self._call_anthropic()
            else:
                raw = self._call_openai()
            value = _parse_cooling_rate_from_response(raw)
            if value is not None:
                return max(0.1, min(100.0, value))

        return COOLDOWN_FALLBACK_RATE

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
        on_step: Optional[Callable[[int, float, float, bool], None]] = None,
        use_tools: bool = False,
    ) -> List[HistoryEntry]:
        """
        Run the loop: simulate -> log -> get LLM suggestion -> repeat for max_iterations.

        Args:
            initial_cooling_rate_K_per_min: Starting cooling rate.
            on_step: Optional callback invoked after each simulation step with
                (iteration_1based, cooling_rate_K_per_min, yield_strength_MPa, success).
            use_tools: When True, runs a pre-computation phase before the loop
                where the LLM may call registered tools to gather material
                properties. Results are injected into the system prompt.
                Defaults to False to preserve backward compatibility.

        Returns:
            Full history of (cooling_rate_K_per_min, yield_strength_MPa, success).
        """
        self.history = []
        self.timing = []

        if use_tools:
            self._prefetch_tool_context()

        cooling_rate_K_per_min = initial_cooling_rate_K_per_min

        for i in range(self.max_iterations):
            yield_strength_MPa, success = self.run_simulation(cooling_rate_K_per_min)
            self.history.append((cooling_rate_K_per_min, yield_strength_MPa, success))
            if on_step is not None:
                on_step(i + 1, cooling_rate_K_per_min, yield_strength_MPa, success)

            cooling_rate_K_per_min = self.get_llm_suggestion()

        return self.history

    def run_and_report(
        self,
        initial_cooling_rate_K_per_min: float = DEFAULT_COOLING_RATE,
        use_tools: bool = False,
    ) -> Tuple[List[HistoryEntry], str]:
        """
        Run the optimization loop and return (history, output_string).

        Args:
            initial_cooling_rate_K_per_min: Starting cooling rate.
            use_tools: Passed through to run_optimization_loop. When True,
                enables the pre-computation tool-calling phase.

        Returns:
            Tuple of (history, human-readable log string).
        """
        lines: List[str] = []

        def on_step(iteration: int, rate: float, y_MPa: float, ok: bool) -> None:
            lines.append(
                f"  Iteration {iteration}: cooling_rate={rate} K/min -> yield_strength={y_MPa:.2f} MPa, success={ok}"
            )

        self.run_optimization_loop(
            initial_cooling_rate_K_per_min=initial_cooling_rate_K_per_min,
            on_step=on_step,
            use_tools=use_tools,
        )
        return self.history, format_simulation_output(self.history, step_lines=lines)

    # ------------------------------------------------------------------
    # Private: pre-computation phase
    # ------------------------------------------------------------------

    def _prefetch_tool_context(self) -> str:
        """Run the pre-computation tool-calling phase.

        Sends _PREFETCH_PROMPT to the LLM with all registered tool schemas.
        The LLM decides autonomously whether to call any tools. The resulting
        text summary is stored in self._tool_context.

        Returns:
            The LLM's response text (also stored as self._tool_context).
        """
        from src.wrapper import complete_with_tools

        self._tool_context = complete_with_tools(
            [{"role": "user", "content": _PREFETCH_PROMPT}],
            provider=self.provider,
        )
        return self._tool_context

    def _system_prompt(self) -> str:
        """Return the system prompt, appending tool context when available.

        Returns:
            Base SYSTEM_PROMPT, or SYSTEM_PROMPT with tool context appended.
            Never mutates the module-level SYSTEM_PROMPT constant.
        """
        if not self._tool_context:
            return SYSTEM_PROMPT
        return SYSTEM_PROMPT + "\n\nMaterial properties gathered before this run:\n" + self._tool_context

    # ------------------------------------------------------------------
    # Private: provider calls
    # ------------------------------------------------------------------

    def _call_openai(self) -> str:
        from openai import OpenAI

        user_content = self._format_history_for_prompt()
        t0 = time.perf_counter()
        response = OpenAI().chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": user_content},
            ],
            max_tokens=50,
        )
        elapsed = time.perf_counter() - t0
        self._record_timing(
            elapsed,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )
        msg = response.choices[0].message
        if getattr(msg, "refusal", None):
            return ""
        return (msg.content or "").strip()

    def _call_anthropic(self) -> str:
        from anthropic import Anthropic

        user_content = self._format_history_for_prompt()
        t0 = time.perf_counter()
        response = Anthropic().messages.create(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            max_tokens=int(os.environ.get("MAX_TOKENS", "128")),
            system=[{"type": "text", "text": self._system_prompt()}],
            messages=[{"role": "user", "content": user_content}],
        )
        elapsed = time.perf_counter() - t0
        self._record_timing(
            elapsed,
            response.usage.input_tokens,
            response.usage.output_tokens,
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

    def _record_timing(
        self,
        elapsed_seconds: float,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        """Append a timing entry to self.timing.

        Args:
            elapsed_seconds: Wall-clock time for the API call.
            prompt_tokens: Number of prompt/input tokens reported by the API.
            completion_tokens: Number of completion/output tokens reported.
        """
        self.timing.append({
            "iteration": len(self.timing) + 1,
            "elapsed_seconds": round(elapsed_seconds, 3),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tokens_per_second": round(
                completion_tokens / elapsed_seconds if elapsed_seconds > 0 else 0.0, 2
            ),
        })
