"""
Optimizer agent: uses an LLM (OpenAI or Anthropic) to suggest cooling_rate_K_per_min
for the nickel-based superalloy material simulation. Goal: maximize yield_strength_MPa
while keeping porosity_percent below 5.0. Schema-aligned variable names.

Real-simulation phase
---------------------
Before the optimization loop, ``perform_real_simulation(original_prompt)`` may be
called by the pipeline dispatcher (``_execute_simulation`` in ``executor.py`` via
``CURRENT_SIMULATION_MODE=real_sim_mode``). No LLM call is made; instead, a fixed
pool of Docker-based elastic-constant simulations is run directly via
``compute_elastic_constants_tool``. The number of simulations executed is
``len(original_prompt) % 6 + 1`` (range: 1–6). Each JSON result is appended to
``self._current_sim_results`` (a ``list[str]``), which is injected into the system
prompt for every subsequent cooling rate suggestion.

Telemetry instrumentation:
    ``_call_openai``, ``_call_anthropic``, and ``get_llm_suggestion`` accept an
    optional ``ctx`` (``CallContext``).  ``get_llm_suggestion`` stamps
    ``agent="sim_agent"`` and ``span="sim_iter_<n>"`` per iteration.
    ``run_optimization_loop`` accepts and threads ``ctx`` through the loop.

Dependencies:
    os, re, dataclasses, time, datetime, dotenv, src.llm_pipeline_telemetry.

Pillar compliance:
    - Pillar 1: Functional parity with pre-telemetry code.
    - Pillar 4: No hardcoding; model / provider from env vars.
    - Pillar 7: try/except with error-path telemetry records; GIL-safe appends.
"""

import dataclasses
import os
import re
import time
from datetime import datetime, timezone
from typing import Callable, List, Optional, Tuple

from dotenv import load_dotenv

from src.llm_pipeline_telemetry import (
    CallContext,
    get_anthropic_client,
    get_openai_client,
    log_llm_call,
)

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

# Predefined calls to compute_elastic_constants_tool for the real-simulation
#  Each entry is a 2-tuple of strings (composition, supercell_size).
# The call site unpacks the tuple and casts supercell_size to int.  Parameters
# mirror the six per-element tests in tests/test_integration_lammps.py exactly.
_PREDEFINED_SIM_CALLS: tuple[tuple[str, str], ...] = (
    ("Al", "3"),
    ("Cu", "3"),
    ("Ni", "4"),
    ("Fe", "4"),
    ("W", "3"),
    ("Mo", "5"),
)


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
        # Expected to be a list of JSON strings
        self._current_sim_results: List[str] = []

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

    def get_llm_suggestion(self, ctx: Optional[CallContext] = None) -> float:
        """
        Ask the LLM for the next cooling_rate_K_per_min given the current history.
        Returns a float. On non-numeric or empty response, uses a cooldown fallback
        after optional retries.

        Args:
            ctx: Optional ``CallContext``.  When provided, a snapshot is created with
                ``agent="sim_agent"`` and ``span="sim_iter_<n>"``.

        Returns:
            Next cooling_rate_K_per_min as a float in [0.1, 100.0].
        """
        call_ctx = (
            dataclasses.replace(
                ctx,
                agent="sim_agent",
                span=f"sim_iter_{len(self.history) + 1}",
                iteration=len(self.history) + 1,
            )
            if ctx is not None
            else None
        )
        if self.provider == "anthropic":
            raw = self._call_anthropic(ctx=call_ctx)
        else:
            raw = self._call_openai(ctx=call_ctx)

        value = _parse_cooling_rate_from_response(raw)
        if value is not None:
            return max(0.1, min(100.0, value))

        for _ in range(MAX_PARSE_ATTEMPTS - 1):
            if self.provider == "anthropic":
                raw = self._call_anthropic(ctx=call_ctx)
            else:
                raw = self._call_openai(ctx=call_ctx)
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
        ctx: Optional[CallContext] = None,
    ) -> List[HistoryEntry]:
        """Run the loop: simulate -> log -> get LLM suggestion -> repeat for max_iterations.

        Args:
            initial_cooling_rate_K_per_min: Starting cooling rate.
            on_step: Optional callback invoked after each simulation step with
                (iteration_1based, cooling_rate_K_per_min, yield_strength_MPa, success).
            use_tools: Deprecated. Previously triggered an in-loop pre-computation
                phase; that responsibility has moved to the pipeline dispatcher
                (``_execute_simulation`` in ``executor.py`` via
                ``CURRENT_SIMULATION_MODE``). This parameter is accepted for
                backward compatibility but has no effect.
            ctx: Optional ``CallContext`` propagated from the caller.  When
                provided, one ``llm_call`` record per iteration is appended.

        Returns:
            Full history of (cooling_rate_K_per_min, yield_strength_MPa, success).

        Postconditions:
            - ``perform_real_simulation`` is never called by this method.
            - ``self.history`` is reset at entry and contains exactly
              ``max_iterations`` entries on return.

        Complexity:
            Θ(max_iterations) LLM calls + simulation calls.
        """
        self.history = []
        del use_tools  # intentionally unused — see deprecation note in docstring

        cooling_rate_K_per_min = initial_cooling_rate_K_per_min

        for i in range(self.max_iterations):
            yield_strength_MPa, success = self.run_simulation(cooling_rate_K_per_min)
            self.history.append((cooling_rate_K_per_min, yield_strength_MPa, success))
            if on_step is not None:
                on_step(i + 1, cooling_rate_K_per_min, yield_strength_MPa, success)

            cooling_rate_K_per_min = self.get_llm_suggestion(ctx=ctx)

        return self.history

    def run_and_report(
        self,
        initial_cooling_rate_K_per_min: float = DEFAULT_COOLING_RATE,
        use_tools: bool = False,
        ctx: Optional[CallContext] = None,
    ) -> Tuple[List[HistoryEntry], str]:
        """Run the optimization loop and return (history, output_string).

        Args:
            initial_cooling_rate_K_per_min: Starting cooling rate.
            use_tools: Deprecated. Forwarded to ``run_optimization_loop`` where
                it is also a no-op. Pre-computation is now controlled at the
                pipeline level via ``CURRENT_SIMULATION_MODE`` in
                ``executor.py``.
            ctx: Optional ``CallContext`` propagated to the loop.

        Returns:
            Tuple of (history, human-readable log string).

        Complexity:
            Θ(max_iterations) — delegates entirely to ``run_optimization_loop``.
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
            ctx=ctx,
        )
        return self.history, format_simulation_output(self.history, step_lines=lines)

    # ------------------------------------------------------------------
    # Public: real simulation execution
    # ------------------------------------------------------------------

    def perform_real_simulation(self, original_prompt: str) -> List[str]:
        """Run the deterministic real-simulation phase: execute 1–6 predefined elastic-constant simulations, determined by ``len(original_prompt) % 6 + 1``.

        Validates that the active LLM provider is ``"openai"``, then computes
        ``number_sims_to_run = len(original_prompt) % 6 + 1`` (range: 1–6) and
        indexes into ``_PREDEFINED_SIM_CALLS[0..number_sims_to_run-1]``.
        ``_PREDEFINED_SIM_CALLS`` is a pool of up to 6 ``(composition, supercell_size)``
        string 2-tuples (Al/3, Cu/3, Ni/4, Fe/4, W/3, Mo/5); the number of entries
        consumed varies by prompt length, providing intentional variability for testing
        without randomness:

        +---------+---------------+
        | element | supercell_size|
        +=========+===============+
        | Al      | 3             |
        | Cu      | 3             |
        | Ni      | 4             |
        | Fe      | 4             |
        | W       | 3             |
        | Mo      | 5             |
        +---------+---------------+

        Each call runs the elastic-lammps Docker container via
        ``compute_elastic_constants_tool``; ``supercell_size`` is cast from
        string to ``int`` at the call site. The result dict is JSON-serialised
        and appended to ``self._current_sim_results``.

        Args:
            original_prompt: The original user prompt forwarded from the
                pipeline dispatcher. Its length determines
                ``number_sims_to_run = len(original_prompt) % 6 + 1``
                (1–6 simulations), providing prompt-driven variability for
                testing purposes.

        Returns:
            The updated ``self._current_sim_results`` list; each entry is the
            JSON-serialised result dict from one ``compute_elastic_constants_tool``
            call.

        Raises:
            RuntimeError: If ``self.provider`` is not ``"openai"``. Real
                simulations are currently only supported with
                ``LLM_PROVIDER=openai``.

        Postconditions:
            - On success, between 1 and 6 entries are appended to
              ``self._current_sim_results`` (``len(original_prompt) % 6 + 1``
              entries, one per selected element in ``_PREDEFINED_SIM_CALLS``).
            - ``self._current_sim_results`` equals the returned list.

        Complexity:
            Θ(len(original_prompt) % 6 + 1) — between 1 and 6 Docker container
            invocations depending on prompt length.
        """
        if self.provider != "openai":
            raise RuntimeError(
                f"Real simulations are only supported with LLM_PROVIDER=openai. "
                f"Current provider: {self.provider!r}"
            )

        import json

        from src.tools.elastic_constants_lammps.host_wrapper import (
            compute_elastic_constants_tool,
        )

        number_sims_to_run = len(original_prompt) % 6 + 1
        for i in range(number_sims_to_run):
            current_sim_result = compute_elastic_constants_tool(
                _PREDEFINED_SIM_CALLS[i][0], _PREDEFINED_SIM_CALLS[i][1]
            )
            self._current_sim_results.append(json.dumps(current_sim_result))
        return self._current_sim_results

    def _system_prompt(self) -> str:
        """Return the system prompt, appending simulation results when available.

        Returns:
            Base SYSTEM_PROMPT, or SYSTEM_PROMPT with ``_current_sim_results``
            entries joined and appended. Never mutates the module-level
            SYSTEM_PROMPT constant.
        """
        if not self._current_sim_results:
            return SYSTEM_PROMPT
        context_text = "\n".join(self._current_sim_results)
        return (
            SYSTEM_PROMPT
            + "\n\nMaterial properties gathered before this run:\n"
            + context_text
        )

    # ------------------------------------------------------------------
    # Private: provider calls
    # ------------------------------------------------------------------

    def _call_openai(self, ctx: Optional[CallContext] = None) -> str:
        """Call OpenAI to get a cooling rate suggestion with telemetry.

        Args:
            ctx: Optional ``CallContext`` snapshot (labels already stamped by
                ``get_llm_suggestion``).

        Returns:
            Raw LLM response string.

        Postconditions:
            - If ctx is not None, exactly one ``llm_call`` record is appended.

        Complexity:
            O(1).
        """
        client = get_openai_client()
        user_content = self._format_history_for_prompt()
        call_start_ts = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        try:
            raw = client.with_raw_response.chat.completions.create(
                model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=50,
            )
            client_elapsed_ms = (time.perf_counter() - t0) * 1000
            call_end_ts = datetime.now(timezone.utc)
            if ctx is not None:
                usage = raw.parse().usage
                raw_ms = raw.headers.get("openai-processing-ms")
                server_ms = int(raw_ms) if raw_ms and int(raw_ms) > 0 else None
                log_llm_call(
                    ctx,
                    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                    input_tokens=usage.prompt_tokens,
                    output_tokens=usage.completion_tokens,
                    provider_server_latency_ms=server_ms,
                    client_elapsed_ms=client_elapsed_ms,
                    call_start_ts=call_start_ts.isoformat(),
                    call_end_ts=call_end_ts.isoformat(),
                    status="ok",
                )
            msg = raw.parse().choices[0].message
            if getattr(msg, "refusal", None):
                return ""
            return (msg.content or "").strip()
        except Exception:
            client_elapsed_ms = (time.perf_counter() - t0) * 1000
            call_end_ts = datetime.now(timezone.utc)
            if ctx is not None:
                log_llm_call(
                    ctx,
                    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
                    input_tokens=0,
                    output_tokens=0,
                    provider_server_latency_ms=None,
                    client_elapsed_ms=client_elapsed_ms,
                    call_start_ts=call_start_ts.isoformat(),
                    call_end_ts=call_end_ts.isoformat(),
                    status="error",
                )
            raise

    def _call_anthropic(self, ctx: Optional[CallContext] = None) -> str:
        """Call Anthropic to get a cooling rate suggestion with telemetry.

        Args:
            ctx: Optional ``CallContext`` snapshot (labels already stamped by
                ``get_llm_suggestion``).

        Returns:
            Raw LLM response string.

        Postconditions:
            - If ctx is not None, exactly one ``llm_call`` record is appended.
            - ``provider_server_latency_ms`` is always ``None`` (Anthropic).

        Complexity:
            O(1).
        """
        client = get_anthropic_client()
        user_content = self._format_history_for_prompt()
        call_start_ts = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        try:
            raw = client.with_raw_response.messages.create(
                model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
                max_tokens=int(os.environ.get("MAX_TOKENS", "128")),
                system=[{"type": "text", "text": self._system_prompt()}],
                messages=[{"role": "user", "content": user_content}],
            )
            client_elapsed_ms = (time.perf_counter() - t0) * 1000
            call_end_ts = datetime.now(timezone.utc)
            if ctx is not None:
                usage = raw.parse().usage
                log_llm_call(
                    ctx,
                    model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    provider_server_latency_ms=None,
                    client_elapsed_ms=client_elapsed_ms,
                    call_start_ts=call_start_ts.isoformat(),
                    call_end_ts=call_end_ts.isoformat(),
                    status="ok",
                )
            for block in raw.parse().content:
                if getattr(block, "type", None) == "text":
                    return (getattr(block, "text", "") or "").strip()
            return ""
        except Exception:
            client_elapsed_ms = (time.perf_counter() - t0) * 1000
            call_end_ts = datetime.now(timezone.utc)
            if ctx is not None:
                log_llm_call(
                    ctx,
                    model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
                    input_tokens=0,
                    output_tokens=0,
                    provider_server_latency_ms=None,
                    client_elapsed_ms=client_elapsed_ms,
                    call_start_ts=call_start_ts.isoformat(),
                    call_end_ts=call_end_ts.isoformat(),
                    status="error",
                )
            raise

    def _format_history_for_prompt(self) -> str:
        if not self.history:
            return "No previous attempts. Suggest the first cooling_rate_K_per_min (one number only)."
        lines = [
            f"cooling_rate_K_per_min={rate}, yield_strength_MPa={y:.2f}, success={ok}"
            for rate, y, ok in self.history
        ]
        return (
            "Previous attempts:\n"
            + "\n".join(lines)
            + "\n\nNext cooling_rate_K_per_min (reply with one number only):"
        )

    def _get_elastic_constants_params_from_LLM(
        self, original_prompt: str
    ) -> tuple[tuple[str, str], ...]:
        """Extract elastic-constants simulation parameters from the LLM response.

        Given the raw original user prompt, queries the OpenAI API and parses its
        response into an immutable sequence of (composition, supercell_size) string
        pairs suitable for direct consumption by ``compute_elastic_constants_tool``.

        Note:
            This method is currently intended to work with the OpenAI API only
            (``self.provider == "openai"``). Anthropic and other providers are not
            supported at this time.

        Args:
            original_prompt: The raw user prompt string forwarded from the
                pipeline dispatcher. Must be a non-empty string.

        Returns:
            A tuple of 2-tuples ``(composition, supercell_size)`` where both
            elements are strings.  Each inner tuple maps directly to one
            ``compute_elastic_constants_tool`` call parameter set.

        Raises:
            RuntimeError: If the API response contains no ``function_call``
                output item, indicating the provider failed to invoke the
                simulation parameter selection tool.
            RuntimeError: Planned — will be raised when ``self.provider`` is
                not ``"openai"``.

        Examples:
            >>> agent = SimulationAgent()
            >>> agent._get_elastic_constants_params_from_LLM(
            ...     "Compute Ni elastic constants"
            ... )
            NotImplementedError

        Pre-conditions:
            - ``original_prompt`` is a non-empty ``str``.
            - ``self.provider == "openai"``.

        Post-conditions:
            - Each inner tuple contains exactly two strings.
            - ``self._current_sim_results`` is not mutated by this method.

        Complexity:
            Θ(1) for the stub. Expected Θ(n) where n = number of parameter
            pairs extracted from the LLM response once implemented.
        """
        import json

        # call helper to build message body, and tool schemas message
        input_message, tools = (
            self._build_tool_message_for_sim_param_api_request_openAI(original_prompt)
        )
        # call OpenAI API to get the params list
        openAI_client = get_openai_client()
        params_list_response = openAI_client.responses.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            input=input_message,
            tools=tools,
            tool_choice="required",
            temperature=0.0,
        )
        # Find the first function_call item in the Responses API output list
        tool_call = next(
            (
                item
                for item in params_list_response.output
                if item.type == "function_call"
            ),
            None,
        )
        if tool_call is None:
            raise RuntimeError(
                f"LLM provider '{self.provider}' failed to select simulation parameters: "
                "no function_call tool invocation was found in the API response. "
                "Ensure the model supports tool use and that tool_choice='required' is honoured."
            )
        parsed = json.loads(tool_call.arguments)
        # parsed["selected_pairs"] is a list[list[str]], e.g. [["Al", "3"], ["Cu", "3"]]
        params_list = tuple((pair[0], pair[1]) for pair in parsed["selected_pairs"])
        return params_list

    def _build_tool_message_for_sim_param_api_request_openAI(
        self, original_prompt: str
    ) -> tuple[list[dict], list[dict]]:
        """Build the OpenAI function-calling tool definition for parameter extraction.

        Constructs and returns the list of tool/function schemas (in the OpenAI
        ``tools`` API format) that describe how the model should extract
        elastic-constants simulation parameters from ``original_prompt``.

        Note:
            This method is intended to work with the OpenAI API only. The
            returned schema conforms to the OpenAI ``tools`` request field
            format and is not compatible with other providers.

        Args:
            original_prompt: The raw user prompt string forwarded from the
                pipeline dispatcher. Used to inform or parameterise the tool
                schema as needed. Must be a non-empty string.

        Returns:
            A 2-tuple ``(messages, tools)`` where ``messages`` is a list of
            OpenAI chat message dicts and ``tools`` is a list of OpenAI
            function-calling tool schema dicts.

        Examples:
            >>> agent = SimulationAgent()
            >>> agent._build_tool_message_for_sim_param_api_request_openAI(
            ...     "Compute Ni elastic constants"
            ... )
            NotImplementedError

        Pre-conditions:
            - ``original_prompt`` is a non-empty ``str``.
            - ``self.provider == "openai"``.

        Post-conditions:
            - ``messages`` is a non-empty list of dicts with ``"role"`` and ``"content"`` keys.
            - ``tools`` is a non-empty list of OpenAI tool schema dicts.
            - ``self._current_sim_results`` is not mutated by this method.

        Complexity:
            Θ(1) for the stub. Expected Θ(k) where k = number of tool schemas
            constructed once implemented.
        """
        # Deterministically determine the number of simulations to run based on the original_prompt length
        # Up to 6 simulations can be run
        number_sims_to_run = len(original_prompt) % 6 + 1
        import json

        input_message = [
            {
                "role": "system",
                "content": f"You are a deterministic prefix extractor. Always call the tool and return exactly the first {number_sims_to_run} pairs.",
            },
            {
                "role": "user",
                "content": f"""Here is the complete ordered list of valid parameter pairs (JSON):{json.dumps(_PREDEFINED_SIM_CALLS, indent=2)}. Extract exactly the first {number_sims_to_run} pairs using the tool.""",
            },
        ]
        tools = [
            {
                "type": "function",
                "name": "select_first_pairs",
                "description": (
                    f"Return EXACTLY the first {number_sims_to_run} pairs from the JSON list. "
                    "Take them in the exact order they appear in the list (literal prefix). "
                    "Do NOT choose based on relevance, content, or the user's request semantics. "
                    f"This is pure mechanical extraction of the first {number_sims_to_run} pairs only. "
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "selected_pairs": {
                            "type": "array",
                            "description": f"Exactly the first {number_sims_to_run} pairs from the provided list",
                            "minItems": number_sims_to_run,  # enforces exactly number_sims_to_run
                            "maxItems": number_sims_to_run,  # enforces exactly number_sims_to_run
                            "items": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                        }
                    },
                    "required": ["selected_pairs"],
                    "additionalProperties": False,
                },
                "strict": True,
            }
        ]
        return (input_message, tools)
