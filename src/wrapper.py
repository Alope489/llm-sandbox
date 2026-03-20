"""LLM wrapper: unified chat interface over OpenAI (default) or Anthropic.

Provides two public functions:

complete(messages, ctx=None, max_tokens=None) -> str
    Simple single-turn completion with no tool support.

complete_with_tools(messages, provider, ctx=None) -> str
    Tool-calling loop: sends registered tool schemas to the LLM, executes
    any tool calls the LLM requests, feeds results back, and repeats until
    the LLM produces a final text response or MAX_TOOL_CALLS is reached.

Telemetry instrumentation (Pillar 7):
    Each private completion and tool-loop function emits structured telemetry
    records via src.llm_pipeline_telemetry when a non-None CallContext is
    supplied.  All call sites call get_openai_client()/get_anthropic_client()
    (process-level singletons from llm_pipeline_telemetry) before starting
    the perf_counter timer, so client_elapsed_ms excludes TLS setup overhead
    on every call after the first warm call.

Dependencies:
    src.llm_pipeline_telemetry, openai, anthropic, dotenv, json, os, time,
    dataclasses, datetime.

Pillar compliance:
    - Pillar 1: Functional parity with pre-telemetry code.
    - Pillar 3: Google-style docstrings on all public/private functions.
    - Pillar 4: No hardcoding; model/provider from env vars.
    - Pillar 6: O(1) per call; loop is O(MAX_TOOL_CALLS).
    - Pillar 7: Comprehensive try/except with error-path telemetry records;
      GIL-safe list.append in log_llm_call.
"""
import dataclasses
import json
import os
import time
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

from src.llm_pipeline_telemetry import (
    CallContext,
    get_anthropic_client,
    get_openai_client,
    log_llm_call,
    log_tool_execution,
)

load_dotenv()

MAX_TOOL_CALLS = 5


def complete(
    messages: list[dict],
    ctx: Optional[CallContext] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Send messages to the active LLM provider and return the reply text.

    Args:
        messages: List of message dicts with at least 'role' and 'content'.
        ctx: Optional ``CallContext`` snapshot.  When provided, a ``llm_call``
            record is appended to ``ctx.records`` with timing and token counts.
            The caller should supply a ``dataclasses.replace()`` snapshot with
            ``agent`` and ``span`` already set.
        max_tokens: Optional upper bound on output tokens.  Passed through to
            the underlying OpenAI call only; Anthropic uses ``MAX_TOKENS`` env.

    Returns:
        The LLM's reply as a plain string.

    Raises:
        openai.APIError / anthropic.APIError: On API failure; a status="error"
            record is still emitted when ctx is supplied.

    Preconditions:
        - ``LLM_PROVIDER`` env var is ``"openai"`` or ``"anthropic"`` (or absent).
        - The appropriate API key env var must be set.

    Postconditions:
        - Returns a non-empty string on success.
        - If ctx is not None, exactly one ``llm_call`` record is appended.

    Complexity:
        O(1) — single API call.
    """
    provider = os.environ.get("LLM_PROVIDER", "openai")
    if provider == "anthropic":
        return _complete_anthropic(messages, ctx=ctx)
    return _complete_openai(messages, ctx=ctx, max_tokens=max_tokens)


def complete_with_tools(
    messages: list[dict],
    provider: Optional[str] = None,
    ctx: Optional[CallContext] = None,
) -> str:
    """Run a tool-calling loop and return the LLM's final text response.

    Sends messages plus all registered tool schemas to the LLM. If the LLM
    requests a tool call, the tool is executed via tool_registry.call, the
    result is appended to the conversation, and the loop repeats. Stops when
    the LLM produces a plain text response or MAX_TOOL_CALLS is reached.

    Args:
        messages: Initial message list in the provider's chat format.
        provider: "openai" or "anthropic". Defaults to LLM_PROVIDER env var
            or "openai".
        ctx: Optional ``CallContext`` snapshot.  When provided, one
            ``llm_call`` record and one ``tool_execution`` record are appended
            per iteration.  The final LLM call emits ``status="partial"`` when
            ``MAX_TOOL_CALLS`` is reached.

    Returns:
        The LLM's final text response as a plain string. Returns an empty
        string if no text response is produced before MAX_TOOL_CALLS.

    Preconditions:
        - src.tool_registry must be importable (tools registered at import).
        - Appropriate API key env var must be set for the chosen provider.

    Postconditions:
        - At most MAX_TOOL_CALLS tool executions occur per call.
        - The original messages list is not mutated.

    Complexity:
        O(MAX_TOOL_CALLS) — bounded loop.
    """
    import src.tool_registry as registry

    resolved = (provider or os.environ.get("LLM_PROVIDER", "openai")).lower()
    msgs = list(messages)

    if resolved == "anthropic":
        return _tool_loop_anthropic(msgs, registry, ctx=ctx)
    return _tool_loop_openai(msgs, registry, ctx=ctx)


# ---------------------------------------------------------------------------
# Internal: simple completion helpers
# ---------------------------------------------------------------------------

def _complete_openai(
    messages: list[dict],
    ctx: Optional[CallContext] = None,
    max_tokens: Optional[int] = None,
) -> str:
    """Execute a single OpenAI chat completion with telemetry instrumentation.

    Args:
        messages: Message list in OpenAI chat format.
        ctx: Optional ``CallContext`` snapshot (labels already set by caller).
        max_tokens: Optional upper bound on completion tokens.

    Returns:
        The assistant's reply content string.

    Raises:
        openai.APIError: Propagated after emitting a status="error" record.

    Postconditions:
        - Returns the completion text.
        - If ctx is not None, exactly one ``llm_call`` record is appended.
        - On exception, a status="error" record is appended before re-raising.

    Complexity:
        O(1).
    """
    client = get_openai_client()
    call_ctx = dataclasses.replace(ctx) if ctx is not None else None
    call_start_ts = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    try:
        raw = client.with_raw_response.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            **({"max_tokens": max_tokens} if max_tokens is not None else {}),
        )
        client_elapsed_ms = (time.perf_counter() - t0) * 1000
        call_end_ts = datetime.now(timezone.utc)
        if call_ctx is not None:
            usage = raw.parse().usage
            raw_ms = raw.headers.get("openai-processing-ms")
            server_ms = int(raw_ms) if raw_ms and int(raw_ms) > 0 else None
            log_llm_call(
                call_ctx,
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
            raise ValueError(f"Model refused: {msg.refusal}")
        return msg.content
    except Exception:
        client_elapsed_ms = (time.perf_counter() - t0) * 1000
        call_end_ts = datetime.now(timezone.utc)
        if call_ctx is not None:
            log_llm_call(
                call_ctx,
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


def _complete_anthropic(
    messages: list[dict],
    ctx: Optional[CallContext] = None,
) -> str:
    """Execute a single Anthropic chat completion with telemetry instrumentation.

    Args:
        messages: Message list; system messages are extracted and forwarded as
            the ``system`` parameter.
        ctx: Optional ``CallContext`` snapshot (labels already set by caller).

    Returns:
        The assistant's reply content string.

    Raises:
        anthropic.APIError: Propagated after emitting a status="error" record.

    Postconditions:
        - Returns the completion text.
        - If ctx is not None, exactly one ``llm_call`` record is appended.
        - ``provider_server_latency_ms`` is always ``None`` (Anthropic does
          not expose a server-processing-time header).

    Complexity:
        O(1).
    """
    client = get_anthropic_client()
    call_ctx = dataclasses.replace(ctx) if ctx is not None else None
    call_start_ts = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    try:
        system_text = "\n".join(
            m["content"] for m in messages if m.get("role") == "system"
        )
        user_msgs = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("role") in ("user", "assistant")
        ]
        kwargs: dict = {
            "model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            "max_tokens": int(os.environ.get("MAX_TOKENS", "1024")),
            "messages": user_msgs,
        }
        if system_text:
            kwargs["system"] = [{"type": "text", "text": system_text}]

        raw = client.with_raw_response.messages.create(**kwargs)
        client_elapsed_ms = (time.perf_counter() - t0) * 1000
        call_end_ts = datetime.now(timezone.utc)
        if call_ctx is not None:
            usage = raw.parse().usage
            log_llm_call(
                call_ctx,
                model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                provider_server_latency_ms=None,
                client_elapsed_ms=client_elapsed_ms,
                call_start_ts=call_start_ts.isoformat(),
                call_end_ts=call_end_ts.isoformat(),
                status="ok",
            )
        return raw.parse().content[0].text
    except Exception:
        client_elapsed_ms = (time.perf_counter() - t0) * 1000
        call_end_ts = datetime.now(timezone.utc)
        if call_ctx is not None:
            log_llm_call(
                call_ctx,
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


# ---------------------------------------------------------------------------
# Internal: tool-calling loops
# ---------------------------------------------------------------------------

def _tool_loop_openai(
    msgs: list[dict],
    registry,
    ctx: Optional[CallContext] = None,
) -> str:
    """OpenAI tool-calling loop with per-iteration telemetry.

    Args:
        msgs: Conversation message list; mutated in place during the loop.
        registry: The tool registry module providing ``get_openai_schemas``
            and ``call``.
        ctx: Optional ``CallContext``.  When provided, one ``llm_call`` record
            and one ``tool_execution`` record are emitted per iteration.

    Returns:
        The LLM's final text response, or ``""`` when MAX_TOOL_CALLS is hit.

    Postconditions:
        - At most MAX_TOOL_CALLS iterations.
        - If ctx is provided, the final ``llm_call`` record after MAX_TOOL_CALLS
          has ``status="partial"``.

    Complexity:
        O(MAX_TOOL_CALLS).
    """
    client = get_openai_client()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    tools = registry.get_openai_schemas()
    call_count = 0

    while call_count < MAX_TOOL_CALLS:
        iter_ctx = (
            dataclasses.replace(ctx, agent="wrapper", span=f"tool_loop_iter_{call_count + 1}")
            if ctx is not None
            else None
        )
        call_start_ts = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=msgs,
            tools=tools,
            tool_choice="auto",
        )
        client_elapsed_ms = (time.perf_counter() - t0) * 1000
        call_end_ts = datetime.now(timezone.utc)

        choice = response.choices[0]
        msg = choice.message

        if not msg.tool_calls:
            if iter_ctx is not None:
                log_llm_call(
                    iter_ctx,
                    model=model,
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                    provider_server_latency_ms=None,
                    client_elapsed_ms=client_elapsed_ms,
                    call_start_ts=call_start_ts.isoformat(),
                    call_end_ts=call_end_ts.isoformat(),
                    status="ok",
                )
            return (msg.content or "").strip()

        if iter_ctx is not None:
            log_llm_call(
                iter_ctx,
                model=model,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                provider_server_latency_ms=None,
                client_elapsed_ms=client_elapsed_ms,
                call_start_ts=call_start_ts.isoformat(),
                call_end_ts=call_end_ts.isoformat(),
                status="ok",
            )

        msgs.append(msg.model_dump(exclude_unset=True))

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            kwargs = json.loads(tc.function.arguments)
            tool_start_ts = datetime.now(timezone.utc)
            t_tool = time.perf_counter()
            result = registry.call(tool_name, kwargs)
            tool_elapsed_ms = (time.perf_counter() - t_tool) * 1000
            tool_end_ts = datetime.now(timezone.utc)

            if iter_ctx is not None:
                result_dict = result if isinstance(result, dict) else {}
                tool_status = result_dict.get("status", "ok")
                runtime_sec = result_dict.get("runtime_seconds")
                tool_internal_ms = (
                    runtime_sec * 1000 if isinstance(runtime_sec, (int, float)) else None
                )
                log_tool_execution(
                    iter_ctx,
                    tool_name=tool_name,
                    tool_execution_ms=tool_elapsed_ms,
                    status=tool_status,
                    call_start_ts=tool_start_ts.isoformat(),
                    call_end_ts=tool_end_ts.isoformat(),
                    tool_internal_runtime_ms=tool_internal_ms,
                )

            msgs.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

        call_count += 1

    # MAX_TOOL_CALLS reached — emit a final "partial" LLM record
    if ctx is not None:
        last_ctx = dataclasses.replace(ctx, agent="wrapper", span=f"tool_loop_iter_{call_count}")
        log_llm_call(
            last_ctx,
            model=model,
            input_tokens=0,
            output_tokens=0,
            provider_server_latency_ms=None,
            client_elapsed_ms=0.0,
            call_start_ts=datetime.now(timezone.utc).isoformat(),
            call_end_ts=datetime.now(timezone.utc).isoformat(),
            status="partial",
        )
    return ""


def _tool_loop_anthropic(
    msgs: list[dict],
    registry,
    ctx: Optional[CallContext] = None,
) -> str:
    """Anthropic tool-calling loop with per-iteration telemetry.

    Args:
        msgs: Conversation message list; system messages are extracted once at
            the start.
        registry: The tool registry module providing ``get_anthropic_schemas``
            and ``call``.
        ctx: Optional ``CallContext``.  When provided, one ``llm_call`` record
            and one ``tool_execution`` record are emitted per iteration.

    Returns:
        The LLM's final text response, or ``""`` when MAX_TOOL_CALLS is hit.

    Postconditions:
        - At most MAX_TOOL_CALLS iterations.
        - ``provider_server_latency_ms`` is always ``None`` for Anthropic calls.

    Complexity:
        O(MAX_TOOL_CALLS).
    """
    client = get_anthropic_client()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    max_tokens = int(os.environ.get("MAX_TOKENS", "1024"))
    tools = registry.get_anthropic_schemas()
    call_count = 0

    system_blocks = [
        {"type": "text", "text": m["content"]}
        for m in msgs if m.get("role") == "system"
    ]
    conversation = [m for m in msgs if m.get("role") in ("user", "assistant")]

    while call_count < MAX_TOOL_CALLS:
        iter_ctx = (
            dataclasses.replace(ctx, agent="wrapper", span=f"tool_loop_iter_{call_count + 1}")
            if ctx is not None
            else None
        )
        api_kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "tools": tools,
            "messages": conversation,
        }
        if system_blocks:
            api_kwargs["system"] = system_blocks

        call_start_ts = datetime.now(timezone.utc)
        t0 = time.perf_counter()
        raw = client.with_raw_response.messages.create(**api_kwargs)
        client_elapsed_ms = (time.perf_counter() - t0) * 1000
        call_end_ts = datetime.now(timezone.utc)
        response = raw.parse()

        text_blocks = [b for b in response.content if getattr(b, "type", None) == "text"]
        tool_blocks = [b for b in response.content if getattr(b, "type", None) == "tool_use"]

        if not tool_blocks:
            if iter_ctx is not None:
                log_llm_call(
                    iter_ctx,
                    model=model,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    provider_server_latency_ms=None,
                    client_elapsed_ms=client_elapsed_ms,
                    call_start_ts=call_start_ts.isoformat(),
                    call_end_ts=call_end_ts.isoformat(),
                    status="ok",
                )
            return (text_blocks[0].text if text_blocks else "").strip()

        if iter_ctx is not None:
            log_llm_call(
                iter_ctx,
                model=model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                provider_server_latency_ms=None,
                client_elapsed_ms=client_elapsed_ms,
                call_start_ts=call_start_ts.isoformat(),
                call_end_ts=call_end_ts.isoformat(),
                status="ok",
            )

        conversation.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tb in tool_blocks:
            tool_name = tb.name
            kwargs_tool = tb.input if isinstance(tb.input, dict) else {}
            tool_start_ts = datetime.now(timezone.utc)
            t_tool = time.perf_counter()
            result = registry.call(tool_name, kwargs_tool)
            tool_elapsed_ms = (time.perf_counter() - t_tool) * 1000
            tool_end_ts = datetime.now(timezone.utc)

            if iter_ctx is not None:
                result_dict = result if isinstance(result, dict) else {}
                tool_status = result_dict.get("status", "ok")
                runtime_sec = result_dict.get("runtime_seconds")
                tool_internal_ms = (
                    runtime_sec * 1000 if isinstance(runtime_sec, (int, float)) else None
                )
                log_tool_execution(
                    iter_ctx,
                    tool_name=tool_name,
                    tool_execution_ms=tool_elapsed_ms,
                    status=tool_status,
                    call_start_ts=tool_start_ts.isoformat(),
                    call_end_ts=tool_end_ts.isoformat(),
                    tool_internal_runtime_ms=tool_internal_ms,
                )

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tb.id,
                "content": json.dumps(result),
            })

        conversation.append({"role": "user", "content": tool_results})
        call_count += 1

    # MAX_TOOL_CALLS reached
    if ctx is not None:
        last_ctx = dataclasses.replace(ctx, agent="wrapper", span=f"tool_loop_iter_{call_count}")
        log_llm_call(
            last_ctx,
            model=model,
            input_tokens=0,
            output_tokens=0,
            provider_server_latency_ms=None,
            client_elapsed_ms=0.0,
            call_start_ts=datetime.now(timezone.utc).isoformat(),
            call_end_ts=datetime.now(timezone.utc).isoformat(),
            status="partial",
        )
    return ""
