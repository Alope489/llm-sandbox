"""Knowledge-base agent: query KB first, fall back to LLM-native web search if empty.

The single public entry point, ask(), dispatches to a provider-specific
implementation based on the LLM_PROVIDER environment variable.  Each
implementation follows the same two-step strategy:

1. Search the knowledge base (high-veracity data source).
2. If no results are found, issue a web search through the LLM's own native
   web-search tool — no third-party search service is used.

Provider implementations
------------------------
OpenAI   : Uses file_store.query_openai for KB lookup (OpenAI Assistants with
           file_search).  Falls back to the OpenAI Responses API with the
           web_search_preview tool.
Anthropic: Uses knowledge_base.search for KB lookup (in-memory vector store).
           Falls back to the Anthropic Messages API with the
           web_search_20250305 tool.

Telemetry instrumentation:
    ``ask``, ``_ask_openai``, ``_ask_anthropic``, ``_web_search_openai``, and
    ``_web_search_anthropic`` all accept an optional ``ctx`` (``CallContext``).
    Span labels: ``"kb_query"`` for the file_store path and ``"web_search"``
    for the web-search fallback.

Environment variables
---------------------
LLM_PROVIDER    : "openai" (default) or "anthropic".
OPENAI_API_KEY  : Required for OpenAI paths.
ANTHROPIC_API_KEY: Required for Anthropic paths.
OPENAI_MODEL    : Model for OpenAI web search (default: gpt-4o-mini).
ANTHROPIC_MODEL : Model for Anthropic web search (default: claude-sonnet-4-6).
MAX_TOKENS      : Max tokens for Anthropic responses (default: 1024).

Dependencies:
    os, dataclasses, time, datetime, dotenv, openai, anthropic,
    src.llm_pipeline_telemetry.

Pillar compliance:
    - Pillar 4: No hardcoding; all config from env vars.
    - Pillar 7: try/except with error-path records; error records emitted
      before re-raising.
"""
import dataclasses
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
)

load_dotenv()


def ask(query: str, ctx: Optional[CallContext] = None) -> str:
    """Answer a query using KB lookup, with automatic web-search fallback.

    Dispatches to the OpenAI or Anthropic implementation based on LLM_PROVIDER.

    Args:
        query: The natural-language question to answer.
        ctx: Optional ``CallContext``.  Span labels are applied by the private
            helpers.

    Returns:
        A string response from either the knowledge base or the LLM's web search.
        Returns an empty string only if both the KB and the web search return
        no usable content.

    Preconditions:
        - LLM_PROVIDER must be "openai" or "anthropic" (or absent for default).
        - The appropriate API key must be set for the active provider.

    Postconditions:
        - If the KB contains relevant results, the response is derived from those
          results without a live web request.
        - If the KB is empty or returns no results, a live web search is performed
          and its output is returned.
        - The KB store is not modified by this call.

    Complexity:
        O(1) — one or two API calls.
    """
    if os.environ.get("LLM_PROVIDER", "openai") == "anthropic":
        return _ask_anthropic(query, ctx=ctx)
    return _ask_openai(query, ctx=ctx)


def _ask_openai(query: str, ctx: Optional[CallContext] = None) -> str:
    """Answer a query for the OpenAI provider: file search first, then web search.

    Args:
        query: The natural-language question to answer.
        ctx: Optional ``CallContext``.

    Returns:
        The OpenAI Assistant's response text if it returns a file-cited answer;
        otherwise the output of _web_search_openai().

    Preconditions:
        - OPENAI_API_KEY must be set in the environment.

    Postconditions:
        - Returns a non-empty string when either the KB or the web search
          produces a result.
        - The KB state in file_store is not modified.

    Complexity:
        O(1).
    """
    from .file_store import query_openai
    kb_ctx = (
        dataclasses.replace(ctx, agent="kb_agent", span="kb_query")
        if ctx is not None
        else None
    )
    result = query_openai(query, ctx=kb_ctx)
    if result:
        return result
    ws_ctx = (
        dataclasses.replace(ctx, agent="kb_agent", span="web_search")
        if ctx is not None
        else None
    )
    return _web_search_openai(query, ctx=ws_ctx)


def _ask_anthropic(query: str, ctx: Optional[CallContext] = None) -> str:
    """Answer a query for the Anthropic provider: in-memory KB first, then web search.

    Args:
        query: The natural-language question to answer.
        ctx: Optional ``CallContext``.

    Returns:
        The augmented completion text from complete_with_knowledge() if the KB
        returns results; otherwise the output of _web_search_anthropic().

    Preconditions:
        - OPENAI_API_KEY must be set (used for KB embedding via knowledge_base).
        - ANTHROPIC_API_KEY must be set for both KB completion and web search.

    Postconditions:
        - Returns a non-empty string when either the KB or the web search
          produces a result.
        - The KB store in knowledge_base is not modified.

    Complexity:
        O(1).
    """
    from .knowledge_base import search
    from .wrapper import complete_with_knowledge
    if search(query, top_k=5):
        return complete_with_knowledge(
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": query}],
            query,
        )
    ws_ctx = (
        dataclasses.replace(ctx, agent="kb_agent", span="web_search")
        if ctx is not None
        else None
    )
    return _web_search_anthropic(query, ctx=ws_ctx)


def _web_search_openai(query: str, ctx: Optional[CallContext] = None) -> str:
    """Perform a web search using the OpenAI Responses API with web_search_preview.

    Args:
        query: The search query string.
        ctx: Optional ``CallContext`` snapshot (labels already set by caller).

    Returns:
        The full text output of the OpenAI response after web search grounding.
        Returns an empty string if the API returns no output_text.

    Preconditions:
        - OPENAI_API_KEY must be set in the environment.
        - OPENAI_MODEL must reference a model that supports the web_search_preview
          tool (default: gpt-4o-mini).

    Postconditions:
        - A live HTTP request is made to the OpenAI Responses API.
        - If ctx is not None, one ``llm_call`` record is appended.

    Complexity:
        O(1).
    """
    client = get_openai_client()
    call_ctx = dataclasses.replace(ctx) if ctx is not None else None
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    call_start_ts = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    try:
        raw = client.with_raw_response.responses.create(
            model=model,
            tools=[{"type": "web_search_preview"}],
            input=query,
        )
        client_elapsed_ms = (time.perf_counter() - t0) * 1000
        call_end_ts = datetime.now(timezone.utc)
        if call_ctx is not None:
            parsed = raw.parse()
            raw_ms = raw.headers.get("openai-processing-ms")
            server_ms = int(raw_ms) if raw_ms and int(raw_ms) > 0 else None
            usage = getattr(parsed, "usage", None)
            input_tokens = getattr(usage, "input_tokens", 0) or 0
            output_tokens = getattr(usage, "output_tokens", 0) or 0
            log_llm_call(
                call_ctx,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                provider_server_latency_ms=server_ms,
                client_elapsed_ms=client_elapsed_ms,
                call_start_ts=call_start_ts.isoformat(),
                call_end_ts=call_end_ts.isoformat(),
                status="ok",
            )
        return raw.parse().output_text
    except Exception:
        client_elapsed_ms = (time.perf_counter() - t0) * 1000
        call_end_ts = datetime.now(timezone.utc)
        if call_ctx is not None:
            log_llm_call(
                call_ctx,
                model=model,
                input_tokens=0,
                output_tokens=0,
                provider_server_latency_ms=None,
                client_elapsed_ms=client_elapsed_ms,
                call_start_ts=call_start_ts.isoformat(),
                call_end_ts=call_end_ts.isoformat(),
                status="error",
            )
        raise


def _web_search_anthropic(query: str, ctx: Optional[CallContext] = None) -> str:
    """Perform a web search using the Anthropic Messages API with web_search_20250305.

    Args:
        query: The search query string.
        ctx: Optional ``CallContext`` snapshot (labels already set by caller).

    Returns:
        The text content of the first TextBlock in the response that has a
        "text" attribute.  Returns an empty string if no such block exists.

    Preconditions:
        - ANTHROPIC_API_KEY must be set in the environment.

    Postconditions:
        - A live HTTP request is made to the Anthropic Messages API.
        - If ctx is not None, one ``llm_call`` record is appended.
        - ``provider_server_latency_ms`` is always ``None`` (Anthropic).

    Complexity:
        O(1).
    """
    client = get_anthropic_client()
    call_ctx = dataclasses.replace(ctx) if ctx is not None else None
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    call_start_ts = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    try:
        raw = client.with_raw_response.messages.create(
            model=model,
            max_tokens=int(os.environ.get("MAX_TOKENS", "1024")),
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": query}],
        )
        client_elapsed_ms = (time.perf_counter() - t0) * 1000
        call_end_ts = datetime.now(timezone.utc)
        if call_ctx is not None:
            usage = raw.parse().usage
            log_llm_call(
                call_ctx,
                model=model,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                provider_server_latency_ms=None,
                client_elapsed_ms=client_elapsed_ms,
                call_start_ts=call_start_ts.isoformat(),
                call_end_ts=call_end_ts.isoformat(),
                status="ok",
            )
        return next(
            (b.text for b in raw.parse().content if hasattr(b, "text")),
            "",
        )
    except Exception:
        client_elapsed_ms = (time.perf_counter() - t0) * 1000
        call_end_ts = datetime.now(timezone.utc)
        if call_ctx is not None:
            log_llm_call(
                call_ctx,
                model=model,
                input_tokens=0,
                output_tokens=0,
                provider_server_latency_ms=None,
                client_elapsed_ms=client_elapsed_ms,
                call_start_ts=call_start_ts.isoformat(),
                call_end_ts=call_end_ts.isoformat(),
                status="error",
            )
        raise
