"""Provider-aware file storage: OpenAI Responses API vector store or Anthropic in-memory KB.

For OpenAI, files are uploaded to a persistent vector store.  Subsequent queries
use the Responses API with the file_search tool so that answers are grounded in
uploaded documents and carry file_citation annotations.

For Anthropic, files are ingested into the module-level in-memory store managed
by knowledge_base.py, since Anthropic has no native file-search equivalent.

Module-level state (OpenAI only)
---------------------------------
_VECTOR_STORE_ID : str or None  -- ID of the active OpenAI vector store.

Telemetry instrumentation:
    ``query_openai`` accepts an optional ``ctx`` (``CallContext``) and emits a
    ``llm_call`` record via ``log_llm_call``.  ``provider_server_latency_ms``
    is populated from the ``openai-processing-ms`` response header returned by
    ``responses.create``.

Environment variables
---------------------
OPENAI_API_KEY : Required for OpenAI operations.
OPENAI_MODEL   : Model used for KB queries (default: gpt-4o-mini).
LLM_PROVIDER   : "openai" (default) or "anthropic".

Dependencies:
    os, dataclasses, time, datetime, dotenv, src.llm_pipeline_telemetry.

Pillar compliance:
    - Pillar 4: No hardcoding; model read from env vars at call time.
    - Pillar 7: try/except with error-path records; error records emitted
      before re-raising.
"""
import dataclasses
import os
import time
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

from src.llm_pipeline_telemetry import CallContext, get_openai_client, log_llm_call

load_dotenv()

_VECTOR_STORE_ID: Optional[str] = None


def clear_openai() -> None:
    """Reset the module-level OpenAI vector store ID reference.

    Returns:
        None

    Preconditions:
        - None. Safe to call even if no store has been created yet.

    Postconditions:
        - _VECTOR_STORE_ID is None.
        - The next call to _upload_openai() will create a fresh vector store
          rather than reusing the previously stored ID.
    """
    global _VECTOR_STORE_ID
    _VECTOR_STORE_ID = None


def upload_files(paths: list) -> None:
    """Ingest files into the appropriate store for the active LLM provider.

    Dispatches to the OpenAI vector store upload path or to the Anthropic
    in-memory knowledge base depending on the LLM_PROVIDER environment variable.

    Args:
        paths: A list of file path strings to upload or ingest.

    Returns:
        None

    Preconditions:
        - paths must be a non-empty list of valid, readable file paths.
        - For OpenAI: OPENAI_API_KEY must be set.
        - For Anthropic: OPENAI_API_KEY must be set (used for embeddings in
          knowledge_base.index).

    Postconditions:
        - For OpenAI: files are uploaded to the vector store;
          _VECTOR_STORE_ID is set.
        - For Anthropic: chunks derived from each file are appended to _STORE
          in knowledge_base.py.
    """
    if os.environ.get("LLM_PROVIDER", "openai") == "anthropic":
        from .knowledge_base import index
        index(paths)
    else:
        _upload_openai(paths)


def _upload_openai(paths: list) -> None:
    """Upload files to an OpenAI vector store.

    If no vector store exists (_VECTOR_STORE_ID is None), a new one named "kb"
    is created.  Files are uploaded in a single batch and polled until complete.

    Args:
        paths: A list of file path strings to upload. Each file is opened in
               binary mode.

    Returns:
        None

    Preconditions:
        - OPENAI_API_KEY must be set in the environment.
        - Each path in paths must point to an existing, readable file.

    Postconditions:
        - _VECTOR_STORE_ID is set to the ID of the active vector store.
        - All files in paths have been uploaded and indexed in the vector store.
        - All file streams opened during upload are closed.
    """
    global _VECTOR_STORE_ID
    client = get_openai_client()
    if _VECTOR_STORE_ID is None:
        _VECTOR_STORE_ID = client.vector_stores.create(name="kb").id
    file_streams = [open(p, "rb") for p in paths]
    try:
        client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=_VECTOR_STORE_ID,
            files=file_streams,
        )
    finally:
        for f in file_streams:
            f.close()


def _has_file_citation(response) -> bool:
    """Return True iff the response contains at least one file_citation annotation.

    Enforces the anti-hallucination contract: only grounded answers (where the model
    cited an uploaded document) are returned to the caller. Ungrounded answers cause
    query_openai to return "" and trigger the web-search fallback in kb_agent.ask().

    Uses SDK type discriminators (item.type, block.type) rather than hasattr guards
    — immune to future output types that might incidentally carry a content field.
    The early-return guard on response.output handles the empty-list edge case
    explicitly; Response.output is typed List (never None) and direct attribute
    access is safe because raw.parse() validates the schema before this helper
    is called — any schema violation raises before we reach this point.

    Args:
        response: A parsed openai.types.responses.Response object.

    Returns:
        True if at least one file_citation annotation object is present in any output_text
        block of any message item; False otherwise.

    Complexity:
        O(n) where n is the total number of annotation objects across all output items.
    """
    if not response.output:
        return False
    for item in response.output:
        if item.type != "message":
            continue
        for block in item.content:
            if block.type != "output_text":
                continue
            if any(ann.type == "file_citation" for ann in block.annotations):
                return True
    return False


def query_openai(query: str, ctx: Optional[CallContext] = None) -> str:
    """Query the OpenAI knowledge base via the Responses API and return grounded text.

    Issues a single responses.create call with the file_search tool attached to
    the active vector store.  A non-empty string is returned only when the response
    contains at least one file_citation annotation, confirming the answer is
    grounded in an uploaded document.  Ungrounded responses return "" so that
    kb_agent._ask_openai falls back to web search.

    Args:
        query: The natural-language question to ask.
        ctx: Optional ``CallContext`` snapshot.  When provided, a ``llm_call``
            record is emitted via ``log_llm_call`` with token counts from
            ``response.usage`` and ``provider_server_latency_ms`` populated from
            the ``openai-processing-ms`` response header.  Telemetry is always
            emitted for completed API calls, including calls where no citations
            were found (tokens were consumed regardless of citation outcome).
            Pass a snapshot with ``agent="kb_agent"`` and ``span="kb_query"``
            already set.

    Returns:
        The response output_text if at least one file_citation annotation is
        present; an empty string otherwise (including when _VECTOR_STORE_ID is
        None, when no citations are found, or on API error).

    Raises:
        Any exception raised by the OpenAI SDK is re-raised after logging an
        error-path llm_call record (input_tokens=0, output_tokens=0,
        status="error").

    Preconditions:
        - OPENAI_API_KEY must be set in the environment.
        - _VECTOR_STORE_ID must be set (upload_files called at least once); if
          None the function returns "" immediately without any API call or
          telemetry record.

    Postconditions:
        - _VECTOR_STORE_ID is not modified.
        - If ctx is not None and an API call was made, exactly one llm_call
          record is appended (status="ok" on success, "error" on exception).
        - No record is emitted when _VECTOR_STORE_ID is None (no API call made).

    Complexity:
        O(1) amortized — dominated by a single synchronous HTTP round-trip to the
        Responses API.  Post-response annotation traversal is O(n) where n is total
        annotation objects across all output items, negligible in practice.
    """
    if _VECTOR_STORE_ID is None:          # guard: no upload has happened yet
        return ""
    client = get_openai_client()
    call_ctx = dataclasses.replace(ctx) if ctx is not None else None
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    call_start_ts = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    try:
        raw = client.with_raw_response.responses.create(
            model=model,
            tools=[{"type": "file_search", "vector_store_ids": [_VECTOR_STORE_ID]}],
            input=query,
        )
        client_elapsed_ms = (time.perf_counter() - t0) * 1000
        call_end_ts = datetime.now(timezone.utc)
        parsed = raw.parse()              # parse exactly once
        raw_ms = raw.headers.get("openai-processing-ms")
        server_ms = int(raw_ms) if raw_ms and int(raw_ms) > 0 else None
        usage = getattr(parsed, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        if call_ctx is not None:
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
        # Enforce anti-hallucination contract — only grounded responses are returned
        return parsed.output_text if _has_file_citation(parsed) else ""
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
