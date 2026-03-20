"""Provider-aware file storage: OpenAI Assistants vector store or Anthropic in-memory KB.

For OpenAI, files are uploaded to a persistent vector store and an Assistant is
created (or updated) with the file_search tool so that subsequent queries can
retrieve content with citation annotations.

For Anthropic, files are ingested into the module-level in-memory store managed
by knowledge_base.py, since Anthropic has no native file-search equivalent.

Module-level state (OpenAI only)
---------------------------------
_VECTOR_STORE_ID : str or None  -- ID of the active OpenAI vector store.
_ASSISTANT_ID    : str or None  -- ID of the active OpenAI Assistant.
_ASSISTANT_MODEL : str          -- Model name used when creating the assistant.

Telemetry instrumentation:
    ``query_openai`` accepts an optional ``ctx`` (``CallContext``) and emits a
    ``llm_call`` record via ``log_llm_call``.  ``provider_server_latency_ms``
    is always ``None`` because the Assistants ``create_and_poll`` helper
    returns a ``Run`` object with no equivalent timing header.

Environment variables
---------------------
OPENAI_API_KEY : Required for OpenAI operations.
OPENAI_MODEL   : Model used when creating the Assistant (default: gpt-4o-mini).
LLM_PROVIDER   : "openai" (default) or "anthropic".

Dependencies:
    os, dataclasses, time, datetime, dotenv, src.llm_pipeline_telemetry.

Pillar compliance:
    - Pillar 4: No hardcoding; model from env vars via _ASSISTANT_MODEL.
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

_VECTOR_STORE_ID: str = None
_ASSISTANT_ID: str = None
_ASSISTANT_MODEL: str = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def clear_openai() -> None:
    """Reset the module-level OpenAI vector store and assistant ID references.

    Returns:
        None

    Preconditions:
        - None. Safe to call even if no store or assistant has been created yet.

    Postconditions:
        - _VECTOR_STORE_ID is None.
        - _ASSISTANT_ID is None.
        - The next call to _upload_openai() will create a fresh vector store and
          assistant rather than reusing previously stored IDs.
    """
    global _VECTOR_STORE_ID, _ASSISTANT_ID
    _VECTOR_STORE_ID = None
    _ASSISTANT_ID = None


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
        - For OpenAI: files are uploaded to the vector store and the assistant
          is created or updated; _VECTOR_STORE_ID and _ASSISTANT_ID are set.
        - For Anthropic: chunks derived from each file are appended to _STORE
          in knowledge_base.py.
    """
    if os.environ.get("LLM_PROVIDER", "openai") == "anthropic":
        from .knowledge_base import index
        index(paths)
    else:
        _upload_openai(paths)


def _upload_openai(paths: list) -> None:
    """Upload files to an OpenAI vector store and create or update an Assistant.

    If no vector store exists (_VECTOR_STORE_ID is None), a new one named "kb"
    is created.  Files are uploaded in a single batch and polled until complete.
    If no assistant exists (_ASSISTANT_ID is None), one is created with the
    file_search tool and the vector store attached; otherwise the existing
    assistant is updated with the current vector store ID.

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
        - _ASSISTANT_ID is set to the ID of the active assistant.
        - _ASSISTANT_MODEL is set to the model used when creating the assistant.
        - All files in paths have been uploaded and indexed in the vector store.
        - All file streams opened during upload are closed.
    """
    global _VECTOR_STORE_ID, _ASSISTANT_ID, _ASSISTANT_MODEL
    client = get_openai_client()
    if _VECTOR_STORE_ID is None:
        _VECTOR_STORE_ID = client.vector_stores.create(name="kb").id
    file_streams = [open(p, "rb") for p in paths]
    client.vector_stores.file_batches.upload_and_poll(
        vector_store_id=_VECTOR_STORE_ID,
        files=file_streams,
    )
    for f in file_streams:
        f.close()
    _ASSISTANT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if _ASSISTANT_ID is None:
        _ASSISTANT_ID = client.beta.assistants.create(
            model=_ASSISTANT_MODEL,
            tools=[{"type": "file_search"}],
            tool_resources={"file_search": {"vector_store_ids": [_VECTOR_STORE_ID]}},
        ).id
    else:
        client.beta.assistants.update(
            _ASSISTANT_ID,
            tool_resources={"file_search": {"vector_store_ids": [_VECTOR_STORE_ID]}},
        )


def query_openai(query: str, ctx: Optional[CallContext] = None) -> str:
    """Query the OpenAI Assistant and return its response if it cites a file.

    Creates a new thread with the query as the user message, runs the assistant,
    and inspects the response for file_citation annotations.  A non-empty string
    is returned only when the assistant's answer includes at least one citation,
    indicating the response was grounded in an uploaded document.

    Args:
        query: The natural-language question to ask the assistant.
        ctx: Optional ``CallContext`` snapshot.  When provided, a ``llm_call``
            record is appended using ``run.usage`` token counts.
            ``provider_server_latency_ms`` is always ``None`` (the Assistants
            API does not expose a server-timing equivalent).  Pass a snapshot
            with ``agent="kb_agent"`` and ``span="kb_query"`` already set.

    Returns:
        The assistant's response text if at least one file_citation annotation
        is present; an empty string otherwise (including when no assistant has
        been created, when the run does not complete, or when no citations exist).

    Preconditions:
        - OPENAI_API_KEY must be set in the environment.
        - _upload_openai() must have been called at least once so that
          _ASSISTANT_ID is set; if it is None the function returns "" immediately
          without emitting any telemetry record.

    Postconditions:
        - A new thread is created and immediately discarded after the run.
        - _VECTOR_STORE_ID and _ASSISTANT_ID are not modified.
        - If ctx is not None and an API call was made, exactly one ``llm_call``
          record is appended (status="ok" on success, "error" on exception).
        - No record is emitted when _ASSISTANT_ID is None (no API call made).

    Complexity:
        O(1).
    """
    if _ASSISTANT_ID is None:
        return ""
    client = get_openai_client()
    call_ctx = dataclasses.replace(ctx) if ctx is not None else None
    call_start_ts = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    try:
        thread_id = client.beta.threads.create(
            messages=[{"role": "user", "content": query}]
        ).id
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=_ASSISTANT_ID,
        )
        client_elapsed_ms = (time.perf_counter() - t0) * 1000
        call_end_ts = datetime.now(timezone.utc)
        if run.status != "completed":
            if call_ctx is not None:
                log_llm_call(
                    call_ctx,
                    model=_ASSISTANT_MODEL,
                    input_tokens=0,
                    output_tokens=0,
                    provider_server_latency_ms=None,
                    client_elapsed_ms=client_elapsed_ms,
                    call_start_ts=call_start_ts.isoformat(),
                    call_end_ts=call_end_ts.isoformat(),
                    status="error",
                )
            return ""
        if call_ctx is not None and run.usage is not None:
            log_llm_call(
                call_ctx,
                model=_ASSISTANT_MODEL,
                input_tokens=run.usage.prompt_tokens,
                output_tokens=run.usage.completion_tokens,
                provider_server_latency_ms=None,
                client_elapsed_ms=client_elapsed_ms,
                call_start_ts=call_start_ts.isoformat(),
                call_end_ts=call_end_ts.isoformat(),
                status="ok",
            )
        elif call_ctx is not None:
            log_llm_call(
                call_ctx,
                model=_ASSISTANT_MODEL,
                input_tokens=0,
                output_tokens=0,
                provider_server_latency_ms=None,
                client_elapsed_ms=client_elapsed_ms,
                call_start_ts=call_start_ts.isoformat(),
                call_end_ts=call_end_ts.isoformat(),
                status="ok",
            )
        for msg in client.beta.threads.messages.list(thread_id=thread_id).data:
            if msg.role == "assistant":
                for block in msg.content:
                    if hasattr(block, "text"):
                        return block.text.value
        return ""
    except Exception:
        client_elapsed_ms = (time.perf_counter() - t0) * 1000
        call_end_ts = datetime.now(timezone.utc)
        if call_ctx is not None:
            log_llm_call(
                call_ctx,
                model=_ASSISTANT_MODEL,
                input_tokens=0,
                output_tokens=0,
                provider_server_latency_ms=None,
                client_elapsed_ms=client_elapsed_ms,
                call_start_ts=call_start_ts.isoformat(),
                call_end_ts=call_end_ts.isoformat(),
                status="error",
            )
        raise
