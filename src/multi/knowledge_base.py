"""In-memory vector store for document ingestion and semantic retrieval.

Documents are split into overlapping text chunks, embedded with the OpenAI
Embeddings API, and held in a module-level list (_STORE).  Retrieval uses
cosine similarity between the query embedding and all stored chunk embeddings.

Telemetry instrumentation:
    ``_embed`` and ``index`` accept an optional ``ctx`` (``CallContext``).
    ``_embed`` emits a ``llm_call`` record with ``output_tokens=0``
    (embeddings have no completion tokens) and ``provider_server_latency_ms=None``.

Environment variables
---------------------
OPENAI_API_KEY   : Required. Used by the OpenAI client for embedding calls.
EMBEDDING_MODEL  : Optional. Embedding model name (default: text-embedding-3-small).

Dependencies:
    os, dataclasses, time, datetime, numpy, dotenv, src.llm_pipeline_telemetry.

Pillar compliance:
    - Pillar 4: No hardcoding; model from env vars.
    - Pillar 7: try/except with error-path records; error records emitted
      before re-raising.
"""
import dataclasses
import os
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from dotenv import load_dotenv

from src.llm_pipeline_telemetry import CallContext, get_openai_client, log_llm_call

load_dotenv()

_STORE = []


def clear() -> None:
    """Remove all entries from the in-memory vector store.

    Returns:
        None

    Preconditions:
        - None. Safe to call on an already-empty store.

    Postconditions:
        - _STORE is empty and store_size() returns 0.
    """
    _STORE.clear()


def store_size() -> int:
    """Return the number of chunk entries currently held in the store.

    Returns:
        The integer count of stored chunk entries.

    Preconditions:
        - None.

    Postconditions:
        - The return value is a non-negative integer equal to len(_STORE).
    """
    return len(_STORE)


def _chunk(text: str, source: str, title: str, chunk_size: int = 800, overlap: int = 100) -> list[dict]:
    """Split a document string into overlapping fixed-size text chunks.

    Args:
        text: The full document text to be split.
        source: An identifier for the origin of the text (file path or "inline").
        title: A human-readable name for the document.
        chunk_size: Maximum number of characters per chunk (default: 800).
        overlap: Number of characters shared between consecutive chunks (default: 100).

    Returns:
        A list of dicts, each containing:
        - "content" (str): The chunk text.
        - "source" (str): Passed-through source identifier.
        - "title" (str): Passed-through document title.
        - "chunk_index" (int): Zero-based position of the chunk within the document.

    Preconditions:
        - text must be a non-empty string.
        - chunk_size must be greater than overlap.

    Postconditions:
        - Every character of text appears in at least one chunk.
        - Consecutive chunks overlap by exactly overlap characters (except the last chunk).
        - The returned list is non-empty when text is non-empty.
    """
    chunks = []
    i = 0
    idx = 0
    while i < len(text):
        end = min(i + chunk_size, len(text))
        chunk_text = text[i:end]
        chunks.append({"content": chunk_text, "source": source, "title": title, "chunk_index": idx})
        i = end - overlap if end < len(text) else len(text)
        idx += 1
    return chunks


def _embed(texts: list[str], ctx: Optional[CallContext] = None) -> list[list[float]]:
    """Request embedding vectors for a list of strings from the OpenAI API.

    Args:
        texts: A non-empty list of strings to embed.
        ctx: Optional ``CallContext``.  When provided, a snapshot is created with
            ``agent="knowledge_base"`` and ``span="embed"`` and a ``llm_call``
            record is appended with ``output_tokens=0`` (embeddings have no
            completion tokens) and ``provider_server_latency_ms=None``.

    Returns:
        A list of float vectors, one per input string, in the same order as texts.
        Each vector dimensionality is determined by the embedding model.

    Preconditions:
        - OPENAI_API_KEY must be set in the environment.
        - texts must be a non-empty list of strings.

    Postconditions:
        - The returned list has the same length as texts.
        - Each element is a list of floats representing the embedding vector.
        - If ctx is not None, exactly one ``llm_call`` record is appended.

    Complexity:
        O(n) where n = len(texts).
    """
    client = get_openai_client()
    model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    call_ctx = (
        dataclasses.replace(ctx, agent="knowledge_base", span="embed")
        if ctx is not None
        else None
    )
    call_start_ts = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    try:
        response = client.embeddings.create(input=texts, model=model)
        client_elapsed_ms = (time.perf_counter() - t0) * 1000
        call_end_ts = datetime.now(timezone.utc)
        if call_ctx is not None:
            usage = getattr(response, "usage", None)
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            log_llm_call(
                call_ctx,
                model=model,
                input_tokens=input_tokens,
                output_tokens=0,
                provider_server_latency_ms=None,
                client_elapsed_ms=client_elapsed_ms,
                call_start_ts=call_start_ts.isoformat(),
                call_end_ts=call_end_ts.isoformat(),
                status="ok",
            )
        return [d.embedding for d in response.data]
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


def _embed_and_store(chunks: list[dict], ctx: Optional[CallContext] = None) -> None:
    """Embed a list of chunk dicts and append them with their vectors to _STORE.

    Args:
        chunks: A list of chunk dicts as produced by _chunk(), each containing
                at minimum a "content" key with the text to embed.
        ctx: Optional ``CallContext`` forwarded to ``_embed``.

    Returns:
        None

    Preconditions:
        - OPENAI_API_KEY must be set in the environment.
        - chunks must be a non-empty list of dicts with a "content" key.

    Postconditions:
        - Each dict in chunks is copied and augmented with a "vector" key, then
          appended to _STORE.
        - store_size() increases by len(chunks).

    Complexity:
        O(n) where n = len(chunks).
    """
    texts = [c["content"] for c in chunks]
    vectors = _embed(texts, ctx=ctx)
    for c, v in zip(chunks, vectors):
        entry = c.copy()
        entry["vector"] = v
        _STORE.append(entry)


def index(paths: list[str], ctx: Optional[CallContext] = None) -> None:
    """Ingest one or more documents into the in-memory vector store.

    Each element of paths is treated as a file path if it points to an existing
    file on disk; otherwise it is treated as raw inline text.

    Args:
        paths: A list of file path strings or raw text strings to ingest.
        ctx: Optional ``CallContext`` forwarded to ``_embed`` for each batch.

    Returns:
        None

    Preconditions:
        - OPENAI_API_KEY must be set in the environment (used by _embed).
        - paths must be a non-empty list.
        - File paths that are provided must be readable UTF-8 text files.

    Postconditions:
        - All chunks derived from each path are embedded and stored in _STORE.
        - store_size() increases by the total number of chunks generated.

    Complexity:
        O(|paths| * chunks_per_doc).
    """
    for path in paths:
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            source = path
            title = os.path.basename(path)
        else:
            text = path
            source = "inline"
            title = "Inline Text"
        chunks = _chunk(text, source, title)
        _embed_and_store(chunks, ctx=ctx)


def search(query: str, top_k: int = 5) -> list[dict]:
    """Retrieve the top-k most semantically similar chunks for a query string.

    Similarity is measured with cosine similarity between the query embedding
    and every stored chunk embedding.

    Args:
        query: The natural-language query string.
        top_k: Maximum number of results to return (default: 5).

    Returns:
        A list of up to top_k dicts, sorted by descending similarity score, each
        containing:
        - "content" (str): The chunk text.
        - "source" (str): Origin identifier of the chunk.
        - "title" (str): Document title of the chunk.
        - "score" (float): Cosine similarity score in the range [-1.0, 1.0].
        Returns an empty list if the store is empty.

    Preconditions:
        - OPENAI_API_KEY must be set in the environment.
        - query must be a non-empty string.
        - top_k must be a positive integer.

    Postconditions:
        - The returned list has at most min(top_k, store_size()) elements.
        - Results are ordered from highest to lowest cosine similarity.
        - _STORE is not modified.
    """
    if not _STORE:
        return []
    q_vec = np.array(_embed([query])[0])
    vectors = np.array([e["vector"] for e in _STORE])
    dots = np.dot(vectors, q_vec)
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(q_vec)
    scores = np.divide(dots, norms, out=np.zeros_like(dots), where=norms != 0)
    indices = np.argsort(-scores)[:top_k]
    results = []
    for idx in indices:
        e = _STORE[idx]
        results.append({
            "content": e["content"],
            "source": e["source"],
            "title": e["title"],
            "score": scores[idx]
        })
    return results