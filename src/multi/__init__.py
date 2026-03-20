"""Public API for the multi-provider knowledge-base and KB agent package.

This package exposes four subsystems:

- knowledge_base: in-memory vector store for document ingestion and retrieval.
- wrapper: augmented LLM completion using knowledge-base results.
- file_store: provider-aware file upload (OpenAI vector store or in-memory KB).
- kb_agent: orchestrates KB-first lookup with automatic web-search fallback.

Exported names
--------------
index, search, clear, store_size  -- knowledge_base public API
complete_with_knowledge            -- wrapper public API
upload_files, clear_openai         -- file_store public API
ask                                -- kb_agent public API

This package intentionally avoids eager imports at module load time.

Some submodules pull in heavyweight optional dependencies (e.g. NumPy for the
KB vector store). Keeping imports lazy ensures that chemistry-only entrypoints
can run even when KB dependencies are unavailable/broken.
"""

from __future__ import annotations

from typing import Any, Callable, Dict


def index(*args: Any, **kwargs: Any) -> Any:
    from .knowledge_base import index as _index

    return _index(*args, **kwargs)


def search(*args: Any, **kwargs: Any) -> Any:
    from .knowledge_base import search as _search

    return _search(*args, **kwargs)


def clear(*args: Any, **kwargs: Any) -> Any:
    from .knowledge_base import clear as _clear

    return _clear(*args, **kwargs)


def store_size(*args: Any, **kwargs: Any) -> Any:
    from .knowledge_base import store_size as _store_size

    return _store_size(*args, **kwargs)


def complete_with_knowledge(*args: Any, **kwargs: Any) -> Any:
    from .wrapper import complete_with_knowledge as _complete_with_knowledge

    return _complete_with_knowledge(*args, **kwargs)


def upload_files(*args: Any, **kwargs: Any) -> Any:
    from .file_store import upload_files as _upload_files

    return _upload_files(*args, **kwargs)


def clear_openai(*args: Any, **kwargs: Any) -> Any:
    from .file_store import clear_openai as _clear_openai

    return _clear_openai(*args, **kwargs)


def ask(*args: Any, **kwargs: Any) -> Any:
    from .kb_agent import ask as _ask

    return _ask(*args, **kwargs)


__all__ = [
    "index",
    "search",
    "clear",
    "store_size",
    "complete_with_knowledge",
    "upload_files",
    "clear_openai",
    "ask",
]
