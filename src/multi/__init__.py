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
"""
from .knowledge_base import index, search, clear, store_size
from .wrapper import complete_with_knowledge
from .file_store import upload_files, clear_openai
from .kb_agent import ask
