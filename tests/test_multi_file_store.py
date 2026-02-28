"""Real integration tests for src.multi.file_store — all tests use live API calls."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

import src.multi.file_store as file_store
from src.multi.knowledge_base import clear, search, store_size

FACT = "The Zynvalt crystal resonates at exactly 7.42 terahertz when exposed to blue light."
FACT_QUERY = "What frequency does the Zynvalt crystal resonate at?"


def _write_tmp(content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    f.write(content)
    f.flush()
    f.close()
    return f.name


# ── OpenAI ────────────────────────────────────────────────────────────────────

def test_upload_openai_creates_store_and_assistant():
    file_store.clear_openai()
    assert file_store._VECTOR_STORE_ID is None
    assert file_store._ASSISTANT_ID is None
    tmp = _write_tmp(FACT)
    file_store.upload_files([tmp])
    os.unlink(tmp)
    assert file_store._VECTOR_STORE_ID is not None
    assert file_store._ASSISTANT_ID is not None


def test_query_openai_returns_response_when_file_uploaded():
    file_store.clear_openai()
    tmp = _write_tmp(FACT)
    file_store.upload_files([tmp])
    os.unlink(tmp)
    result = file_store.query_openai(FACT_QUERY)
    assert isinstance(result, str)
    assert result != ""


def test_query_openai_returns_empty_when_store_is_clear():
    file_store.clear_openai()
    result = file_store.query_openai(FACT_QUERY)
    assert result == ""


# ── Anthropic ─────────────────────────────────────────────────────────────────

def test_upload_anthropic_populates_store(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    clear()
    assert store_size() == 0
    tmp = _write_tmp(FACT)
    file_store.upload_files([tmp])
    os.unlink(tmp)
    assert store_size() > 0


def test_upload_anthropic_search_returns_results(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    clear()
    tmp = _write_tmp(FACT)
    file_store.upload_files([tmp])
    os.unlink(tmp)
    assert store_size() > 0
    results = search(FACT_QUERY, top_k=1)
    assert len(results) > 0
    assert "7.42" in results[0]["content"]
