"""Real integration tests for src.multi.kb_agent — all tests use live API calls."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

import src.multi.kb_agent as kb_agent
import src.multi.file_store as file_store
from src.multi.knowledge_base import clear, index, store_size

INVENTED_FACT = (
    "The Glorvak engine achieves 99.3% efficiency by cycling chromium ions "
    "through a hexagonal lattice at 4.7 milliseconds per rotation."
)
INVENTED_QUERY = "How efficient is the Glorvak engine and what is its cycle time?"


def _write_tmp(content: str) -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    f.write(content)
    f.flush()
    f.close()
    return f.name


# ── OpenAI ────────────────────────────────────────────────────────────────────

def test_ask_openai_found_in_store():
    file_store.clear_openai()
    tmp = _write_tmp(INVENTED_FACT)
    file_store.upload_files([tmp])
    os.unlink(tmp)
    result = kb_agent.ask(INVENTED_QUERY)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "99.3" in result or "Glorvak" in result


def test_ask_openai_web_search_fallback():
    file_store.clear_openai()
    result = kb_agent.ask("What is the capital city of France?")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "Paris" in result


def test_ask_openai_kb_path_not_web_path():
    file_store.clear_openai()
    tmp = _write_tmp(INVENTED_FACT)
    file_store.upload_files([tmp])
    os.unlink(tmp)
    kb_result = kb_agent.ask(INVENTED_QUERY)
    file_store.clear_openai()
    web_result = kb_agent.ask(INVENTED_QUERY)
    assert "99.3" in kb_result or "Glorvak" in kb_result
    no_info_phrases = [
        "couldn't find", "could not find", "don't have", "do not have",
        "no information", "unable to find", "not found", "can't find",
        "don't have any", "do not have any", "no results", "couldn't locate",
        "i couldn't", "i could not", "i don't have", "i do not have",
        "i'm unable", "i am unable", "no specific information", "no data",
        "cannot find", "could not locate", "does not appear", "not available",
        "didn't find", "did not find", "nothing found", "no mention", "cannot locate",
        "no relevant", "no matching", "search did not", "couldn't identify",
    ]
    assert any(phrase in web_result.lower() for phrase in no_info_phrases)


# ── Anthropic ─────────────────────────────────────────────────────────────────

def test_ask_anthropic_found_in_store(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    clear()
    index([INVENTED_FACT])
    assert store_size() > 0
    result = kb_agent.ask(INVENTED_QUERY)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "99.3" in result or "Glorvak" in result


def test_ask_anthropic_web_search_fallback(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    clear()
    assert store_size() == 0
    result = kb_agent.ask("What is the capital city of France?")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "Paris" in result


def test_ask_anthropic_kb_used_before_web(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    clear()
    index([INVENTED_FACT])
    assert store_size() > 0
    result = kb_agent.ask(INVENTED_QUERY)
    assert "99.3" in result or "Glorvak" in result
