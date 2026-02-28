"""Detailed integration tests for the multi/knowledge-base pipeline: index, search, complete_with_knowledge, file_store, kb_agent.
Tests use live API calls. Marked as integration + multi. Skip when required API key is missing."""
import os
import tempfile

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.multi,
]

# ─── Knowledge base (index, search, complete_with_knowledge) ─────────────────

@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run knowledge_base integration tests (embeddings)",
)
class TestKnowledgeBaseIntegration:
    """Index, search, and complete_with_knowledge with live embeddings and LLM."""

    def test_integration_index_search_returns_relevant_chunks(self):
        from src.multi.knowledge_base import clear, index, search, store_size

        clear()
        index([
            "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
            "Newton's three laws of motion describe the relationship between force, mass, and acceleration.",
        ])
        assert store_size() >= 2
        results = search("How do plants convert light to energy?", top_k=2)
        assert len(results) >= 1
        assert any("photosynthesis" in r["content"].lower() or "sunlight" in r["content"].lower() for r in results)
        assert all("source" in r and "content" in r for r in results)
        clear()

    def test_integration_index_search_ranking_by_relevance(self):
        from src.multi.knowledge_base import clear, index, search

        clear()
        index([
            "The capital of France is Paris.",
            "Quantum entanglement describes correlated behavior of particles.",
        ])
        results = search("What is the capital of France?", top_k=2)
        assert len(results) >= 1
        assert results[0]["content"].lower().find("paris") >= 0 or results[0]["content"].lower().find("france") >= 0
        clear()

    def test_integration_clear_isolates_sessions(self):
        from src.multi.knowledge_base import clear, index, store_size

        clear()
        assert store_size() == 0
        index(["Single sentence for isolation test."])
        assert store_size() > 0
        clear()
        assert store_size() == 0

    def test_integration_complete_with_knowledge_uses_retrieved_context(self):
        from src.multi.knowledge_base import clear, index, store_size
        from src.multi import complete_with_knowledge

        clear()
        index(["The secret code for this test is X7K9."])
        assert store_size() > 0
        messages = [{"role": "user", "content": "What is the secret code?"}]
        response = complete_with_knowledge(messages, "secret code", top_k=3)
        assert isinstance(response, str)
        assert "X7K9" in response
        clear()


# ─── File store (OpenAI: upload + query) ────────────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run file_store OpenAI integration tests",
)
class TestFileStoreOpenAIIntegration:
    """OpenAI file_store: upload_files, query_openai, clear_openai."""

    def test_integration_upload_then_query_returns_content(self):
        import src.multi.file_store as file_store

        file_store.clear_openai()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("The Trexel value is 8842.")
            f.flush()
            path = f.name
        try:
            file_store.upload_files([path])
            result = file_store.query_openai("What is the Trexel value?")
            assert isinstance(result, str)
            assert len(result) > 0
            assert "8842" in result
        finally:
            os.unlink(path)
        file_store.clear_openai()

    def test_integration_clear_openai_empties_store(self):
        import src.multi.file_store as file_store

        file_store.clear_openai()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Temporary content.")
            f.flush()
            path = f.name
        try:
            file_store.upload_files([path])
            result_before = file_store.query_openai("Temporary content")
            assert len(result_before) > 0
            file_store.clear_openai()
            result_after = file_store.query_openai("Temporary content")
            assert result_after == ""
        finally:
            os.unlink(path)
        file_store.clear_openai()


# ─── KB agent (ask: KB first, web fallback) ──────────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="Set OPENAI_API_KEY to run kb_agent OpenAI integration tests",
)
class TestKbAgentOpenAIIntegration:
    """OpenAI kb_agent.ask: KB path and web-search fallback."""

    def test_integration_ask_with_kb_returns_kb_content(self):
        import src.multi.file_store as file_store
        import src.multi.kb_agent as kb_agent

        file_store.clear_openai()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("The Zylon metric equals 3.14.")
            f.flush()
            path = f.name
        try:
            file_store.upload_files([path])
            result = kb_agent.ask("What is the Zylon metric?")
            assert isinstance(result, str)
            assert len(result) > 0
            assert "3.14" in result
        finally:
            os.unlink(path)
        file_store.clear_openai()

    def test_integration_ask_web_fallback_returns_answer(self):
        import src.multi.file_store as file_store
        import src.multi.kb_agent as kb_agent

        file_store.clear_openai()
        result = kb_agent.ask("What is the capital of Germany?")
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Berlin" in result


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="Set ANTHROPIC_API_KEY to run kb_agent Anthropic integration tests",
)
class TestKbAgentAnthropicIntegration:
    """Anthropic kb_agent.ask: in-memory KB first, web fallback."""

    def test_integration_ask_with_kb_returns_kb_content(self, monkeypatch):
        from src.multi.knowledge_base import clear, index, store_size
        import src.multi.kb_agent as kb_agent

        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        clear()
        index(["The AnthroFact number is 7777."])
        assert store_size() > 0
        result = kb_agent.ask("What is the AnthroFact number?")
        assert isinstance(result, str)
        assert "7777" in result

    def test_integration_ask_web_fallback_returns_answer(self, monkeypatch):
        from src.multi.knowledge_base import clear
        import src.multi.kb_agent as kb_agent

        monkeypatch.setenv("LLM_PROVIDER", "anthropic")
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
        clear()
        result = kb_agent.ask("What is the capital of Italy?")
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Rome" in result


# ─── Complete-with-knowledge (Anthropic path) ─────────────────────────────────

@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY") or not os.environ.get("ANTHROPIC_API_KEY"),
    reason="Set OPENAI_API_KEY (embeddings) and ANTHROPIC_API_KEY to run Anthropic complete_with_knowledge",
)
def test_integration_complete_with_knowledge_anthropic_uses_context(monkeypatch):
    from src.multi.knowledge_base import clear, index, store_size
    from src.multi import complete_with_knowledge

    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    clear()
    index(["The shared secret is Banana42."])
    assert store_size() > 0
    messages = [{"role": "user", "content": "What is the shared secret?"}]
    response = complete_with_knowledge(messages, "shared secret", top_k=3)
    assert isinstance(response, str)
    assert "Banana42" in response
    clear()
