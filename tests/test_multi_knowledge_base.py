"""Unit tests for src.multi.knowledge_base (mocked)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.multi import knowledge_base

@pytest.fixture(autouse=True)
def clear_store():
    knowledge_base.clear()
    yield
    knowledge_base.clear()

def test_chunk_produces_correct_fields():
    text = "a" * 1000
    source = "test_source"
    title = "Test Title"
    chunks = knowledge_base._chunk(text, source, title, chunk_size=500, overlap=0)
    assert len(chunks) == 2
    for c in chunks:
        assert "content" in c
        assert "source" in c
        assert "title" in c
        assert "chunk_index" in c
    assert chunks[0]["content"] == "a" * 500
    assert chunks[1]["content"] == "a" * 500
    assert chunks[0]["chunk_index"] == 0
    assert chunks[1]["chunk_index"] == 1
    assert chunks[0]["source"] == source
    assert chunks[0]["title"] == title

def test_chunk_overlap_is_applied():
    text = "1234567890"  # length 10
    chunks = knowledge_base._chunk(text, "", "", chunk_size=5, overlap=2)
    assert len(chunks) == 3
    assert chunks[0]["content"] == "12345"
    assert chunks[1]["content"] == "45678"
    assert chunks[2]["content"] == "7890"
    assert chunks[1]["content"][:2] == chunks[0]["content"][-2:]
    assert chunks[2]["content"][:2] == chunks[1]["content"][-2:]

def test_search_returns_top_k_sorted_by_score(monkeypatch):
    def mock_embed(texts):
        if texts == ["query"]:
            return [[1.0, 0.0]]
        vec_map = {"text1": [1.0, 0.0], "text2": [0.5, 0.5], "text3": [0.0, 1.0]}
        return [vec_map.get(t, [0.0, 0.0]) for t in texts]
    monkeypatch.setattr(knowledge_base, "_embed", mock_embed)
    knowledge_base.index(["text1", "text2", "text3"])
    results = knowledge_base.search("query", top_k=2)
    assert len(results) == 2
    assert results[0]["content"] == "text1"
    assert results[1]["content"] == "text2"
    assert results[0]["score"] > results[1]["score"]

def test_search_result_schema(monkeypatch):
    monkeypatch.setattr(knowledge_base, "_embed", lambda texts: [[0.0] * 1536 for _ in texts])
    knowledge_base.index(["sample text"])
    results = knowledge_base.search("query")
    for r in results:
        assert set(r.keys()) == {"content", "source", "title", "score"}

def test_clear_resets_store(monkeypatch):
    monkeypatch.setattr(knowledge_base, "_embed", lambda texts: [[0.0] * 1536 for _ in texts])
    knowledge_base.index(["text"])
    assert knowledge_base.store_size() > 0
    knowledge_base.clear()
    assert knowledge_base.store_size() == 0
    assert knowledge_base.search("query") == []

@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="No OpenAI API key set; set OPENAI_API_KEY to run integration test",
)
def test_integration_index_and_search():
    knowledge_base.clear()
    assert knowledge_base.store_size() == 0
    doc1 = "Photosynthesis is the process by which plants convert sunlight into energy."
    doc2 = "Newton's laws describe the motion of objects."
    knowledge_base.index([doc1, doc2])
    assert knowledge_base.store_size() > 0
    assert knowledge_base.store_size() == 2  # one chunk each, short texts
    results = knowledge_base.search("How do plants make energy?", top_k=1)
    assert len(results) == 1
    assert "photosynthesis" in results[0]["content"].lower()
    assert results[0]["source"] == "inline"
    knowledge_base.clear()
    assert knowledge_base.store_size() == 0