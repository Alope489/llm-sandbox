import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_STORE = []

def clear() -> None:
    _STORE.clear()

def store_size() -> int:
    return len(_STORE)

def _chunk(text: str, source: str, title: str, chunk_size: int = 800, overlap: int = 100) -> list[dict]:
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

def _embed(texts: list[str]) -> list[list[float]]:
    client = OpenAI()
    model = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
    response = client.embeddings.create(input=texts, model=model)
    return [d.embedding for d in response.data]

def _embed_and_store(chunks: list[dict]) -> None:
    texts = [c["content"] for c in chunks]
    vectors = _embed(texts)
    for c, v in zip(chunks, vectors):
        entry = c.copy()
        entry["vector"] = v
        _STORE.append(entry)

def index(paths: list[str]) -> None:
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
        _embed_and_store(chunks)

def search(query: str, top_k: int = 5) -> list[dict]:
    if not _STORE:
        return []
    q_vec = np.array(_embed([query])[0])
    vectors = np.array([e["vector"] for e in _STORE])
    dots = np.dot(vectors, q_vec)
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(q_vec)
    scores = np.divide(dots, norms, where=norms != 0)
    scores[norms == 0] = 0
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