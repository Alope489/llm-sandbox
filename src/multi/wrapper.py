"""Augmented completion with knowledge base retrieval."""
import os
import copy

from .knowledge_base import search
from src.wrapper import complete

def complete_with_knowledge(messages: list[dict], query: str, top_k: int = 5) -> str:
    results = search(query, top_k)
    provider = os.environ.get("LLM_PROVIDER", "openai")
    augmented = copy.deepcopy(messages)
    if provider == "openai":
        context_lines = [f"[Source: {r['source']}] {r['content']}" for r in results]
        context = "Relevant context:\n" + "\n".join(context_lines)
        augmented.insert(0, {"role": "system", "content": context})
    else:  # anthropic
        for msg in augmented:
            if msg["role"] == "user":
                if isinstance(msg["content"], str):
                    msg["content"] = [{"type": "text", "text": msg["content"]}]
                for r in results:
                    search_block = {
                        "type": "search_result",
                        "source": r["source"],
                        "title": r["title"],
                        "content": [{"type": "text", "text": r["content"]}]
                    }
                    msg["content"].insert(0, search_block)
                break
    return complete(augmented)