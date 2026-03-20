"""Augmented LLM completion using knowledge-base retrieval results.

This module provides a single public function, complete_with_knowledge, which
injects relevant KB context into the message list before forwarding to the
underlying LLM provider via src.wrapper.complete.

Provider-specific injection strategies
---------------------------------------
OpenAI  : A system message containing "Relevant context:" followed by labelled
          chunk excerpts is prepended to the message list.
Anthropic: Each retrieved chunk is prepended as a search_result content block
           inside the first user message, conforming to Anthropic's content
           block schema.

Environment variables
---------------------
LLM_PROVIDER : "openai" (default) or "anthropic".
"""
import os
import copy

from .knowledge_base import search
from src.wrapper import complete


def complete_with_knowledge(messages: list[dict], query: str, top_k: int = 5) -> str:
    """Augment messages with KB context and return the LLM's response.

    Retrieves the top_k most relevant chunks for query from the knowledge base,
    injects them into messages in the format appropriate for the active LLM
    provider, then delegates to src.wrapper.complete.

    Args:
        messages: A list of message dicts in the provider's chat format.
                  For Anthropic, at least one dict must have role "user".
        query: The natural-language question used to retrieve context chunks.
        top_k: Maximum number of KB chunks to inject (default: 5).

    Returns:
        The LLM's response as a plain string.

    Preconditions:
        - The knowledge base must have been populated via index() before calling
          this function; if it is empty, no context is injected.
        - LLM_PROVIDER must be set to "openai" or "anthropic" (or absent, which
          defaults to "openai").
        - For Anthropic, messages must contain at least one message with
          role "user" and a valid system message elsewhere in the list.
        - OPENAI_API_KEY or ANTHROPIC_API_KEY must be set as required by the
          active provider.

    Postconditions:
        - The original messages list passed by the caller is not mutated.
        - The returned string is the text of the LLM's reply after context
          injection; it may be empty if the LLM returns no content.
    """
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