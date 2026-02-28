"""Knowledge-base agent: query KB first, fall back to LLM-native web search if empty.

The single public entry point, ask(), dispatches to a provider-specific
implementation based on the LLM_PROVIDER environment variable.  Each
implementation follows the same two-step strategy:

1. Search the knowledge base (high-veracity data source).
2. If no results are found, issue a web search through the LLM's own native
   web-search tool — no third-party search service is used.

Provider implementations
------------------------
OpenAI   : Uses file_store.query_openai for KB lookup (OpenAI Assistants with
           file_search).  Falls back to the OpenAI Responses API with the
           web_search_preview tool.
Anthropic: Uses knowledge_base.search for KB lookup (in-memory vector store).
           Falls back to the Anthropic Messages API with the
           web_search_20250305 tool.

Environment variables
---------------------
LLM_PROVIDER    : "openai" (default) or "anthropic".
OPENAI_API_KEY  : Required for OpenAI paths.
ANTHROPIC_API_KEY: Required for Anthropic paths.
OPENAI_MODEL    : Model for OpenAI web search (default: gpt-4o-mini).
ANTHROPIC_MODEL : Model for Anthropic web search (default: claude-sonnet-4-6).
MAX_TOKENS      : Max tokens for Anthropic responses (default: 1024).
"""
import os
from dotenv import load_dotenv

load_dotenv()


def ask(query: str) -> str:
    """Answer a query using KB lookup, with automatic web-search fallback.

    Dispatches to the OpenAI or Anthropic implementation based on LLM_PROVIDER.

    Args:
        query: The natural-language question to answer.

    Returns:
        A string response from either the knowledge base or the LLM's web search.
        Returns an empty string only if both the KB and the web search return
        no usable content.

    Preconditions:
        - LLM_PROVIDER must be "openai" or "anthropic" (or absent for default).
        - The appropriate API key must be set for the active provider.

    Postconditions:
        - If the KB contains relevant results, the response is derived from those
          results without a live web request.
        - If the KB is empty or returns no results, a live web search is performed
          and its output is returned.
        - The KB store is not modified by this call.
    """
    if os.environ.get("LLM_PROVIDER", "openai") == "anthropic":
        return _ask_anthropic(query)
    return _ask_openai(query)


def _ask_openai(query: str) -> str:
    """Answer a query for the OpenAI provider: file search first, then web search.

    Args:
        query: The natural-language question to answer.

    Returns:
        The OpenAI Assistant's response text if it returns a file-cited answer;
        otherwise the output of _web_search_openai().

    Preconditions:
        - OPENAI_API_KEY must be set in the environment.
        - file_store._upload_openai() should have been called previously to
          populate the vector store; if not, query_openai returns "" and the
          function falls through to the web search.

    Postconditions:
        - Returns a non-empty string when either the KB or the web search
          produces a result.
        - The KB state in file_store is not modified.
    """
    from .file_store import query_openai
    result = query_openai(query)
    if result:
        return result
    return _web_search_openai(query)


def _ask_anthropic(query: str) -> str:
    """Answer a query for the Anthropic provider: in-memory KB first, then web search.

    Args:
        query: The natural-language question to answer.

    Returns:
        The augmented completion text from complete_with_knowledge() if the KB
        returns results; otherwise the output of _web_search_anthropic().

    Preconditions:
        - OPENAI_API_KEY must be set (used for KB embedding via knowledge_base).
        - ANTHROPIC_API_KEY must be set for both KB completion and web search.
        - knowledge_base.index() should have been called previously to populate
          the store; if the store is empty, falls through to web search.

    Postconditions:
        - Returns a non-empty string when either the KB or the web search
          produces a result.
        - The KB store in knowledge_base is not modified.
    """
    from .knowledge_base import search
    from .wrapper import complete_with_knowledge
    if search(query, top_k=5):
        return complete_with_knowledge(
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": query}],
            query,
        )
    return _web_search_anthropic(query)


def _web_search_openai(query: str) -> str:
    """Perform a web search using the OpenAI Responses API with web_search_preview.

    Args:
        query: The search query string.

    Returns:
        The full text output of the OpenAI response after web search grounding.
        Returns an empty string if the API returns no output_text.

    Preconditions:
        - OPENAI_API_KEY must be set in the environment.
        - OPENAI_MODEL must reference a model that supports the web_search_preview
          tool (default: gpt-4o-mini).

    Postconditions:
        - A live HTTP request is made to the OpenAI Responses API.
        - The KB store is not affected.
    """
    from openai import OpenAI
    return OpenAI().responses.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        tools=[{"type": "web_search_preview"}],
        input=query,
    ).output_text


def _web_search_anthropic(query: str) -> str:
    """Perform a web search using the Anthropic Messages API with web_search_20250305.

    Args:
        query: The search query string.

    Returns:
        The text content of the first TextBlock in the response that has a
        "text" attribute.  Returns an empty string if no such block exists.

    Preconditions:
        - ANTHROPIC_API_KEY must be set in the environment.
        - ANTHROPIC_MODEL must reference a model that supports the
          web_search_20250305 tool (default: claude-sonnet-4-6).
        - MAX_TOKENS must be a valid positive integer string if set (default: 1024).

    Postconditions:
        - A live HTTP request is made to the Anthropic Messages API.
        - The KB store is not affected.
    """
    from anthropic import Anthropic
    response = Anthropic().messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        max_tokens=int(os.environ.get("MAX_TOKENS", "1024")),
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": query}],
    )
    return next(
        (b.text for b in response.content if hasattr(b, "text")),
        "",
    )
