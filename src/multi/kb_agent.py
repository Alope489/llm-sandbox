"""Knowledge-base agent: query KB first, fall back to web search if empty."""
import os
from dotenv import load_dotenv

load_dotenv()


def ask(query: str) -> str:
    if os.environ.get("LLM_PROVIDER", "openai") == "anthropic":
        return _ask_anthropic(query)
    return _ask_openai(query)


def _ask_openai(query: str) -> str:
    from .file_store import query_openai
    result = query_openai(query)
    if result:
        return result
    return _web_search_openai(query)


def _ask_anthropic(query: str) -> str:
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
    from openai import OpenAI
    return OpenAI().responses.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        tools=[{"type": "web_search_preview"}],
        input=query,
    ).output_text


def _web_search_anthropic(query: str) -> str:
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
