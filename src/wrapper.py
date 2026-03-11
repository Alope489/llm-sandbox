"""LLM wrapper: unified chat interface over OpenAI (default) or Anthropic."""
import os

from dotenv import load_dotenv

load_dotenv()


def complete(messages: list[dict]) -> str:
    provider = os.environ.get("LLM_PROVIDER", "openai")
    if provider == "anthropic":
        return _complete_anthropic(messages)
    return _complete_openai(messages)


def _complete_openai(messages: list[dict]) -> str:
    from openai import OpenAI

    return OpenAI().chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
    ).choices[0].message.content


def _complete_anthropic(messages: list[dict]) -> str:
    from anthropic import Anthropic

    system_parts = [m["content"] for m in messages if m.get("role") == "system"]
    create_kw: dict = {
        "model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        "max_tokens": int(os.environ.get("MAX_TOKENS", "1024")),
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages if m.get("role") in ("user", "assistant")],
    }
    if system_parts:
        create_kw["system"] = [{"type": "text", "text": "\n".join(system_parts)}]
    return Anthropic().messages.create(**create_kw).content[0].text