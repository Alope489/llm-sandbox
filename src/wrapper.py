"""LLM wrapper: unified chat interface over OpenAI (default) or Anthropic.

Provides two public functions:

complete(messages) -> str
    Simple single-turn completion with no tool support.

complete_with_tools(messages, provider) -> str
    Tool-calling loop: sends registered tool schemas to the LLM, executes
    any tool calls the LLM requests, feeds results back, and repeats until
    the LLM produces a final text response or MAX_TOOL_CALLS is reached.
"""
import json
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

MAX_TOOL_CALLS = 5


def complete(messages: list[dict]) -> str:
    """Send messages to the active LLM provider and return the reply text.

    Args:
        messages: List of message dicts with at least 'role' and 'content'.

    Returns:
        The LLM's reply as a plain string.
    """
    provider = os.environ.get("LLM_PROVIDER", "openai")
    if provider == "anthropic":
        return _complete_anthropic(messages)
    return _complete_openai(messages)


def complete_with_tools(
    messages: list[dict],
    provider: Optional[str] = None,
) -> str:
    """Run a tool-calling loop and return the LLM's final text response.

    Sends messages plus all registered tool schemas to the LLM. If the LLM
    requests a tool call, the tool is executed via tool_registry.call, the
    result is appended to the conversation, and the loop repeats. Stops when
    the LLM produces a plain text response or MAX_TOOL_CALLS is reached.

    Args:
        messages: Initial message list in the provider's chat format.
        provider: "openai" or "anthropic". Defaults to LLM_PROVIDER env var
            or "openai".

    Returns:
        The LLM's final text response as a plain string. Returns an empty
        string if no text response is produced before MAX_TOOL_CALLS.

    Preconditions:
        - src.tool_registry must be importable (tools registered at import).
        - Appropriate API key env var must be set for the chosen provider.
    Postconditions:
        - At most MAX_TOOL_CALLS tool executions occur per call.
        - The original messages list is not mutated.
    """
    import src.tool_registry as registry

    resolved = (provider or os.environ.get("LLM_PROVIDER", "openai")).lower()
    msgs = list(messages)

    if resolved == "anthropic":
        return _tool_loop_anthropic(msgs, registry)
    return _tool_loop_openai(msgs, registry)


# ---------------------------------------------------------------------------
# Internal: simple completion helpers
# ---------------------------------------------------------------------------

def _complete_openai(messages: list[dict]) -> str:
    from openai import OpenAI

    return OpenAI().chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": m["role"], "content": m["content"]} for m in messages],
    ).choices[0].message.content


def _complete_anthropic(messages: list[dict]) -> str:
    from anthropic import Anthropic

    return Anthropic().messages.create(
        model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        max_tokens=int(os.environ.get("MAX_TOKENS", "1024")),
        **({"system": [{"type": "text", "text": "\n".join(m["content"] for m in messages if m.get("role") == "system")}]} if any(m.get("role") == "system" for m in messages) else {}),
        messages=[{"role": m["role"], "content": m["content"]} for m in messages if m.get("role") in ("user", "assistant")],
    ).content[0].text


# ---------------------------------------------------------------------------
# Internal: tool-calling loops
# ---------------------------------------------------------------------------

def _tool_loop_openai(msgs: list[dict], registry) -> str:
    """OpenAI tool-calling loop. Mutates msgs in place."""
    from openai import OpenAI

    client = OpenAI()
    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    tools = registry.get_openai_schemas()
    call_count = 0

    while call_count < MAX_TOOL_CALLS:
        response = client.chat.completions.create(
            model=model,
            messages=msgs,
            tools=tools,
            tool_choice="auto",
        )
        choice = response.choices[0]
        msg = choice.message

        if not msg.tool_calls:
            return (msg.content or "").strip()

        msgs.append(msg.model_dump(exclude_unset=True))

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            kwargs = json.loads(tc.function.arguments)
            result = registry.call(tool_name, kwargs)
            msgs.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result),
            })

        call_count += 1

    return ""


def _tool_loop_anthropic(msgs: list[dict], registry) -> str:
    """Anthropic tool-calling loop. Mutates msgs in place."""
    from anthropic import Anthropic

    client = Anthropic()
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    max_tokens = int(os.environ.get("MAX_TOKENS", "1024"))
    tools = registry.get_anthropic_schemas()
    call_count = 0

    system_blocks = [
        {"type": "text", "text": m["content"]}
        for m in msgs if m.get("role") == "system"
    ]
    conversation = [m for m in msgs if m.get("role") in ("user", "assistant")]

    while call_count < MAX_TOOL_CALLS:
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "tools": tools,
            "messages": conversation,
        }
        if system_blocks:
            kwargs["system"] = system_blocks

        response = client.messages.create(**kwargs)

        text_blocks = [b for b in response.content if getattr(b, "type", None) == "text"]
        tool_blocks = [b for b in response.content if getattr(b, "type", None) == "tool_use"]

        if not tool_blocks:
            return (text_blocks[0].text if text_blocks else "").strip()

        conversation.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tb in tool_blocks:
            tool_name = tb.name
            kwargs_tool = tb.input if isinstance(tb.input, dict) else {}
            result = registry.call(tool_name, kwargs_tool)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tb.id,
                "content": json.dumps(result),
            })

        conversation.append({"role": "user", "content": tool_results})
        call_count += 1

    return ""
