"""Shared tool registry for all LLM pipeline agents.

This module maintains a central dictionary of callable tools and their
provider-specific schemas. Any agent can call get_openai_schemas() or
get_anthropic_schemas() to retrieve the full tool list for a provider,
and call() to execute a tool by name.

Adding a new tool requires a single register() call — no changes to any
agent or wrapper code.

Public API
----------
register(name, fn, openai_schema, anthropic_schema)
    Register a callable tool with both provider schemas.
get_openai_schemas() -> list[dict]
    Return all registered OpenAI function-calling schemas.
get_anthropic_schemas() -> list[dict]
    Return all registered Anthropic tool-use schemas.
get_entries() -> list[dict]
    Return all registry entries as dicts with keys:
    name, fn, openai_schema, anthropic_schema.
call(name, kwargs) -> dict
    Execute a registered tool by name and return its result dict.
"""

from __future__ import annotations

_REGISTRY: dict[str, dict] = {}


def register(
    name: str,
    fn,
    openai_schema: dict,
    anthropic_schema: dict,
) -> None:
    """Register a tool in the shared registry.

    Args:
        name: Unique tool name matching the function_name in schemas.
        fn: Callable that accepts keyword arguments and returns a dict.
        openai_schema: OpenAI function-calling schema dict
            (type: "function" wrapper).
        anthropic_schema: Anthropic tool-use schema dict
            (with input_schema key).

    Preconditions:
        - name must not already be registered (silently overwrites if it is).
    Postconditions:
        - Tool is available via get_entries(), get_openai_schemas(),
          get_anthropic_schemas(), and call().
    """
    _REGISTRY[name] = {
        "name": name,
        "fn": fn,
        "openai_schema": openai_schema,
        "anthropic_schema": anthropic_schema,
    }


def get_openai_schemas() -> list[dict]:
    """Return a list of all registered OpenAI function-calling schemas.

    Returns:
        List of dicts each with shape {"type": "function", "function": {...}}.
        Empty list if no tools are registered.
    """
    return [entry["openai_schema"] for entry in _REGISTRY.values()]


def get_anthropic_schemas() -> list[dict]:
    """Return a list of all registered Anthropic tool-use schemas.

    Returns:
        List of dicts each with shape {"name": ..., "input_schema": {...}}.
        Empty list if no tools are registered.
    """
    return [entry["anthropic_schema"] for entry in _REGISTRY.values()]


def get_entries() -> list[dict]:
    """Return all registry entries.

    Returns:
        List of dicts with keys: name, fn, openai_schema, anthropic_schema.
    """
    return list(_REGISTRY.values())


def call(name: str, kwargs: dict) -> dict:
    """Execute a registered tool by name.

    Args:
        name: The registered tool name.
        kwargs: Keyword arguments forwarded to the tool function.

    Returns:
        The dict returned by the tool function.

    Raises:
        KeyError: If name is not found in the registry.

    Preconditions:
        - name must exist in the registry.
    Postconditions:
        - Returns whatever the underlying tool function returns; never
          suppresses exceptions raised by the tool itself.
    """
    return _REGISTRY[name]["fn"](**kwargs)


# ---------------------------------------------------------------------------
# Built-in tool registrations
# ---------------------------------------------------------------------------

from src.tools.elastic_constants_lammps.host_wrapper import (  # noqa: E402
    ANTHROPIC_TOOL_SCHEMA,
    OPENAI_TOOL_SCHEMA,
    compute_elastic_constants_tool,
)

register(
    "compute_elastic_constants_tool",
    compute_elastic_constants_tool,
    OPENAI_TOOL_SCHEMA,
    ANTHROPIC_TOOL_SCHEMA,
)
