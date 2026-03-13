"""Unit tests for src/tool_registry.py."""
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dummy_schemas(name: str):
    openai = {"type": "function", "function": {"name": name, "description": "test", "parameters": {"type": "object", "properties": {}}}}
    anthropic = {"name": name, "description": "test", "input_schema": {"type": "object", "properties": {}}}
    return openai, anthropic


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_register_and_get_entries():
    """Registering a tool makes it appear in get_entries()."""
    import src.tool_registry as reg

    dummy_fn = lambda **kw: {"status": "ok"}
    oai, ant = _make_dummy_schemas("dummy_test_tool")
    reg.register("dummy_test_tool", dummy_fn, oai, ant)

    names = [e["name"] for e in reg.get_entries()]
    assert "dummy_test_tool" in names

    entry = next(e for e in reg.get_entries() if e["name"] == "dummy_test_tool")
    assert entry["fn"] is dummy_fn
    assert entry["openai_schema"] == oai
    assert entry["anthropic_schema"] == ant


def test_get_openai_schemas_returns_list():
    """get_openai_schemas returns a non-empty list of dicts with type='function'."""
    import src.tool_registry as reg

    schemas = reg.get_openai_schemas()
    assert isinstance(schemas, list)
    assert len(schemas) > 0
    for s in schemas:
        assert s.get("type") == "function"
        assert "function" in s


def test_get_anthropic_schemas_returns_list():
    """get_anthropic_schemas returns a non-empty list of dicts with input_schema."""
    import src.tool_registry as reg

    schemas = reg.get_anthropic_schemas()
    assert isinstance(schemas, list)
    assert len(schemas) > 0
    for s in schemas:
        assert "name" in s
        assert "input_schema" in s


def test_call_dispatches_to_fn():
    """call(name, kwargs) invokes the registered function with the given kwargs."""
    import src.tool_registry as reg

    received = {}

    def capture_fn(**kw):
        received.update(kw)
        return {"status": "ok", "value": kw.get("x")}

    oai, ant = _make_dummy_schemas("capture_test_tool")
    reg.register("capture_test_tool", capture_fn, oai, ant)

    result = reg.call("capture_test_tool", {"x": 42})
    assert received == {"x": 42}
    assert result == {"status": "ok", "value": 42}


def test_elastic_tool_registered():
    """compute_elastic_constants_tool is registered at module import time."""
    import src.tool_registry as reg

    names = [e["name"] for e in reg.get_entries()]
    assert "compute_elastic_constants_tool" in names

    entry = next(e for e in reg.get_entries() if e["name"] == "compute_elastic_constants_tool")
    assert callable(entry["fn"])
    assert entry["openai_schema"]["type"] == "function"
    assert "input_schema" in entry["anthropic_schema"]
