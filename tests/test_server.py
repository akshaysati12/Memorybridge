"""Tests for the MCP server layer (tool wiring, not just the store).

These exercise ``build_server``'s tool functions in-process. A separate
end-to-end stdio test in ``test_e2e.py`` proves the real MCP handshake.
"""

from __future__ import annotations

import json

import pytest

from memorybridge.server import build_server
from memorybridge.store import MemoryStore


def _tool(server, name):
    """Fetch a registered tool function by name from the MCPServer."""
    tools = server._tool_manager.list_tools()
    for tool in tools:
        if tool.name == name:
            return tool
    raise AssertionError(f"tool {name!r} not registered")


@pytest.fixture()
def tools(tmp_path):
    store = MemoryStore(str(tmp_path / "srv.db"), owner="alice")
    server = build_server(store)
    yield {t.name: t.fn for t in server._tool_manager.list_tools()}
    store.close()


class TestServerTools:
    def test_all_required_tools_registered(self, tools):
        expected = {
            "add_memory",
            "search_memories",
            "list_memories",
            "update_memory",
            "delete_memory",
        }
        assert expected.issubset(tools.keys())

    def test_tool_descriptions_are_meaningful(self, tools):
        # The model reads descriptions to decide when to use a tool,
        # so each must be a real sentence, not a stub.
        for name, fn in tools.items():
            doc = (fn.__doc__ or "").strip()
            assert len(doc) > 20, f"{name} has an empty/useless docstring"

    def test_add_then_search_round_trip(self, tools):
        added = tools["add_memory"]("I prefer Python examples", "preference", "test")
        assert added["status"] == "created"
        found = tools["search_memories"]("what language does he like?")
        assert any(r["id"] == added["memory"]["id"] for r in found["results"])

    def test_add_duplicate_via_server(self, tools):
        tools["add_memory"]("I prefer Python examples")
        second = tools["add_memory"]("I prefer Python examples")
        assert second["status"] == "duplicate"

    def test_update_and_delete_via_server(self, tools):
        added = tools["add_memory"]("I can study one hour daily")["memory"]
        updated = tools["update_memory"](added["id"], content="I can study two hours daily")
        assert updated["status"] == "updated"
        deleted = tools["delete_memory"](added["id"])
        assert deleted["status"] == "deleted"
        assert tools["delete_memory"](added["id"])["status"] == "not_found"

    def test_bad_input_returns_error_not_exception(self, tools):
        assert tools["add_memory"]("")["status"] == "error"
        assert tools["search_memories"]("")["status"] == "error"
        assert tools["update_memory"]("nope", content="x")["status"] == "not_found"

    def test_export_import_round_trip(self, tmp_path, tools):
        tools["add_memory"]("I prefer Python examples", "preference")
        tools["add_memory"]("I am learning machine learning", "goal")
        exported = tools["export_memories"]()
        payload = json.loads(exported["json"])
        assert payload["count"] == 2

        # Import into a fresh store for a different user.
        other = MemoryStore(str(tmp_path / "srv.db"), owner="bob")
        try:
            other_tools = {t.name: t.fn for t in build_server(other)._tool_manager.list_tools()}
            result = other_tools["import_memories"](exported["json"])
            assert result["imported"] == 2
            assert other.search("machine learning")
        finally:
            other.close()

    def test_import_rejects_garbage(self, tools):
        assert tools["import_memories"]("not json")["status"] == "error"
        assert tools["import_memories"]('{"wrong": 1}')["status"] == "error"
