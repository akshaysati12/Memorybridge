"""End-to-end test over the real MCP stdio transport.

Spawns the actual server as a subprocess and talks to it through the MCP
client, proving the wire protocol, tool registration, persistence and
isolation all work outside of in-process unit tests.
"""

from __future__ import annotations

import os
import sys

import pytest

mcp = pytest.importorskip("mcp")
from mcp import ClientSession, StdioServerParameters, stdio_client  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _params(user: str, db_path: str) -> StdioServerParameters:
    env = dict(os.environ)
    env["PYTHONPATH"] = PROJECT_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "memorybridge.server", "--user", user, "--db", db_path],
        env=env,
        cwd=PROJECT_ROOT,
    )


async def _call(session: ClientSession, name: str, arguments: dict) -> dict:
    import json

    result = await session.call_tool(name, arguments)
    texts = [item.text for item in result.content if getattr(item, "type", None) == "text"]
    return json.loads("\n".join(texts))


@pytest.mark.asyncio
async def test_full_mcp_handshake_and_lifecycle(tmp_path):
    db_path = str(tmp_path / "e2e.db")

    async with stdio_client(_params("alice", db_path)) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            assert init.server_info.name == "memorybridge"

            tools = await session.list_tools()
            names = {t.name for t in tools.tools}
            assert {"add_memory", "search_memories", "list_memories",
                    "update_memory", "delete_memory"} <= names

            added = await _call(session, "add_memory",
                                {"content": "I prefer Python examples",
                                 "category": "preference", "source": "e2e"})
            assert added["status"] == "created"
            mem_id = added["memory"]["id"]

            found = await _call(session, "search_memories",
                                {"query": "what language does he want samples in?"})
            assert any(r["id"] == mem_id for r in found["results"])

            updated = await _call(session, "update_memory",
                                  {"memory_id": mem_id,
                                   "content": "I prefer JavaScript examples"})
            assert updated["status"] == "updated"

            listed = await _call(session, "list_memories", {})
            assert listed["count"] == 1

            deleted = await _call(session, "delete_memory", {"memory_id": mem_id})
            assert deleted["status"] == "deleted"

            found2 = await _call(session, "search_memories", {"query": "JavaScript samples"})
            assert found2["results"] == []


@pytest.mark.asyncio
async def test_isolation_over_the_wire(tmp_path):
    db_path = str(tmp_path / "e2e.db")

    async with stdio_client(_params("alice", db_path)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            added = await _call(session, "add_memory",
                                {"content": "alice secret codename Nebula"})
            mem_id = added["memory"]["id"]

    # A different user's server process must not see or touch it.
    async with stdio_client(_params("bob", db_path)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await _call(session, "list_memories", {})
            assert listed["count"] == 0
            found = await _call(session, "search_memories", {"query": "Nebula codename"})
            assert found["results"] == []
            deleted = await _call(session, "delete_memory", {"memory_id": mem_id})
            assert deleted["status"] == "not_found"

    # Alice's memory is still intact after bob's attempt.
    async with stdio_client(_params("alice", db_path)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            listed = await _call(session, "list_memories", {})
            assert listed["count"] == 1


@pytest.mark.asyncio
async def test_persistence_across_server_restarts(tmp_path):
    db_path = str(tmp_path / "e2e.db")

    for _ in range(2):  # save, then restart the server
        async with stdio_client(_params("alice", db_path)) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                listed = await _call(session, "list_memories", {})
                if listed["count"] == 0:
                    await _call(session, "add_memory",
                                {"content": "I am learning machine learning"})
                else:
                    assert listed["count"] == 1
