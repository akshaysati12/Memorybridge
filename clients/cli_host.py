"""A minimal MCP host (client) for MemoryBridge.

This CLI host plays the role of an "AI assistant" during demos and tests:
it connects to the MemoryBridge MCP server over stdio, lists the available
tools, and invokes them from the command line. Running it twice against the
same database demonstrates portability across two assistant clients.

Examples:
    # Save a memory (Assistant A)
    python -m clients.cli_host --user alice add "I am learning machine learning" --category goal

    # Reuse it from another client (Assistant B)
    python -m clients.cli_host --user alice search "what should I study?"

    # Inspect / correct / delete
    python -m clients.cli_host --user alice list
    python -m clients.cli_host --user alice update <id> --content "..."
    python -m clients.cli_host --user alice delete <id>

    # Another user sees nothing of alice's
    python -m clients.cli_host --user bob list
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

from mcp import ClientSession, StdioServerParameters, stdio_client

from memorybridge.store import DEFAULT_MIN_SCORE

DEFAULT_DB = os.path.join("data", "memorybridge.db")


def _server_params(user: str, db: str) -> StdioServerParameters:
    """Build stdio spawn parameters for the MemoryBridge server."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ)
    env["MEMORYBRIDGE_USER"] = user
    env["MEMORYBRIDGE_DB"] = db
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "memorybridge.server", "--user", user, "--db", db],
        env=env,
        cwd=project_root,
    )


async def _call_tool(user: str, db: str, name: str, arguments: dict) -> dict:
    """Connect, call one tool, disconnect, return the parsed result."""
    params = _server_params(user, db)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(name, arguments)
            # Collect text content from the tool response.
            chunks = []
            for item in result.content:
                if getattr(item, "type", None) == "text":
                    chunks.append(item.text)
            text_blob = "\n".join(chunks)
            try:
                return json.loads(text_blob)
            except json.JSONDecodeError:
                return {"raw": text_blob, "isError": bool(getattr(result, "isError", False))}


async def _list_tools(user: str, db: str) -> list[dict]:
    params = _server_params(user, db)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            return [
                {"name": t.name, "description": (t.description or "").strip()}
                for t in tools.tools
            ]


def _print(payload) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli_host",
        description="Minimal MCP host for the MemoryBridge server.",
    )
    parser.add_argument("--user", default=os.environ.get("MEMORYBRIDGE_USER", "demo"))
    parser.add_argument("--db", default=os.environ.get("MEMORYBRIDGE_DB", DEFAULT_DB))

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("tools", help="List the MCP tools the server exposes.")

    p_add = sub.add_parser("add", help="Save a new memory.")
    p_add.add_argument("content")
    p_add.add_argument("--category", default="general")
    p_add.add_argument("--source", default="cli-host")

    p_search = sub.add_parser("search", help="Search memories by a natural-language query.")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE)

    p_list = sub.add_parser("list", help="List stored memories.")
    p_list.add_argument("--category", default=None)
    p_list.add_argument("--limit", type=int, default=100)

    p_update = sub.add_parser("update", help="Update a memory by id.")
    p_update.add_argument("memory_id")
    p_update.add_argument("--content", default=None)
    p_update.add_argument("--category", default=None)

    p_delete = sub.add_parser("delete", help="Delete a memory by id.")
    p_delete.add_argument("memory_id")

    sub.add_parser("export", help="Export memories as JSON.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    async def run() -> None:
        if args.command == "tools":
            _print(await _list_tools(args.user, args.db))
        elif args.command == "add":
            _print(await _call_tool(
                args.user, args.db, "add_memory",
                {"content": args.content, "category": args.category, "source": args.source},
            ))
        elif args.command == "search":
            _print(await _call_tool(
                args.user, args.db, "search_memories",
                {"query": args.query, "top_k": args.top_k, "min_score": args.min_score},
            ))
        elif args.command == "list":
            payload: dict = {"limit": args.limit}
            if args.category:
                payload["category"] = args.category
            _print(await _call_tool(args.user, args.db, "list_memories", payload))
        elif args.command == "update":
            payload = {"memory_id": args.memory_id}
            if args.content is not None:
                payload["content"] = args.content
            if args.category is not None:
                payload["category"] = args.category
            _print(await _call_tool(args.user, args.db, "update_memory", payload))
        elif args.command == "delete":
            _print(await _call_tool(
                args.user, args.db, "delete_memory", {"memory_id": args.memory_id}
            ))
        elif args.command == "export":
            _print(await _call_tool(args.user, args.db, "export_memories", {}))

    asyncio.run(run())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
