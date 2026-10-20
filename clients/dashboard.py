"""MemoryBridge dashboard — a web GUI that drives the real MCP server.

The dashboard is itself an MCP host: every button click spawns the actual
MemoryBridge MCP server as a subprocess, connects over stdio, calls one tool,
and shows the raw result. Nothing bypasses the protocol, so what you see in
the browser is exactly what Claude Desktop or Cursor would do.

Run:
    python -m clients.dashboard --user alice --port 8765
then open http://127.0.0.1:8765
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, stdio_client

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = str(PROJECT_ROOT / "data" / "memorybridge.db")
DEFAULT_PORT = 8765
HTML_PATH = Path(__file__).resolve().parent / "dashboard.html"

# Per-call budget so a wedged server cannot hang the browser forever.
CALL_TIMEOUT = 25.0


def _server_params(user: str, db: str) -> StdioServerParameters:
    """Build stdio spawn parameters for the MemoryBridge server."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "memorybridge.server", "--user", user, "--db", db],
        env=env,
        cwd=str(PROJECT_ROOT),
    )


async def _call_tool(user: str, db: str, name: str, arguments: dict) -> dict:
    """Spawn the server, call one tool, disconnect, return parsed JSON."""
    params = _server_params(user, db)
    async with asyncio.timeout(CALL_TIMEOUT):
        # Server logs go to /dev/null: stdout is the MCP pipe, and the
        # browser surfaces any real error through the tool result.
        async with stdio_client(params, errlog=subprocess.DEVNULL) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                chunks = [
                    item.text
                    for item in result.content
                    if getattr(item, "type", None) == "text"
                ]
                blob = "\n".join(chunks)
                try:
                    return json.loads(blob)
                except json.JSONDecodeError:
                    return {
                        "raw": blob,
                        "isError": bool(getattr(result, "isError", False)),
                    }


async def _list_tools(user: str, db: str) -> list[dict]:
    """Spawn the server and list the tools it advertises."""
    params = _server_params(user, db)
    async with asyncio.timeout(CALL_TIMEOUT):
        async with stdio_client(params, errlog=subprocess.DEVNULL) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                return [
                    {
                        "name": t.name,
                        "description": (t.description or "").strip(),
                    }
                    for t in tools.tools
                ]


class Bridge:
    """Marshals blocking HTTP-handler calls onto one asyncio loop thread."""

    def __init__(self, db: str) -> None:
        self.db = db
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()

    def _run(self, coro, timeout: float = CALL_TIMEOUT + 10):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def call(self, user: str, tool: str, args: dict) -> dict:
        return self._run(_call_tool(user, self.db, tool, args))

    def tools(self, user: str) -> list[dict]:
        return self._run(_list_tools(user, self.db))

    def list_users(self) -> list[str]:
        """Return every distinct owner stored in the database.

        Each user owns an isolated slice of the same SQLite file, so this is a
        direct read (no MCP round-trip needed).
        """
        import sqlite3

        conn = sqlite3.connect(self.db)
        try:
            rows = conn.execute(
                "SELECT DISTINCT owner FROM memories ORDER BY owner"
            ).fetchall()
        finally:
            conn.close()
        return [r[0] for r in rows]


class Handler(BaseHTTPRequestHandler):
    """Minimal JSON API plus a single static page."""

    bridge: Bridge  # injected at startup

    def log_message(self, *_args) -> None:  # keep the console readable
        pass

    def _send(self, code: int, body, content_type: str = "application/json") -> None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path in ("/", "/index.html"):
            if HTML_PATH.exists():
                self._send(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send(500, {"error": "dashboard.html is missing"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0) or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self._send(400, {"error": "request body must be JSON"})
            return

        user = str(payload.get("user", "")).strip()
        if self.path != "/api/users" and not user:
            self._send(400, {"error": "user is required"})
            return

        try:
            if self.path == "/api/users":
                self._send(200, {"users": self.bridge.list_users()})
            elif self.path == "/api/tools":
                self._send(200, {"tools": self.bridge.tools(user)})
            elif self.path == "/api/call":
                tool = str(payload.get("tool", "")).strip()
                args = payload.get("args") or {}
                if not tool:
                    self._send(400, {"error": "tool is required"})
                    return
                if not isinstance(args, dict):
                    self._send(400, {"error": "args must be an object"})
                    return
                self._send(200, {"result": self.bridge.call(user, tool, args)})
            else:
                self._send(404, {"error": "not found"})
        except Exception as exc:  # surface, never crash the server
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def make_server(db: str, host: str, port: int) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"bridge": Bridge(db)})
    return ThreadingHTTPServer((host, port), handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dashboard",
        description="Web GUI for the MemoryBridge MCP server.",
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("MEMORYBRIDGE_USER", "alice"),
        help="Owner identity the dashboard starts under (switchable in the UI).",
    )
    parser.add_argument(
        "--db",
        default=os.environ.get("MEMORYBRIDGE_DB", DEFAULT_DB),
        help=f"Path to the SQLite database (default: {DEFAULT_DB}).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args(argv)

    server = make_server(args.db, args.host, args.port)
    print(
        f"memorybridge dashboard: http://{args.host}:{args.port}  "
        f"(user={args.user!r}, db={args.db!r})",
        flush=True,
    )
    print("Open that URL in a browser. Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
