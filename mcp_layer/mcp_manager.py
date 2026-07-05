"""
mcp_layer/mcp_manager.py — Orchestrates all MCP server connections.

Responsibilities:
  - Start/stop all configured MCP servers
  - Discover tools from each server at startup
  - Route tool calls to the correct server
  - Expose a unified tool registry for the Planner Agent
  - Handle errors and timeouts gracefully

Adding a new MCP server:
  1. Add an entry to the mcp_servers list in config.yaml
  2. That's it — the manager auto-discovers tools on startup.
"""
from __future__ import annotations

import logging
from typing import Any, Callable

from .mcp_client import MCPTool, MCPServerConfig, MCPStdioClient, MCPHttpClient

logger = logging.getLogger(__name__)

_AnyMCPClient = MCPStdioClient | MCPHttpClient


class MCPManager:
    """
    Central registry for all MCP server connections.

    Usage:
        manager = MCPManager(server_configs)
        manager.start()
        tools = manager.list_tools()
        result = manager.call("filesystem.read_file", {"path": "foo.txt"})
        manager.stop()
    """

    def __init__(self, server_configs: list[dict[str, Any]]):
        self._configs: list[MCPServerConfig] = [
            MCPServerConfig(**cfg) for cfg in server_configs
        ]
        self._clients: dict[str, _AnyMCPClient] = {}
        self._tools: dict[str, MCPTool] = {}
        # Built-in tools run in-process: full tool name → Python callable
        self._builtin_handlers: dict[str, Callable[..., str]] = {}
        # Tools flagged spoken=True already return a natural spoken sentence —
        # the planner skips the synthesis LLM call for these.
        self._spoken_tools: set[str] = set()
        # Schema-safe alias ("server__tool", what the LLM sees) → full name.
        # An explicit map instead of str.replace("__", ".") on the way back,
        # which corrupted tool names that legitimately contain '__'.
        self._aliases: dict[str, str] = {}

    def start(self) -> None:
        """Start all MCP servers and discover their tools."""
        for cfg in self._configs:
            try:
                client = self._make_client(cfg)
                client.start()
                self._clients[cfg.name] = client
                self._discover_tools(cfg.name, client)
                count = sum(1 for t in self._tools.values() if t.server_name == cfg.name)
                logger.info(f"MCP server '{cfg.name}' ready with {count} tools.")
            except Exception as e:
                logger.warning(f"Failed to start MCP server '{cfg.name}': {e}")

        logger.info(f"MCP Manager ready. Total tools: {len(self._tools)}")

    def stop(self) -> None:
        """Gracefully stop all MCP server subprocesses."""
        for name, client in self._clients.items():
            try:
                client.stop()
            except Exception as e:
                logger.warning(f"Error stopping MCP server '{name}': {e}")
        self._clients.clear()
        logger.info("MCP Manager stopped.")

    def list_tools(self) -> list[MCPTool]:
        """Return all discovered MCP tools across all servers."""
        return list(self._tools.values())

    def get_tool_schemas(self) -> list[dict]:
        """Return Gemini-compatible function declarations for all MCP tools."""
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "name": tool.full_name.replace(".", "__"),  # Gemini names can't have dots
                "description": f"[MCP:{tool.server_name}] {tool.description}",
                "parameters": tool.input_schema or {
                    "type": "object", "properties": {}, "required": []
                }
            })
        return schemas

    def _resolve(self, name: str) -> str | None:
        """Map a dot- or double-underscore-form tool name to its full name."""
        if name in self._tools:
            return name
        return self._aliases.get(name)

    def call(self, full_tool_name: str, arguments: dict[str, Any]) -> str:
        """
        Execute an MCP tool by its full name (e.g. 'filesystem.read_file').

        Handles both dot-separated and double-underscore-separated names
        (Gemini uses __ as separator in function names).
        """
        normalized = self._resolve(full_tool_name)
        if not normalized:
            return f"[MCP Error] Unknown tool: '{full_tool_name}'. Available: {list(self._tools.keys())}"

        tool = self._tools[normalized]

        logger.info(f"MCP call: {normalized}({arguments})")

        # Built-in tools execute in-process — no client/subprocess involved
        handler = self._builtin_handlers.get(normalized)
        if handler:
            try:
                return handler(**arguments)
            except TypeError as e:
                return f"[MCP Error] Bad arguments for '{normalized}': {e}"
            except Exception as e:
                logger.error(f"Built-in tool '{normalized}' failed: {e}")
                return f"[MCP Error] Tool execution failed: {e}"

        client = self._clients.get(tool.server_name)
        if not client:
            return f"[MCP Error] Server '{tool.server_name}' is not running."

        try:
            return client.call_tool(tool.name, arguments)
        except Exception as e:
            logger.error(f"MCP tool call failed: {e}")
            return f"[MCP Error] Tool execution failed: {e}"

    def register_builtin(self, server_name: str, tools_dict: dict) -> None:
        """
        Register built-in Python tools (no subprocess needed).
        Used by builtin_filesystem to inject tools without spawning a process.
        """
        for name, tool_def in tools_dict.items():
            full = f"{server_name}.{name}"
            self._tools[full] = MCPTool(
                name=name,
                server_name=server_name,
                full_name=full,
                description=tool_def.get("description", ""),
                input_schema=tool_def.get("schema", {}),
            )
            self._aliases[full.replace(".", "__")] = full
            fn = tool_def.get("fn")
            if fn is None:
                logger.warning(f"Built-in tool '{full}' has no 'fn' — it will not be callable.")
            else:
                self._builtin_handlers[full] = fn
            if tool_def.get("spoken", False):
                self._spoken_tools.add(full)

    def has_tool(self, full_tool_name: str) -> bool:
        """True if this tool name (dot or double-underscore form) is registered."""
        return self._resolve(full_tool_name) is not None

    def is_spoken_result(self, full_tool_name: str) -> bool:
        """True if this tool's output is already a spoken sentence.

        When True the planner skips the synthesis LLM call, saving ~1.5 s
        per query.  Set by register_builtin() for tools with 'spoken': True.
        """
        return self._resolve(full_tool_name) in self._spoken_tools

    def is_available(self) -> bool:
        return bool(self._clients) or bool(self._tools)

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _make_client(cfg: MCPServerConfig) -> _AnyMCPClient:
        if cfg.transport == "stdio":
            return MCPStdioClient(cfg)
        elif cfg.transport == "http":
            return MCPHttpClient(cfg)
        else:
            raise ValueError(f"Unknown MCP transport: '{cfg.transport}'")

    def _discover_tools(self, server_name: str, client: _AnyMCPClient) -> None:
        raw_tools = client.list_tools()
        for t in raw_tools:
            name = t.get("name", "")
            full = f"{server_name}.{name}"
            self._tools[full] = MCPTool(
                name=name,
                server_name=server_name,
                full_name=full,
                description=t.get("description", ""),
                input_schema=t.get("inputSchema", {}),
            )
            self._aliases[full.replace(".", "__")] = full
