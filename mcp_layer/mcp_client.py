"""
mcp_layer/mcp_client.py — MCP (Model Context Protocol) client for JARVIS.

Supports two transport modes:
  - stdio: spawns a local MCP server process and communicates over stdin/stdout
  - http:  connects to a remote MCP server via HTTP POST

Each MCP server exposes a set of tools via the MCP protocol.
This client:
  1. Discovers available tools from each server (tools/list)
  2. Executes tool calls (tools/call)
  3. Returns structured responses

Reference: https://modelcontextprotocol.io/specification
"""
from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """Describes a single tool exposed by an MCP server."""
    name: str                               # e.g. "read_file"
    server_name: str                        # e.g. "filesystem"
    full_name: str                          # e.g. "filesystem.read_file"
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPServerConfig:
    name: str
    transport: str                          # "stdio" | "http"
    command: list[str] | None = None        # for stdio transport
    url: str | None = None                  # for http transport
    env: dict[str, str] = field(default_factory=dict)
    timeout: int = 10


class MCPStdioClient:
    """
    MCP client that communicates with a local server process via stdin/stdout.
    Uses JSON-RPC 2.0 over newline-delimited messages.
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._msg_id = 0

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def start(self) -> None:
        """Start the MCP server subprocess."""
        if not self.config.command:
            raise ValueError(f"MCP server '{self.config.name}' has no command configured.")

        import os
        env = {**os.environ, **self.config.env}

        self._process = subprocess.Popen(
            self.config.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            bufsize=1,
        )

        self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "jarvis", "version": "2.0"}
        })
        logger.info(f"MCP server '{self.config.name}' started (pid={self._process.pid})")

    def stop(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
        logger.info(f"MCP server '{self.config.name}' stopped.")

    def _send_request(self, method: str, params: dict) -> dict | None:
        """Send a JSON-RPC request and read the response."""
        if not self._process or self._process.poll() is not None:
            logger.error(f"MCP server '{self.config.name}' is not running.")
            return None

        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params
        }

        with self._lock:
            try:
                self._process.stdin.write(json.dumps(request) + "\n")
                self._process.stdin.flush()

                start = time.time()
                while time.time() - start < self.config.timeout:
                    line = self._process.stdout.readline()
                    if line:
                        return json.loads(line.strip())
                    time.sleep(0.05)

                logger.warning(f"MCP request timed out for '{method}'")
                return None

            except Exception as e:
                logger.error(f"MCP stdio communication error: {e}")
                return None

    def list_tools(self) -> list[dict]:
        response = self._send_request("tools/list", {})
        if response and "result" in response:
            return response["result"].get("tools", [])
        return []

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        response = self._send_request("tools/call", {"name": tool_name, "arguments": arguments})
        if not response:
            return f"[MCP Error] No response from server '{self.config.name}'"
        if "error" in response:
            return f"[MCP Error] {response['error'].get('message', str(response['error']))}"

        result = response.get("result", {})
        content = result.get("content", [])
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
            elif isinstance(block, str):
                texts.append(block)
        return "\n".join(texts) if texts else str(result)


class MCPHttpClient:
    """
    MCP client for HTTP-based servers.
    Sends JSON-RPC requests as POST to the server URL.
    """

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._session = requests.Session()
        self._msg_id = 0

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    def start(self) -> None:
        logger.info(f"MCP HTTP server '{self.config.name}' configured at {self.config.url}")

    def stop(self) -> None:
        self._session.close()

    def _post(self, method: str, params: dict) -> dict | None:
        payload = {"jsonrpc": "2.0", "id": self._next_id(), "method": method, "params": params}
        try:
            resp = self._session.post(
                self.config.url, json=payload, timeout=self.config.timeout,
                headers={"Content-Type": "application/json"}
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"MCP HTTP error for '{self.config.name}': {e}")
            return None

    def list_tools(self) -> list[dict]:
        response = self._post("tools/list", {})
        if response and "result" in response:
            return response["result"].get("tools", [])
        return []

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        response = self._post("tools/call", {"name": tool_name, "arguments": arguments})
        if not response:
            return f"[MCP Error] No response from '{self.config.name}'"
        if "error" in response:
            return f"[MCP Error] {response['error'].get('message', '')}"
        result = response.get("result", {})
        content = result.get("content", [])
        texts = [b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(texts) if texts else str(result)
