"""
mcp_layer/builtin_filesystem.py — Built-in Filesystem MCP server.

Implements filesystem tools natively in Python (no Node.js or external
MCP server process required). Registered into MCPManager as a virtual
"filesystem" server via manager.register_builtin().

Tools provided:
  filesystem.read_file          Read a text file
  filesystem.list_directory     List files in a directory
  filesystem.search_files       Find files matching a pattern
  filesystem.get_file_info      Stat a file (size, modified, type)
  filesystem.summarize_pdf      Extract text from a PDF (requires pypdf)
"""
from __future__ import annotations

import fnmatch
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Safety: restrict filesystem access to the JARVIS project directory only.
# These tools are callable by the LLM via function calling — granting the
# whole home directory would let a prompt-injected query read ssh keys,
# browser profiles, or other projects' .env files aloud.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ALLOWED_ROOTS: list[Path] = [
    _PROJECT_ROOT,
    Path.cwd(),
]


_SENSITIVE_NAMES = {".env", "token.pickle", "credentials.json"}


def _is_safe_path(path: Path) -> bool:
    """Ensure path is under an allowed root (prevent path traversal attacks)."""
    resolved = path.resolve()
    if resolved.name.lower() in _SENSITIVE_NAMES:
        return False
    return any(
        resolved == root or root in resolved.parents
        for root in _ALLOWED_ROOTS
    )


def read_file(path: str, max_chars: int = 8000) -> str:
    """Read the contents of a text file."""
    p = Path(path).expanduser()
    if not _is_safe_path(p):
        return f"[Error] Path '{path}' is outside allowed directories."
    if not p.exists():
        return f"[Error] File not found: {path}"
    if not p.is_file():
        return f"[Error] Not a file: {path}"
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... [truncated, {len(text)} total chars]"
        return text
    except Exception as e:
        return f"[Error] Could not read {path}: {e}"


def list_directory(path: str = ".") -> str:
    """List files and directories at the given path."""
    p = Path(path).expanduser()
    if not _is_safe_path(p):
        return "[Error] Path is outside allowed directories."
    if not p.exists():
        return f"[Error] Directory not found: {path}"
    if not p.is_dir():
        return f"[Error] Not a directory: {path}"
    try:
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name))
        lines = []
        for entry in entries[:100]:
            kind = "FILE" if entry.is_file() else "DIR "
            size = f"{entry.stat().st_size:>10,} B" if entry.is_file() else ""
            lines.append(f"{kind}  {entry.name:50s}  {size}")
        result = "\n".join(lines)
        if len(entries) > 100:
            result += f"\n... and {len(entries) - 100} more"
        return result or "(empty directory)"
    except Exception as e:
        return f"[Error] Could not list {path}: {e}"


def search_files(pattern: str, root: str = ".") -> str:
    """Recursively find files matching a glob pattern."""
    root_path = Path(root).expanduser()
    if not _is_safe_path(root_path):
        return "[Error] Search root is outside allowed directories."

    matches: list[str] = []
    try:
        for dirpath, _, filenames in os.walk(root_path):
            for fname in filenames:
                if fnmatch.fnmatch(fname.lower(), pattern.lower()):
                    matches.append(str(Path(dirpath) / fname))
                    if len(matches) >= 50:
                        break
            if len(matches) >= 50:
                break
    except Exception as e:
        return f"[Error] Search failed: {e}"

    if not matches:
        return f"No files matching '{pattern}' found under '{root}'."
    return "\n".join(matches)


def get_file_info(path: str) -> str:
    """Return metadata about a file: size, last modified, type."""
    p = Path(path).expanduser()
    if not _is_safe_path(p):
        return "[Error] Path is outside allowed directories."
    if not p.exists():
        return f"[Error] Path not found: {path}"

    from datetime import datetime
    stat = p.stat()
    kind = "directory" if p.is_dir() else "file"
    size = f"{stat.st_size:,} bytes" if p.is_file() else "N/A"
    modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return f"Path: {p.resolve()}\nType: {kind}\nSize: {size}\nModified: {modified}"


def summarize_pdf(path: str) -> str:
    """Extract text content from a PDF file (requires pypdf)."""
    try:
        import pypdf
    except ImportError:
        return "[Error] pypdf not installed. Run: pip install pypdf"

    p = Path(path).expanduser()
    if not _is_safe_path(p):
        return "[Error] Path is outside allowed directories."
    if not p.exists():
        return f"[Error] File not found: {path}"

    try:
        reader = pypdf.PdfReader(str(p))
        pages = []
        for i, page in enumerate(reader.pages[:20]):
            pages.append(f"[Page {i+1}]\n{page.extract_text()}")
        text = "\n\n".join(pages)
        if len(text) > 10000:
            text = text[:10000] + "\n... [truncated]"
        return text
    except Exception as e:
        return f"[Error] Could not read PDF: {e}"


# ── Tool registry ─────────────────────────────────────────────────────────────
# This dict is passed to MCPManager.register_builtin("filesystem", BUILTIN_TOOLS)

BUILTIN_TOOLS: dict[str, Any] = {
    "read_file": {
        "fn": read_file,
        "description": "Read the text content of a file from disk.",
        "schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Absolute or relative file path"}},
            "required": ["path"]
        }
    },
    "list_directory": {
        "fn": list_directory,
        "description": "List files and subdirectories at a given path.",
        "schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path (default: current dir)"}},
            "required": []
        }
    },
    "search_files": {
        "fn": search_files,
        "description": "Recursively search for files matching a name pattern.",
        "schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern, e.g. '*.py'"},
                "root": {"type": "string", "description": "Root directory to search from"}
            },
            "required": ["pattern"]
        }
    },
    "get_file_info": {
        "fn": get_file_info,
        "description": "Get metadata (size, type, last modified) of a file or directory.",
        "schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        }
    },
    "summarize_pdf": {
        "fn": summarize_pdf,
        "description": "Extract and return text content from a PDF file.",
        "schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to the PDF file"}},
            "required": ["path"]
        }
    },
}


def call_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """Dispatch a call to a built-in filesystem tool."""
    entry = BUILTIN_TOOLS.get(tool_name)
    if not entry:
        return f"[Error] Unknown filesystem tool: '{tool_name}'"
    try:
        return entry["fn"](**arguments)
    except TypeError as e:
        return f"[Error] Bad arguments for '{tool_name}': {e}"
    except Exception as e:
        logger.error(f"Filesystem tool '{tool_name}' failed: {e}")
        return f"[Error] {e}"
