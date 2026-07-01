"""
mcp_layer/builtin_filesystem.py — Built-in Filesystem MCP server.

Implements filesystem tools natively in Python (no Node.js or external
MCP server process required). Registered into MCPManager as a virtual
"filesystem" server via manager.register_builtin().

All tool functions return spoken-English sentences directly, so the planner
can use their output without a second LLM synthesis call.  The 'spoken: True'
flag in BUILTIN_TOOLS tells the planner to skip synthesis.

Tools provided:
  filesystem.read_file          Read a text file
  filesystem.write_file         Create or replace a text file
  filesystem.append_file        Add text to the end of a file (keeps contents)
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


def read_file(path: str, max_chars: int = 4000) -> str:
    """Read the first part of a text file and return it in spoken form."""
    p = Path(path).expanduser()
    if not _is_safe_path(p):
        return f"I can't access that path, sir — it's outside the allowed directories."
    if not p.exists():
        return f"[Error] File not found: {path}"
    if not p.is_file():
        return f"[Error] {path} is not a file, sir."
    try:
        text = p.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as e:
        return f"[Error] Could not read {path}: {e}"

    if not text:
        return f"The file {p.name} is empty, sir."

    tail = ""
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0]
        tail = " The file continues beyond what I read, sir."

    return f"Here is the content of {p.name}, sir. {text}{tail}"


def write_file(path: str, content: str) -> str:
    """Create or overwrite a text file with the given content."""
    p = Path(path).expanduser()
    if not _is_safe_path(p):
        return "I can't write to that location, sir — it's outside the allowed directories."
    if p.name.lower() in _SENSITIVE_NAMES:
        return f"I won't overwrite {p.name}, sir — it's a protected file."
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Done, sir. {p.name} has been written to {p.parent.name}."
    except Exception as e:
        return f"[Error] Could not write {path}: {e}"


def append_file(path: str, content: str) -> str:
    """Append text to the end of a file, creating it if it doesn't exist.

    Unlike write_file this preserves what's already there — the right tool for
    'add a note', 'append to notes.txt', etc.
    """
    p = Path(path).expanduser()
    if not _is_safe_path(p):
        return "I can't write to that location, sir — it's outside the allowed directories."
    if p.name.lower() in _SENSITIVE_NAMES:
        return f"I won't modify {p.name}, sir — it's a protected file."
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        existing = p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""
        # Separate the new text onto its own line so entries don't run together.
        prefix = "" if (not existing or existing.endswith("\n")) else "\n"
        with p.open("a", encoding="utf-8") as f:
            f.write(prefix + content + "\n")
        return f"Done, sir. I've added that to {p.name}."
    except Exception as e:
        return f"[Error] Could not append to {path}: {e}"


def list_directory(path: str = ".") -> str:
    """List files and directories at the given path in spoken form."""
    p = Path(path).expanduser()
    if not _is_safe_path(p):
        return "That path is outside the allowed directories, sir."
    if not p.exists():
        return f"[Error] Directory not found: {path}"
    if not p.is_dir():
        return f"[Error] {path} is not a directory, sir."
    try:
        entries = sorted(p.iterdir(), key=lambda e: (e.is_file(), e.name))
    except Exception as e:
        return f"[Error] Could not list {path}: {e}"

    if not entries:
        return f"The {p.name} folder is empty, sir."

    dirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]

    parts: list[str] = []
    if dirs:
        names = ", ".join(d.name for d in dirs[:4])
        extra = f" and {len(dirs) - 4} more" if len(dirs) > 4 else ""
        parts.append(f"{len(dirs)} folder{'s' if len(dirs) > 1 else ''} ({names}{extra})")
    if files:
        names = ", ".join(f.name for f in files[:4])
        extra = f" and {len(files) - 4} more" if len(files) > 4 else ""
        parts.append(f"{len(files)} file{'s' if len(files) > 1 else ''} ({names}{extra})")

    return f"The {p.name} folder contains {', and '.join(parts)}, sir."


def search_files(pattern: str, root: str = ".") -> str:
    """Recursively find files matching a glob pattern, returning a spoken result."""
    root_path = Path(root).expanduser()
    if not _is_safe_path(root_path):
        return "That search root is outside the allowed directories, sir."

    matches: list[str] = []
    try:
        for dirpath, _, filenames in os.walk(root_path):
            for fname in filenames:
                if fnmatch.fnmatch(fname.lower(), pattern.lower()):
                    matches.append(str(Path(dirpath) / fname))
                    if len(matches) >= 20:
                        break
            if len(matches) >= 20:
                break
    except Exception as e:
        return f"[Error] Search failed: {e}"

    if not matches:
        return f"No files matching '{pattern}' were found under {root_path.name}, sir."

    n = len(matches)
    names = ", ".join(Path(m).name for m in matches[:4])
    extra = f" and {n - 4} more" if n > 4 else ""
    return f"Found {n} file{'s' if n > 1 else ''} matching '{pattern}', sir: {names}{extra}."


def get_file_info(path: str) -> str:
    """Return file metadata in spoken form."""
    p = Path(path).expanduser()
    if not _is_safe_path(p):
        return "That path is outside the allowed directories, sir."
    if not p.exists():
        return f"[Error] Path not found: {path}"

    from datetime import datetime
    stat = p.stat()
    kind = "directory" if p.is_dir() else "file"
    size = f"{stat.st_size:,} bytes" if p.is_file() else "not applicable"
    modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d at %H:%M")
    return f"{p.name} is a {kind}, {size}, last modified on {modified}, sir."


def summarize_pdf(path: str) -> str:
    """Extract text content from a PDF file (requires pypdf)."""
    try:
        import pypdf
    except ImportError:
        return "[Error] pypdf not installed. Run: pip install pypdf"

    p = Path(path).expanduser()
    if not _is_safe_path(p):
        return "That path is outside the allowed directories, sir."
    if not p.exists():
        return f"[Error] File not found: {path}"

    try:
        reader = pypdf.PdfReader(str(p))
        pages = []
        for i, page in enumerate(reader.pages[:20]):
            pages.append(f"[Page {i+1}]\n{page.extract_text()}")
        text = "\n\n".join(pages)
        if len(text) > 8000:
            text = text[:8000] + "\n... [truncated]"
        return f"Here is the extracted text from {p.name}, sir. {text}"
    except Exception as e:
        return f"[Error] Could not read PDF: {e}"


# ── Tool registry ──────────────────────────────────────────────────────────────
# 'spoken': True tells MCPManager (and then the planner) that the tool output
# is already a natural spoken sentence — no synthesis LLM call is needed.

BUILTIN_TOOLS: dict[str, Any] = {
    "read_file": {
        "fn": read_file,
        "description": "Read the text content of a file from disk.",
        "schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Absolute or relative file path"}},
            "required": ["path"]
        },
        "spoken": True,
    },
    "write_file": {
        "fn": write_file,
        "description": (
            "Create a NEW file, or REPLACE the entire contents of an existing "
            "one, with the given text. Parent directories are created "
            "automatically — never ask the user to create the folder first. "
            "Use this for 'create a file', 'make a file', or 'save this as a "
            "file'. Do NOT use this to add to a file that already has content "
            "(it erases what's there) — use append_file for 'add to' requests."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path to write"},
                "content": {"type": "string", "description": "Text content to write into the file"},
            },
            "required": ["path", "content"]
        },
        "spoken": True,
    },
    "append_file": {
        "fn": append_file,
        "description": (
            "Add text to the END of a file without erasing its current "
            "contents (creates the file if it's missing). Use this whenever the "
            "user says 'add to', 'append to', 'add a note to', or 'add "
            "something to' a file — e.g. adding a line to notes.txt. Always "
            "prefer this over write_file for 'add' requests."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to append to"},
                "content": {"type": "string", "description": "Text to add to the end of the file"},
            },
            "required": ["path", "content"]
        },
        "spoken": True,
    },
    "list_directory": {
        "fn": list_directory,
        "description": "List files and subdirectories at a given path.",
        "schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Directory path (default: current dir)"}},
            "required": []
        },
        "spoken": True,
    },
    "search_files": {
        "fn": search_files,
        "description": "Recursively search for files matching a name pattern.",
        "schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern, e.g. '*.py'"},
                "root": {"type": "string", "description": "Root directory to search from"},
            },
            "required": ["pattern"]
        },
        "spoken": True,
    },
    "get_file_info": {
        "fn": get_file_info,
        "description": "Get metadata (size, type, last modified) of a file or directory.",
        "schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        },
        "spoken": True,
    },
    "summarize_pdf": {
        "fn": summarize_pdf,
        "description": "Extract and return text content from a PDF file.",
        "schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path to the PDF file"}},
            "required": ["path"]
        },
        "spoken": True,
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
