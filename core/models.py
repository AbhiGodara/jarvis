"""
core/models.py — Shared dataclasses and enums for JARVIS.

These are the canonical data structures passed between agents,
tools, the MCP layer, and the evaluation system.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolType(str, Enum):
    LOCAL = "local"          # Built-in Python command plugins
    MCP   = "mcp"            # External MCP server tool
    LLM   = "llm"            # Pure LLM answer (no tool)
    RAG   = "rag"            # Retrieval-augmented answer


@dataclass
class ToolCall:
    """A single tool invocation decided by the Planner."""
    tool_type: ToolType
    tool_name: str                          # e.g. "get_weather", "filesystem.read_file"
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    error: str | None = None
    latency_ms: float = 0.0


@dataclass
class AgentStep:
    """One reasoning step produced by the Planner Agent."""
    thought: str                            # Chain-of-thought reasoning
    tool_call: ToolCall | None = None
    final_answer: str | None = None


@dataclass
class AgentResult:
    """The complete output of one agent run (one user command).

    stage_timings: per-stage latency breakdown in milliseconds (Mk-III
    Phase 0). Keys, recorded wherever the stage occurs:
      vad_endpoint    end of speech → recording stopped (stt.py)
      stt             transcription time (stt.py)
      stt_engine      which engine's transcript won: "openai" | "local"
                      (a string; Phase 2)
      route           run() entry → the handling tier starts (planner)
      route_tier      which tier answered: "keyword" | "semantic" | "rag" |
                      "direct_llm" | "tool_loop" (a string; Phase 3)
      semantic_score  best example-similarity score for the query (Phase 3;
                      present whenever the semantic router ran)
      llm_first_token first streamed token (Phase 1+; absent before streaming)
      llm_total       cumulative LLM time: plan + direct + synthesis (planner)
      tool_exec       tool/handler execution, incl. RAG query (planner)
      synthesis       the synthesis LLM call alone, also in llm_total (planner)
      tts_first_byte  speak() start → first PCM block written (tts.py)
      tts_total       full speech duration (tts.py)
      ttfa            end of speech → first PCM block written (main.py)
    """
    query: str
    response: str
    steps: list[AgentStep] = field(default_factory=list)
    tool_type_used: ToolType = ToolType.LLM
    tool_name_used: str = "llm"
    total_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    success: bool = True
    error: str | None = None
    timestamp: float = field(default_factory=time.time)
    stage_timings: dict[str, float | str] = field(default_factory=dict)
    # True when the answer was sentence-streamed to the SpeechQueue during the
    # run (Mk-III Phase 1) — main.py then waits for the queue instead of
    # speaking response itself.
    streamed: bool = False


@dataclass
class Document:
    """A chunk of text from an ingested document."""
    doc_id: str
    source: str                             # filename / URL
    chunk_index: int
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result from the RAG retrieval step."""
    query: str
    chunks: list[Document]
    scores: list[float]
