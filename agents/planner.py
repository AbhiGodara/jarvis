"""
agents/planner.py — Planner Agent for JARVIS.

This is the core architectural upgrade from v1.
Instead of a simple keyword router (brain.py), the Planner uses Gemini's
native function calling to decide *how* to handle each user query:

  Option A: Answer directly (pure LLM, no tools)
  Option B: Call a local command plugin (fast, deterministic)
  Option C: Call an MCP tool (file read, GitHub, browser, etc.)
  Option D: Query the RAG engine (document-grounded answer)

Decision flow:
  1. Fast keyword pre-check (preserves low-latency for obvious commands)
  2. If keyword match → execute local command immediately
  3. If RAG trigger words present → route to RAG engine
  4. Otherwise → ask LLM to pick the best tool (function calling)
  5. Execute the selected tool
  6. If tool result found → synthesize a natural spoken answer via LLM
  7. If tool failed → retry or fall back to pure LLM

This implements a single-step ReAct-style loop with up to max_steps retries.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from core.models import AgentResult, AgentStep, ToolCall, ToolType
import commands.registry as cmd_registry
from agents.memory_manager import _default_manager as memory

if TYPE_CHECKING:
    from llm import LLMClient
    from mcp_layer.mcp_manager import MCPManager
    from retrieval.rag_engine import RAGEngine

logger = logging.getLogger(__name__)

# Phrases that strongly indicate a document-retrieval query
_RAG_TRIGGERS = frozenset([
    "in the document", "in the pdf", "according to", "what does it say",
    "from the file", "search documents", "find in docs", "what did they say",
    "interview", "transcript", "report says", "document says",
    "based on the", "from the report", "summarize the", "read the file",
])

# Queries starting with these words almost certainly need just the LLM —
# no tool lookup required. Skipping function calling saves ~800ms per query.
_DIRECT_LLM_STARTERS = (
    "who is ", "who was ", "who are ", "who were ",
    "what is ", "what was ", "what are ", "what were ",
    "why is ", "why was ", "why are ", "why does ",
    "how is ", "how does ", "how do ", "how did ",
    "explain ", "tell me about ", "describe ",
    "what do you think ", "give me ", "can you ",
    "difference between ", "compare ", "define ",
    "write ", "compose ", "generate ",
)

# Tool-required patterns — these SHOULD go through function calling
_TOOL_REQUIRED_PATTERNS = (
    "weather", "temperature", "forecast", "stocks", "share price",
    "calendar", "schedule", "remind me", "open ", "launch ", "play ",
    "send email", "translate", "screenshot", "search files", "read file",
    "news", "headlines", "time in", "convert ",
)

# History caps per call type — planning doesn't need 40 messages of context
# (runtime logs showed ~1,475 input tokens per plan call).
_DIRECT_HISTORY_MSGS = 12
_PLAN_HISTORY_MSGS = 8

# Tool results larger than this are truncated before synthesis — an 8KB file
# read would otherwise add ~2,000 tokens to the second LLM call.
_SYNTHESIS_RESULT_CHARS = 1200


class PlannerAgent:
    """
    The central intelligence of JARVIS.

    Replaces the old brain.py keyword router with an LLM-driven
    function-calling planner that can use tools, RAG, and MCP servers.
    """

    def __init__(
        self,
        llm: "LLMClient",
        mcp_manager: "MCPManager | None" = None,
        rag_engine: "RAGEngine | None" = None,
        max_steps: int = 3,
    ):
        self.llm = llm
        self.mcp = mcp_manager
        self.rag = rag_engine
        self.max_steps = max_steps
        # Tool schemas are static after startup — build once, not per query
        self._tool_schemas_cache: list[dict] | None = None
        logger.info("PlannerAgent initialized.")

    def run(self, query: str) -> AgentResult:
        """
        Process a user query end-to-end.

        Args:
            query: Lowercased, stripped user command

        Returns:
            AgentResult containing the response and execution metadata
        """
        t_start = time.time()
        steps: list[AgentStep] = []

        if not query or not query.strip():
            return AgentResult(
                query=query,
                response="I'm listening. What can I do for you?",
                tool_type_used=ToolType.LLM,
                tool_name_used="llm",
            )

        # ── Step 1: Fast keyword pre-check (no LLM cost) ──────────────────
        local_response = cmd_registry.match(query)
        if local_response is not None:
            t_ms = (time.time() - t_start) * 1000
            steps.append(AgentStep(
                thought="Keyword match found — using local command.",
                tool_call=ToolCall(
                    tool_type=ToolType.LOCAL,
                    tool_name="keyword_match",
                    result=local_response,
                    latency_ms=t_ms
                ),
                final_answer=local_response
            ))
            memory.add_turn("user", query)
            memory.add_turn("assistant", local_response)
            return AgentResult(
                query=query,
                response=local_response,
                steps=steps,
                tool_type_used=ToolType.LOCAL,
                tool_name_used="keyword_match",
                total_latency_ms=t_ms,
                success=True,
            )

        # ── Step 2: RAG check ──────────────────────────────────────────────
        if self.rag and self.rag.store.ready and self._is_rag_query(query):
            logger.info("Routing to RAG engine.")
            try:
                answer, retrieval = self.rag.query(query)
                t_ms = (time.time() - t_start) * 1000
                steps.append(AgentStep(
                    thought=f"RAG retrieval: {len(retrieval.chunks)} chunks found.",
                    tool_call=ToolCall(
                        tool_type=ToolType.RAG,
                        tool_name="rag_query",
                        arguments={"query": query},
                        result=answer,
                        latency_ms=t_ms,
                    ),
                    final_answer=answer
                ))
                memory.add_turn("user", query)
                memory.add_turn("assistant", answer)
                return AgentResult(
                    query=query,
                    response=answer,
                    steps=steps,
                    tool_type_used=ToolType.RAG,
                    tool_name_used="rag_query",
                    total_latency_ms=t_ms,
                    success=True,
                )
            except Exception as e:
                logger.warning(f"RAG failed: {e}. Falling through to LLM.")

        # ── Step 3: LLM function-calling planner ──────────────────────────
        # Facts only: ask() already sends conversation history, so including
        # recent turns here would duplicate them in every prompt.
        memory_context = memory.get_context(include_turns=False)

        # Fast-path: if query looks like a general knowledge question and doesn't
        # need a specific tool, call LLM directly (skips one full LLM round trip)
        if self._is_direct_llm_query(query):
            logger.info("Direct LLM path (no tool schema overhead)")
            answer = self.llm.ask(
                query,
                memory_context=memory_context,
                max_history_messages=_DIRECT_HISTORY_MSGS,
            )
            t_ms = (time.time() - t_start) * 1000
            logger.info(f"Stage timing: direct_llm={t_ms:.0f}ms")
            steps.append(AgentStep(thought="Direct LLM answer (fast path).", final_answer=answer))
            memory.add_turn("user", query)
            memory.add_turn("assistant", answer)
            return AgentResult(
                query=query, response=answer, steps=steps,
                tool_type_used=ToolType.LLM, tool_name_used="llm_direct",
                total_latency_ms=t_ms,
                input_tokens=self.llm.last_input_tokens,
                output_tokens=self.llm.last_output_tokens,
                success=self.llm.last_error is None,
                error=self.llm.last_error,
            )

        all_tool_schemas = self._build_tool_schemas()
        current_query = query
        for step_num in range(self.max_steps):
            logger.debug(f"Planner step {step_num + 1}/{self.max_steps}")

            t_plan = time.time()
            plan = self.llm.plan(
                current_query,
                tool_schemas=all_tool_schemas,
                max_history_messages=_PLAN_HISTORY_MSGS,
            )
            plan_ms = (time.time() - t_plan) * 1000

            # Direct LLM answer (no tool needed)
            if isinstance(plan, str):
                t_ms = (time.time() - t_start) * 1000
                steps.append(AgentStep(
                    thought="LLM answered directly without tool use.",
                    final_answer=plan
                ))
                memory.add_turn("user", query)
                memory.add_turn("assistant", plan)
                return AgentResult(
                    query=query,
                    response=plan,
                    steps=steps,
                    tool_type_used=ToolType.LLM,
                    tool_name_used="llm",
                    total_latency_ms=t_ms,
                    input_tokens=self.llm.last_input_tokens,
                    output_tokens=self.llm.last_output_tokens,
                    success=self.llm.last_error is None,
                    error=self.llm.last_error,
                )

            # Tool call decision
            tool_name = plan.get("tool", "")
            tool_args = plan.get("args", {})

            t_tool = time.time()
            tool_result, tool_type = self._execute_tool(tool_name, tool_args, current_query)
            tool_latency_ms = (time.time() - t_tool) * 1000

            steps.append(AgentStep(
                thought=f"LLM chose tool '{tool_name}' with args {tool_args}.",
                tool_call=ToolCall(
                    tool_type=tool_type,
                    tool_name=tool_name,
                    arguments=tool_args,
                    result=tool_result,
                    latency_ms=tool_latency_ms,
                ),
            ))

            if tool_result and not self._is_error_result(tool_result):
                # Synthesize final spoken answer from tool result.
                # skip_history: the internal synthesis prompt must not be stored
                # as if the user said it — the real exchange is recorded below.
                result_for_synthesis = tool_result
                if len(result_for_synthesis) > _SYNTHESIS_RESULT_CHARS:
                    result_for_synthesis = (
                        result_for_synthesis[:_SYNTHESIS_RESULT_CHARS] + "... [truncated]"
                    )
                synthesis_prompt = (
                    f"The user asked: '{query}'. "
                    f"Tool '{tool_name}' returned: {result_for_synthesis}. "
                    f"Summarize this in 1-2 natural spoken sentences for voice output. "
                    f"No markdown, no bullet points."
                )
                t_synth = time.time()
                final = self.llm.ask(
                    synthesis_prompt, memory_context=memory_context, skip_history=True
                )
                synth_ms = (time.time() - t_synth) * 1000
                logger.info(
                    f"Stage timing: plan={plan_ms:.0f}ms "
                    f"tool={tool_latency_ms:.0f}ms synth={synth_ms:.0f}ms"
                )
                synthesis_error = self.llm.last_error
                if synthesis_error:
                    # The tool succeeded — don't throw its result away just
                    # because the summarizer is down (seen in runtime logs:
                    # 503 turned a good directory listing into "I'm having
                    # trouble connecting").
                    logger.warning(
                        f"Synthesis failed ({synthesis_error[:80]}). "
                        f"Falling back to raw tool output."
                    )
                    final = self._raw_result_fallback(tool_result)
                t_ms = (time.time() - t_start) * 1000
                steps[-1].final_answer = final

                self.llm.record_turn(query, final)
                memory.add_turn("user", query)
                memory.add_turn("assistant", final)
                return AgentResult(
                    query=query,
                    response=final,
                    steps=steps,
                    tool_type_used=tool_type,
                    tool_name_used=tool_name,
                    total_latency_ms=t_ms,
                    input_tokens=self.llm.last_input_tokens,
                    output_tokens=self.llm.last_output_tokens,
                    # The tool delivered — a failed synthesis was recovered via
                    # raw output, so the run still served the user.
                    success=True,
                    error=synthesis_error,
                )

            # Tool failed — inject failure context and retry
            logger.warning(f"Tool '{tool_name}' failed: {tool_result}")
            reason = (tool_result or "no result")[:150]
            current_query = (
                f"{query} (note: tool '{tool_name}' failed with: {reason}. "
                f"Pick a different tool or answer directly.)"
            )

        # ── Fallback: pure LLM after max steps ────────────────────────────
        logger.info("Max planning steps reached. Falling back to pure LLM.")
        answer = self.llm.ask(
            current_query,
            memory_context=memory_context,
            max_history_messages=_DIRECT_HISTORY_MSGS,
        )
        t_ms = (time.time() - t_start) * 1000
        memory.add_turn("user", query)
        memory.add_turn("assistant", answer)
        return AgentResult(
            query=query,
            response=answer,
            steps=steps,
            tool_type_used=ToolType.LLM,
            tool_name_used="llm_fallback",
            total_latency_ms=t_ms,
            input_tokens=self.llm.last_input_tokens,
            output_tokens=self.llm.last_output_tokens,
            success=self.llm.last_error is None,
            error=self.llm.last_error,
        )

    # ── Private helpers ────────────────────────────────────────────────────

    @staticmethod
    def _is_error_result(result: str) -> bool:
        """True if a tool result string signals failure (local or MCP convention)."""
        return result.startswith("[Error]") or result.startswith("[MCP Error]")

    @staticmethod
    def _raw_result_fallback(tool_result: str) -> str:
        """Voice-friendly rendering of a raw tool result, used when the
        synthesis LLM call fails. Collapses whitespace and truncates."""
        text = " ".join(tool_result.split())
        if len(text) > 350:
            text = text[:350].rsplit(" ", 1)[0] + ", and more"
        return f"The summarizer is unreachable right now, so here is the raw result. {text}"

    def _is_rag_query(self, text: str) -> bool:
        lower = text.lower()
        return any(t in lower for t in _RAG_TRIGGERS)

    def _is_direct_llm_query(self, text: str) -> bool:
        """
        Return True if the query can be answered by LLM directly without tools.

        This skips the function-calling planning overhead (~800ms saved) for
        general knowledge questions like "who is the PM of India?" that don't
        need a weather API, file system, calendar, etc.
        """
        lower = text.lower()

        # If it contains tool-specific keywords, don't skip function calling
        if any(p in lower for p in _TOOL_REQUIRED_PATTERNS):
            return False

        # If it starts with a typical knowledge-question prefix → direct LLM
        return any(lower.startswith(p) or lower.startswith("okay, " + p) or
                   lower.startswith("hey, " + p) or lower.startswith("jarvis, " + p)
                   for p in _DIRECT_LLM_STARTERS)

    def _build_tool_schemas(self) -> list[dict]:
        """Combine local command schemas + MCP tool schemas (cached — both
        registries are static after startup)."""
        if self._tool_schemas_cache is None:
            schemas = list(cmd_registry.get_tool_schemas())
            if self.mcp:
                schemas.extend(self.mcp.get_tool_schemas())
            self._tool_schemas_cache = schemas
        return self._tool_schemas_cache

    def _execute_tool(
        self,
        tool_name: str,
        tool_args: dict,
        original_query: str,
    ) -> tuple[str, ToolType]:
        """
        Execute a tool by name, routing to local commands or MCP.

        Returns:
            (result_string, ToolType)
        """
        # Route by ownership — never probe MCP names against the local
        # registry (that logged "unknown tool" warnings on every MCP call).
        if cmd_registry.has_tool(tool_name):
            local_result = cmd_registry.dispatch(tool_name, tool_args.get("text", original_query))
            if local_result is not None:
                return local_result, ToolType.LOCAL
            return f"[Error] Tool '{tool_name}' declined this request.", ToolType.LOCAL

        if self.mcp and self.mcp.has_tool(tool_name):
            return self.mcp.call(tool_name, tool_args), ToolType.MCP

        return f"[Error] Unknown tool: '{tool_name}'", ToolType.LLM
