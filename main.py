"""
main.py — JARVIS Entry Point.

Startup sequence:
  1. Load config (typed Config dataclass)
  2. Setup logging
  3. Initialize LLM client
  4. Initialize MCP manager (built-in filesystem + any configured servers)
  5. Initialize vector store + RAG engine (if chromadb is installed)
  6. Initialize Planner Agent
  7. Initialize evaluation logger
  8. Pre-load wake word model (once, not per loop)
  9. Start voice loop

Architecture:
  Wake Word → STT → PlannerAgent → TTS
                       ├── Local commands (fast path, no LLM cost)
                       ├── RAG engine (document-grounded Q&A)
                       ├── MCP tools (filesystem, custom servers)
                       └── LLM (open-ended reasoning + function calling)
"""
import atexit
import logging
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

# ── Core setup (must happen before other imports) ─────────────────────────────
from core.config import get_config
from core.logging_config import setup_logging

load_dotenv()
cfg = get_config()
setup_logging(level=cfg.log_level, log_to_file=cfg.log_to_file, log_file=cfg.log_file_path)

logger = logging.getLogger(__name__)

# ── Module imports ─────────────────────────────────────────────────────────────
from wake_word import wait_for_wake_word, load_model
from stt import listen
from tts import speak, stop as stop_tts
from llm import LLMClient
from agents.planner import PlannerAgent
from commands.registry import auto_discover
from evaluation.eval_logger import EvalLogger

# ── Constants ──────────────────────────────────────────────────────────────────
STOP_PHRASES = frozenset([
    "stop", "exit", "quit", "goodbye", "shut down", "power off", "bye",
    "stop listening", "go to sleep", "goodbye jarvis", "shut down jarvis",
])


def _is_stop_command(command: str) -> bool:
    """
    True only when the whole command is a stop phrase (after stripping
    punctuation and politeness). Substring checks are too dangerous here:
    'quit' is inside 'quite', 'stop' is inside 'stop the timer'.
    """
    normalized = command.strip().strip("?!.,;:").strip()
    for filler in ("please ", "jarvis "):
        if normalized.startswith(filler):
            normalized = normalized[len(filler):].strip()
    normalized = normalized.removesuffix(" please").strip()
    return normalized in STOP_PHRASES


def _speak(text: str) -> None:
    speak(text)


def _speak_async(text: str) -> None:
    threading.Thread(target=_speak, args=(text,), daemon=True).start()


def _strip_wake_word(text: str) -> str:
    """Remove accidental wake-word transcription from the start of a command."""
    for prefix in ["hey jarvis", "jarvis"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


def _on_shutdown() -> None:
    logger.info("JARVIS shutting down.")
    _speak("Going offline. Goodbye.")


atexit.register(_on_shutdown)


def _init_mcp() -> "MCPManager | None":
    """Initialize MCP manager with built-in filesystem + any configured servers."""
    if not cfg.mcp_enabled:
        logger.info("MCP disabled in config.")
        return None

    try:
        from mcp_layer.mcp_manager import MCPManager
        from mcp_layer import builtin_filesystem

        manager = MCPManager(server_configs=list(cfg.mcp_servers))

        # Register built-in filesystem tools (no subprocess required)
        manager.register_builtin("filesystem", builtin_filesystem.BUILTIN_TOOLS)
        logger.info(f"MCP built-in filesystem tools registered.")

        # Start any external configured MCP servers
        if cfg.mcp_servers:
            manager.start()

        total = len(manager.list_tools())
        logger.info(f"MCP ready. Total tools: {total}")
        return manager

    except Exception as e:
        logger.warning(f"MCP initialization failed: {e}")
        return None


def _init_rag(llm: "LLMClient") -> "RAGEngine | None":
    """Initialize ChromaDB vector store and RAG engine.
    
    Skips loading the ~80MB embedding model if there are no real user
    documents in the docs/ folder (only README.md doesn't count).
    """
    if not cfg.rag_enabled:
        logger.info("RAG disabled in config.")
        return None

    # Check if there are real user documents (skip README.md)
    docs_dir = Path(cfg.rag_docs_dir)
    real_docs = [
        f for f in docs_dir.iterdir()
        if f.is_file() and f.name.lower() not in {"readme.md", "readme.txt", ".gitkeep"}
    ] if docs_dir.exists() else []

    if not real_docs:
        logger.info(
            f"RAG: No user documents found in '{docs_dir}'. "
            f"Place .pdf/.txt/.md files there to enable document Q&A. "
            f"Skipping embedding model load to save startup time."
        )
        return None

    try:
        from retrieval.vector_store import VectorStore
        from retrieval.rag_engine import RAGEngine

        store = VectorStore()
        if not store.initialize():
            logger.warning("VectorStore initialization failed. RAG unavailable.")
            logger.warning("Install with: pip install chromadb sentence-transformers")
            return None

        from retrieval.ingestion import ingest_directory
        chunks = list(ingest_directory(
            docs_dir,
            chunk_size=cfg.rag_chunk_size,
            overlap=cfg.rag_chunk_overlap,
        ))
        if chunks:
            added = store.add(chunks)
            logger.info(
                f"RAG: ingested {added} new chunks from '{docs_dir}' "
                f"(total: {store.count()})"
            )

        engine = RAGEngine(vector_store=store, llm_client=llm, top_k=cfg.rag_top_k)
        logger.info("RAG engine ready.")
        return engine

    except ImportError:
        logger.info("RAG dependencies not installed. RAG disabled.")
        logger.info("To enable: pip install chromadb sentence-transformers pypdf")
        return None
    except Exception as e:
        logger.warning(f"RAG initialization failed: {e}")
        return None


def main() -> None:
    """Main entry point. Initialises JARVIS and runs the voice assistant loop."""
    logger.info("=" * 60)
    logger.info("  JARVIS initialising...")
    logger.info("=" * 60)

    # ── Auto-discover command plugins ─────────────────────────────────────
    auto_discover()
    logger.info("Command plugins loaded.")

    # ── LLM ───────────────────────────────────────────────────────────────
    llm = LLMClient(
        model_name=cfg.llm_model,
        max_history_turns=cfg.llm_max_history_turns,
    )

    # ── MCP ───────────────────────────────────────────────────────────────
    mcp_manager = _init_mcp()

    # ── RAG ───────────────────────────────────────────────────────────────
    rag_engine = _init_rag(llm)

    # ── Evaluation ────────────────────────────────────────────────────────
    eval_logger = None
    if cfg.eval_enabled:
        eval_logger = EvalLogger(db_path=cfg.eval_db_path)

    # ── Planner Agent ─────────────────────────────────────────────────────
    planner = PlannerAgent(
        llm=llm,
        mcp_manager=mcp_manager,
        rag_engine=rag_engine,
        max_steps=cfg.agent_max_steps,
    )

    # ── Pre-load wake word model ONCE ─────────────────────────────────────
    wake_model = load_model(cfg.wake_word)

    logger.info("JARVIS is ready.")
    _speak("JARVIS online. Agentic mode active.")

    # ── Main loop ─────────────────────────────────────────────────────────
    while True:
        try:
            # Phase 1: Wait for wake word (uses pre-loaded model)
            wait_for_wake_word(oww=wake_model, model_name=cfg.wake_word)
            stop_tts()
            _speak("Yes?")

            # Phase 2: Listen for command
            t_stt = time.time()
            command = listen(timeout=cfg.stt_timeout, phrase_limit=cfg.stt_phrase_limit)
            logger.info(f"Stage timing: stt={((time.time() - t_stt) * 1000):.0f}ms (incl. waiting for speech)")

            if not command:
                _speak("I didn't catch that.")
                continue

            command = _strip_wake_word(command.strip())
            if not command:
                continue

            logger.info(f"Command: '{command}'")

            # Phase 3: Stop keyword check
            if _is_stop_command(command):
                logger.info("Stop command — exiting loop.")
                break

            # Phase 4: Truncate if excessively long
            if len(command) > cfg.llm_max_input_chars:
                command = command[:cfg.llm_max_input_chars]

            # Phase 5: Planner Agent processes command
            result = planner.run(command)
            logger.info(
                f"Result: tool={result.tool_type_used.value}/{result.tool_name_used} "
                f"latency={result.total_latency_ms:.0f}ms"
            )

            # Phase 6: Log evaluation
            if eval_logger:
                eval_logger.log(result)

            # Phase 7: Speak response
            response = result.response
            logger.info(f"Response: '{response[:80]}{'...' if len(response) > 80 else ''}'")
            _speak_async(response)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt.")
            break
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            _speak("Something went wrong. I'll keep listening.")

    # ── Shutdown ──────────────────────────────────────────────────────────
    if mcp_manager:
        mcp_manager.stop()
    if eval_logger:
        eval_logger.print_report(last_n=50)


if __name__ == "__main__":
    main()
