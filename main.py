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

Conversation mode:
  After each response JARVIS stays listening for a follow-up command for
  `conversation_followup_secs` seconds (configurable). Only returns to
  wake-word idle mode if the user goes silent. Set conversation_followup_secs
  to 0 in config.yaml to always require "hey jarvis".
"""
import atexit
import logging
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
from tts import speak, stop as stop_tts, preload as preload_tts, speak_offline
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


# Exact wake word + common Whisper mishearings of "Jarvis" with Indian English.
# Sorted longest-first so "hey yavish" is stripped before "yavish".
_WAKE_PREFIXES = sorted([
    "hey jarvis", "jarvis",
    "hey yavish", "yavish",   # Most common Indian-accent Whisper mishearing
    "hey travis", "travis",
    "hey jarvish", "jarvish",
    "hey service", "hey harris",
], key=len, reverse=True)


def _strip_wake_word(text: str) -> str:
    """Remove accidental wake-word transcription from the start of a command."""
    for prefix in _WAKE_PREFIXES:
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def _on_shutdown() -> None:
    logger.info("JARVIS shutting down.")
    speak_offline("Going offline. Goodbye, sir.")


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
    # Preload activation greeting in background — plays from cache on first hit.
    import threading
    threading.Thread(
        target=preload_tts, args=("Yes, sir?",), daemon=True
    ).start()
    # Use offline TTS for startup: OpenAI TTS can take 30+ s with retries and
    # would block the wake-word listener.  pyttsx3 speaks in < 100 ms.
    speak_offline("JARVIS online, sir. Ready when you are.")

    # ── Main loop ─────────────────────────────────────────────────────────
    # Outer loop: idle → wake word → conversation session → idle
    # Inner loop: conversation session (multiple turns without re-waking)
    should_exit = False

    while not should_exit:
        try:
            # ── Idle: wait for wake word ──────────────────────────────────
            wait_for_wake_word(oww=wake_model, model_name=cfg.wake_word)
            stop_tts()
            _speak("Yes, sir?")

            # ── Conversation session ──────────────────────────────────────
            # is_first: True for the first command after wake, False for follow-ups.
            # Follow-ups use a shorter timeout so we return to idle quickly on
            # silence, without making the user wait 8 s for the first command.
            is_first_listen = True

            while not should_exit:
                # Scale the follow-up timeout with the previous response length:
                # a long response takes longer to speak, so the user needs more
                # time to collect their thoughts before responding.
                _prev_response_chars = getattr(_speak, "_last_response_chars", 0)
                follow_up_timeout = cfg.conversation_followup_secs + min(
                    _prev_response_chars // 60, 6
                )
                listen_timeout = (
                    cfg.stt_timeout if is_first_listen
                    else follow_up_timeout
                )

                t_stt = time.time()
                command = listen(timeout=listen_timeout, phrase_limit=cfg.stt_phrase_limit)
                logger.info(
                    f"Stage timing: stt={((time.time() - t_stt) * 1000):.0f}ms "
                    f"(incl. waiting for speech)"
                )

                if command is None:
                    # True silence: user said nothing within the timeout window
                    if is_first_listen:
                        _speak("I didn't catch that, sir.")
                    break

                if command == "":
                    # Whisper returned empty: audio was brief noise / a cough /
                    # an echo — NOT intentional speech.  Don't drop the
                    # conversation; give one more chance with a short window.
                    logger.info("STT returned empty (noise). Retrying once.")
                    command = listen(
                        timeout=min(follow_up_timeout, 6),
                        phrase_limit=cfg.stt_phrase_limit,
                    )
                    if not command:   # still nothing or noise → give up
                        break

                command = _strip_wake_word(command.strip())
                if not command:
                    break

                logger.info(f"Command: '{command}'")

                if _is_stop_command(command):
                    logger.info("Stop command — exiting loop.")
                    should_exit = True
                    break

                # Truncate runaway transcriptions
                if len(command) > cfg.llm_max_input_chars:
                    command = command[:cfg.llm_max_input_chars]

                # ── Process command ───────────────────────────────────────
                result = planner.run(command)
                logger.info(
                    f"Result: tool={result.tool_type_used.value}/{result.tool_name_used} "
                    f"latency={result.total_latency_ms:.0f}ms"
                )

                if eval_logger:
                    eval_logger.log(result)

                response = result.response
                logger.info(f"Response: '{response[:80]}{'...' if len(response) > 80 else ''}'")

                # Speak synchronously — must finish before we listen again,
                # otherwise the mic captures JARVIS's own voice.
                # Store length so the follow-up timeout can scale.
                _speak._last_response_chars = len(response)  # type: ignore[attr-defined]
                _speak(response)

                # Conversation mode disabled — go back to wake word after reply
                if cfg.conversation_followup_secs <= 0:
                    break

                # After launching external media (YouTube, apps), the mic will
                # immediately pick up audio from the browser/application.
                # Exit conversation mode so the user has to say "hey jarvis"
                # again when they're done watching/using it.
                if response.lower().startswith("opening ") or response.lower().startswith("launching "):
                    break

                # Echo-guard: longer response = more room reverb to decay.
                # Formula: 0.5 s base + 3 ms per character, capped at 1.5 s.
                echo_guard_s = min(0.5 + len(response) * 0.003, 1.5)
                time.sleep(echo_guard_s)
                is_first_listen = False
                # Inner loop continues: listen for follow-up with short timeout

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt.")
            should_exit = True
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            _speak("Something went wrong, sir. I'll keep listening.")

    # ── Shutdown ──────────────────────────────────────────────────────────
    if mcp_manager:
        mcp_manager.stop()
    if eval_logger:
        eval_logger.print_report(last_n=50)


if __name__ == "__main__":
    main()
