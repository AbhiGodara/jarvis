"""
retrieval/rag_engine.py — Retrieval-Augmented Generation for JARVIS.

Given a user query:
  1. Retrieve the top-k most relevant document chunks (semantic search)
  2. Build a grounded prompt combining the query + retrieved context
  3. Pass to the LLM for a factual, source-cited answer

Use cases:
  - "According to my notes, what did I say about X?"
  - "Summarize the uploaded PDF"
  - "What does the interview transcript say about onboarding?"
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from core.models import RetrievalResult

if TYPE_CHECKING:
    from retrieval.vector_store import VectorStore
    from llm import LLMClient

logger = logging.getLogger(__name__)

_MIN_SCORE = 0.30

# Note: the source is appended programmatically after the answer (see
# query()), so the model is told NOT to cite — otherwise sources were
# spoken twice.
_RAG_SYSTEM_PROMPT = """
You are answering questions using ONLY the provided document excerpts.
Rules:
- Answer only from the context. Do not use outside knowledge.
- If the context does not contain the answer, say so clearly.
- Keep answers concise (2-4 sentences for voice output).
- Do not use bullet points or markdown formatting.
- Do not mention source names or cite documents — that is added automatically.
""".strip()


class RAGEngine:
    """
    Retrieval-Augmented Generation engine.

    Wraps a VectorStore for retrieval and an LLMClient for generation.
    """

    def __init__(
        self,
        vector_store: "VectorStore",
        llm_client: "LLMClient",
        top_k: int = 4,
        min_score: float = _MIN_SCORE,
    ):
        self.store = vector_store
        self.llm = llm_client
        self.top_k = top_k
        self.min_score = min_score

    def _prepare(self, question: str):
        """Shared retrieval/guard step for query() and query_stream().

        Returns (early_answer, retrieval, grounded_prompt, source_note) —
        early_answer is a ready-to-speak message when retrieval can't help
        (store not ready, nothing relevant), in which case the prompt/note
        are None.
        """
        if not self.store.ready:
            return (
                "The document store is not ready. Place documents in the docs/ folder and restart.",
                RetrievalResult(question, [], []),
                None,
                None,
            )

        retrieval = self.store.search(question, top_k=self.top_k)

        good_chunks = [
            (chunk, score)
            for chunk, score in zip(retrieval.chunks, retrieval.scores)
            if score >= self.min_score
        ]

        if not good_chunks:
            return (
                "I searched the document store but couldn't find relevant information. "
                "Try asking me directly without the document reference.",
                retrieval,
                None,
                None,
            )

        context_parts = []
        for i, (chunk, score) in enumerate(good_chunks):
            context_parts.append(
                f"[Source {i+1}: {chunk.source}, relevance={score:.2f}]\n{chunk.content}"
            )
        context = "\n\n---\n\n".join(context_parts)

        grounded_prompt = (
            f"Using the following document excerpts, answer this question: {question}\n\n"
            f"DOCUMENTS:\n{context}"
        )

        sources = list(dict.fromkeys(c.source for c, _ in good_chunks))
        source_note = f" Source{'s' if len(sources) > 1 else ''}: {', '.join(sources)}."

        return None, retrieval, grounded_prompt, source_note

    def query(self, question: str) -> tuple[str, RetrievalResult]:
        """
        Answer a question using retrieved documents.

        Returns:
            (answer_text, retrieval_result)
        """
        early_answer, retrieval, grounded_prompt, source_note = self._prepare(question)
        if early_answer is not None:
            return early_answer, retrieval

        answer = self.llm.ask(
            grounded_prompt,
            memory_context=_RAG_SYSTEM_PROMPT,
            skip_history=True,   # RAG answers don't pollute conversation history
        )

        return answer + source_note, retrieval

    def query_stream(self, question: str):
        """Streaming variant of query() (Mk-III Phase 1).

        Returns (delta_iterator, retrieval_result). The iterator yields the
        grounded answer as text deltas with the source note as the final
        delta; guard failures yield their message as a single delta.
        """
        early_answer, retrieval, grounded_prompt, source_note = self._prepare(question)
        if early_answer is not None:
            return iter([early_answer]), retrieval

        def _gen():
            yield from self.llm.ask_stream(
                grounded_prompt,
                memory_context=_RAG_SYSTEM_PROMPT,
                skip_history=True,
            )
            yield source_note

        return _gen(), retrieval

    def ingest_and_index(self, docs_dir: str = "docs") -> dict:
        """
        Convenience method: ingest all documents in a directory and add to store.
        Returns stats dict.
        """
        from pathlib import Path
        from retrieval.ingestion import ingest_directory

        directory = Path(docs_dir)
        chunks = list(ingest_directory(directory))
        result = self.store.sync(chunks)

        stats = {
            "total_chunks": len(chunks),
            "new_chunks_added": result["added"],
            "stale_chunks_removed": result["deleted"],
            "total_in_store": self.store.count(),
            "sources": self.store.list_sources(),
        }
        logger.info(f"Ingestion complete: {stats}")
        return stats
