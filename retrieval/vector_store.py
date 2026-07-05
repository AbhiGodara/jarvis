"""
retrieval/vector_store.py — Vector database layer for JARVIS RAG.

Uses ChromaDB (local, embedded, no server required) with
sentence-transformers for embedding generation.

Why ChromaDB?
  - Runs fully embedded in-process (no Docker, no server)
  - Persistent storage out of the box
  - Fast for the document sizes a voice assistant handles
  - Easy to swap for Pinecone/Weaviate later if needed

Why sentence-transformers?
  - Free, runs locally, no API key needed
  - 'all-MiniLM-L6-v2' is ~80MB and runs fast on CPU
  - Good quality for retrieval tasks
"""
from __future__ import annotations

import logging
from pathlib import Path

from core.models import Document, RetrievalResult

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "all-MiniLM-L6-v2"
_DEFAULT_COLLECTION = "jarvis_docs"
# Anchored to this package so the same store is found regardless of the
# working directory JARVIS was launched from.
_PERSIST_DIR = str(Path(__file__).resolve().parent / "chroma_db")


class VectorStore:
    """
    Persistent vector database for document retrieval.

    Usage:
        store = VectorStore()
        if store.initialize():
            store.add(document_chunks)
            results = store.search("what is X", top_k=4)
    """

    def __init__(
        self,
        persist_dir: str = _PERSIST_DIR,
        collection_name: str = _DEFAULT_COLLECTION,
        embedding_model: str = _DEFAULT_MODEL,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self._collection = None
        self._embedder = None
        self._ready = False

    def initialize(self) -> bool:
        """
        Initialize ChromaDB and the embedding model.
        Returns True on success, False if dependencies are missing.
        """
        try:
            import chromadb
            from chromadb.config import Settings

            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )

            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )

            logger.info(
                f"ChromaDB initialized at '{self.persist_dir}' "
                f"({self._collection.count()} existing chunks)"
            )

        except ImportError:
            logger.error("chromadb not installed. Run: pip install chromadb")
            return False
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            return False

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model '{self.embedding_model_name}'...")
            self._embedder = SentenceTransformer(self.embedding_model_name)
            logger.info("Embedding model ready.")
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            return False
        except Exception as e:
            logger.error(f"Embedding model load failed: {e}")
            return False

        self._ready = True
        return True

    def add(self, documents: list[Document]) -> int:
        """
        Embed and store a list of Document chunks.
        Skips documents already stored (by doc_id).
        Returns the number of new documents added.
        """
        if not self._ready or not documents:
            return 0

        existing_ids = set(self._collection.get(ids=[d.doc_id for d in documents])["ids"])
        new_docs = [d for d in documents if d.doc_id not in existing_ids]

        if not new_docs:
            logger.debug("All documents already in store.")
            return 0

        texts = [d.content for d in new_docs]
        ids = [d.doc_id for d in new_docs]
        metadatas = [d.metadata for d in new_docs]

        embeddings = self._embedder.encode(texts, show_progress_bar=False).tolist()

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info(f"Added {len(new_docs)} new chunks to vector store.")
        return len(new_docs)

    def search(self, query: str, top_k: int = 4) -> RetrievalResult:
        """
        Semantic similarity search.

        Returns RetrievalResult with matching Document chunks and scores.
        """
        if not self._ready:
            return RetrievalResult(query=query, chunks=[], scores=[])

        if self._collection.count() == 0:
            return RetrievalResult(query=query, chunks=[], scores=[])

        query_embedding = self._embedder.encode([query], show_progress_bar=False).tolist()

        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        chunks: list[Document] = []
        scores: list[float] = []

        for i, (text, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        )):
            score = 1.0 - dist  # cosine distance → similarity
            chunks.append(Document(
                doc_id=results["ids"][0][i],
                source=meta.get("source", "unknown"),
                chunk_index=meta.get("chunk", 0),
                content=text,
                metadata=meta,
            ))
            scores.append(round(score, 4))

        return RetrievalResult(query=query, chunks=chunks, scores=scores)

    def delete_source(self, source_name: str) -> int:
        """Remove all chunks from a specific source document."""
        if not self._ready:
            return 0
        results = self._collection.get(where={"source": source_name})
        ids = results.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} chunks from source '{source_name}'.")
        return len(ids)

    def count(self) -> int:
        if not self._ready:
            return 0
        return self._collection.count()

    def list_sources(self) -> list[str]:
        if not self._ready or self._collection.count() == 0:
            return []
        all_meta = self._collection.get(include=["metadatas"])["metadatas"]
        return sorted(set(m.get("source", "") for m in all_meta))

    @property
    def ready(self) -> bool:
        return self._ready
