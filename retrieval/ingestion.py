"""
retrieval/ingestion.py — Document ingestion pipeline for JARVIS RAG.

Supports: .txt, .md, .pdf, .py, .json, .yaml, .csv, .html and more.

Pipeline:
  load_text()      → raw text string
  chunk_text()     → list of Document chunks
  ingest_file()    → chunks from a single file
  ingest_directory() → generator over all chunks in a folder
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Iterator

from core.models import Document

logger = logging.getLogger(__name__)


def _hash_id(source: str, chunk_index: int) -> str:
    """Generate a stable, unique ID for a document chunk."""
    raw = f"{source}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def load_text(path: Path) -> str | None:
    """Load raw text from a supported file format."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf(path)

    text_formats = {
        ".txt", ".md", ".markdown", ".py", ".js", ".ts",
        ".json", ".yaml", ".yml", ".csv", ".rst", ".html", ".htm"
    }
    if suffix in text_formats or suffix == "":
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.error(f"Could not read {path}: {e}")
            return None

    logger.warning(f"Unsupported file format: {suffix} ({path.name})")
    return None


def _load_pdf(path: Path) -> str | None:
    """Extract text from a PDF using pypdf."""
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
    except ImportError:
        logger.error("pypdf not installed. Run: pip install pypdf")
        return None
    except Exception as e:
        logger.error(f"PDF extraction failed for {path}: {e}")
        return None


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Document]:
    """
    Split text into overlapping Document chunks.

    Uses paragraph boundaries first, then falls back to character windows.
    """
    if not text or not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[Document] = []
    current = ""
    chunk_index = 0

    for para in paragraphs:
        if len(current) + len(para) + 2 <= chunk_size:
            current = (current + "\n\n" + para).strip()
        else:
            if current:
                chunks.append(Document(
                    doc_id=_hash_id(source, chunk_index),
                    source=source,
                    chunk_index=chunk_index,
                    content=current,
                    metadata={"source": source, "chunk": chunk_index}
                ))
                chunk_index += 1
                current = current[-overlap:].strip() + "\n\n" + para
            else:
                current = para

    if current:
        chunks.append(Document(
            doc_id=_hash_id(source, chunk_index),
            source=source,
            chunk_index=chunk_index,
            content=current,
            metadata={"source": source, "chunk": chunk_index}
        ))

    logger.debug(f"Chunked '{source}' → {len(chunks)} chunks")
    return chunks


def ingest_file(path: Path, chunk_size: int = 512, overlap: int = 64) -> list[Document]:
    """Load a file and split it into Document chunks."""
    text = load_text(path)
    if text is None:
        return []
    return chunk_text(text, source=path.name, chunk_size=chunk_size, overlap=overlap)


def ingest_directory(
    directory: Path,
    chunk_size: int = 512,
    overlap: int = 64,
    extensions: set[str] | None = None,
) -> Iterator[Document]:
    """
    Walk a directory and yield Document chunks from all supported files.

    Args:
        directory:  Root directory to scan
        chunk_size: Characters per chunk
        overlap:    Overlap between adjacent chunks
        extensions: Optional whitelist of file extensions (e.g. {'.pdf', '.txt'})
    """
    if not directory.exists():
        logger.warning(f"Docs directory not found: {directory}")
        return

    for file_path in sorted(directory.rglob("*")):
        if not file_path.is_file():
            continue
        if extensions and file_path.suffix.lower() not in extensions:
            continue
        chunks = ingest_file(file_path, chunk_size=chunk_size, overlap=overlap)
        if chunks:
            logger.info(f"Ingested '{file_path.name}' → {len(chunks)} chunks")
        yield from chunks
