"""
loader.py

Loads source documents (PDF and Markdown files) from a corpus directory for
the hybrid RAG ingestion pipeline. Every loaded document keeps track of the
original file path (source) so that later retrieval and generation steps can
cite exactly where an answer came from.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Union

from pypdf import PdfReader

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown"}


@dataclass
class LoadedDocument:
    """A single document loaded from the corpus, with source tracking."""

    text: str
    source: str
    file_type: str
    pages: List[str] = field(default_factory=list)


def _iter_corpus_files(corpus_dir: Path) -> Iterable[Path]:
    """Yield all supported files under the corpus directory, recursively."""
    for path in sorted(corpus_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def load_markdown(path: Path) -> LoadedDocument:
    """Load a single markdown file into a LoadedDocument."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return LoadedDocument(
        text=text,
        source=str(path),
        file_type="markdown",
        pages=[text],
    )


def load_pdf(path: Path) -> LoadedDocument:
    """Load a single PDF file into a LoadedDocument, tracking per-page text."""
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n\n".join(pages)
    return LoadedDocument(
        text=full_text,
        source=str(path),
        file_type="pdf",
        pages=pages,
    )


def load_corpus(corpus_dir: Union[str, Path]) -> List[LoadedDocument]:
    """
    Load all supported documents (PDF, Markdown) from a corpus directory.

    Args:
        corpus_dir: Path to the directory containing source documents.

    Returns:
        A list of LoadedDocument objects, one per file, each with its
        original file path recorded as `source` for traceability.
    """
    corpus_path = Path(corpus_dir)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_path}")

    documents: List[LoadedDocument] = []
    for file_path in _iter_corpus_files(corpus_path):
        try:
            if file_path.suffix.lower() == ".pdf":
                documents.append(load_pdf(file_path))
            else:
                documents.append(load_markdown(file_path))
            logger.info("Loaded %s", file_path)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", file_path, exc)

    return documents


if __name__ == "__main__":
    import sys

    corpus = sys.argv[1] if len(sys.argv) > 1 else "corpus"
    docs = load_corpus(corpus)
    print(f"Loaded {len(docs)} documents from '{corpus}'")
    for doc in docs[:5]:
        print(f"- {doc.source} ({doc.file_type}, {len(doc.text)} chars)")