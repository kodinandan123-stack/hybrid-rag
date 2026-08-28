"""tests/test_ingestion.py

Unit tests for the PDF loader and recursive text chunker.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from ingestion.chunker import RecursiveChunker
from ingestion.loader import PDFLoader


class TestPDFLoader:
    def test_load_returns_list(self, tmp_path: Path) -> None:
        fake_pdf = tmp_path / "sample.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake content")
        loader = PDFLoader(str(fake_pdf))
        with patch.object(loader, "_extract_text", return_value="Hello world."):
            docs = loader.load()
        assert isinstance(docs, list)

    def test_load_empty_file_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.pdf"
        empty.write_bytes(b"")
        loader = PDFLoader(str(empty))
        with pytest.raises(ValueError, match="empty"):
            loader.load()

    def test_metadata_contains_source(self, tmp_path: Path) -> None:
        fake_pdf = tmp_path / "report.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")
        loader = PDFLoader(str(fake_pdf))
        with patch.object(loader, "_extract_text", return_value="Content."):
            docs = loader.load()
        for doc in docs:
            assert "source" in doc.metadata


class TestRecursiveChunker:
    def test_chunk_splits_long_text(self) -> None:
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        text = "word " * 100
        chunks = chunker.split(text)
        assert len(chunks) > 1

    def test_chunk_respects_max_size(self) -> None:
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
        text = "sentence. " * 50
        for chunk in chunker.split(text):
            assert len(chunk) <= 120  # allow slight overflow at boundaries

    def test_chunk_empty_string(self) -> None:
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        assert chunker.split("") == []
