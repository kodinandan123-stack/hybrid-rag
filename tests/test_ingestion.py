"""tests/test_ingestion.py

Integration-style tests spanning the PDF loader and the recursive text
chunker together (loader.py and chunker.py are unit-tested individually in
test_loader.py and test_chunker.py).
"""

from unittest.mock import patch

from ingestion.chunker import chunk_document
from ingestion.loader import LoadedDocument, load_corpus, load_pdf


class TestPDFLoader:
    def test_load_pdf_returns_loaded_document(self, tmp_path):
        fake_pdf = tmp_path / "sample.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake content")

        with patch("ingestion.loader.PdfReader") as mock_reader_cls:
            mock_page = type(
                "Page", (), {"extract_text": lambda self: "Hello world."}
            )()
            mock_reader_cls.return_value.pages = [mock_page]
            doc = load_pdf(fake_pdf)

        assert isinstance(doc, LoadedDocument)
        assert doc.file_type == "pdf"
        assert doc.source == str(fake_pdf)
        assert "Hello world." in doc.text

    def test_load_corpus_picks_up_pdf_alongside_markdown(self, tmp_path):
        (tmp_path / "note.md").write_text("Markdown content.", encoding="utf-8")
        pdf_path = tmp_path / "report.pdf"
        pdf_path.write_bytes(b"%PDF-1.4 fake")

        with patch("ingestion.loader.PdfReader") as mock_reader_cls:
            mock_page = type(
                "Page", (), {"extract_text": lambda self: "PDF content."}
            )()
            mock_reader_cls.return_value.pages = [mock_page]
            docs = load_corpus(tmp_path)

        file_types = {doc.file_type for doc in docs}
        assert file_types == {"markdown", "pdf"}

    def test_metadata_contains_source(self, tmp_path):
        fake_pdf = tmp_path / "report.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        with patch("ingestion.loader.PdfReader") as mock_reader_cls:
            mock_page = type("Page", (), {"extract_text": lambda self: "Content."})()
            mock_reader_cls.return_value.pages = [mock_page]
            doc = load_pdf(fake_pdf)

        assert doc.source == str(fake_pdf)


class TestIngestionToChunking:
    def test_loaded_document_can_be_chunked(self):
        doc = LoadedDocument(
            text="word " * 200,
            source="doc.md",
            file_type="markdown",
            pages=["word " * 200],
        )
        chunks = chunk_document(doc)
        assert len(chunks) > 1
        assert all(chunk["source"] == "doc.md" for chunk in chunks)

    def test_empty_document_yields_no_chunks(self):
        doc = LoadedDocument(
            text="", source="empty.md", file_type="markdown", pages=[""]
        )
        assert chunk_document(doc) == []
