"""Unit tests for ingestion/chunker.py."""

from ingestion.chunker import chunk_document, chunk_documents, get_splitter
from ingestion.loader import LoadedDocument


def _doc(text, source="doc.md"):
    return LoadedDocument(text=text, source=source, file_type="markdown", pages=[text])


def test_chunk_document_returns_expected_keys():
    text = "A" * 1200
    chunks = chunk_document(_doc(text))
    assert len(chunks) > 0
    for chunk in chunks:
        assert set(chunk.keys()) == {"text", "source", "chunk_id"}
        assert chunk["source"] == "doc.md"


def test_chunk_document_respects_chunk_size():
    splitter = get_splitter(chunk_size=100, chunk_overlap=10)
    text = "word " * 400
    chunks = chunk_document(_doc(text), splitter=splitter)
    assert all(len(chunk["text"]) <= 100 for chunk in chunks)


def test_chunk_document_ids_are_sequential_and_unique():
    text = "sentence one. sentence two. sentence three. " * 20
    chunks = chunk_document(_doc(text))
    ids = [chunk["chunk_id"] for chunk in chunks]
    assert len(ids) == len(set(ids))
    for index, chunk in enumerate(chunks):
        assert chunk["chunk_id"] == f"doc.md::chunk-{index}"


def test_chunk_documents_flattens_multiple_sources():
    docs = [_doc("first document text " * 50, source="a.md"), _doc("second document text " * 50, source="b.md")]
    chunks = chunk_documents(docs)
    sources = {chunk["source"] for chunk in chunks}
    assert sources == {"a.md", "b.md"}


def test_chunk_document_handles_short_text():
    chunks = chunk_document(_doc("short"))
    assert len(chunks) == 1
    assert chunks[0]["text"] == "short"
