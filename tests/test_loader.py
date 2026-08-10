"""Unit tests for ingestion/loader.py."""

from unittest.mock import patch

import pytest

from ingestion.loader import LoadedDocument, load_corpus, load_markdown


def test_load_markdown_reads_text(tmp_path):
    md_file = tmp_path / "note.md"
    md_file.write_text("# Title\n\nSome content.", encoding="utf-8")

    doc = load_markdown(md_file)

    assert isinstance(doc, LoadedDocument)
    assert doc.file_type == "markdown"
    assert doc.source == str(md_file)
    assert doc.pages == [doc.text]
    assert "Some content." in doc.text


def test_load_corpus_raises_for_missing_directory(tmp_path):
    missing_dir = tmp_path / "does-not-exist"

    with pytest.raises(FileNotFoundError):
        load_corpus(missing_dir)


def test_load_corpus_loads_markdown_and_skips_unsupported_files(tmp_path):
    (tmp_path / "a.md").write_text("alpha", encoding="utf-8")
    (tmp_path / "b.txt").write_text("ignored", encoding="utf-8")

    docs = load_corpus(tmp_path)

    assert len(docs) == 1
    assert docs[0].source.endswith("a.md")
    assert docs[0].file_type == "markdown"


@patch("ingestion.loader.load_markdown", side_effect=RuntimeError("boom"))
def test_load_corpus_skips_files_that_fail_to_load(mock_load_markdown, tmp_path):
    (tmp_path / "broken.md").write_text("x", encoding="utf-8")

    docs = load_corpus(tmp_path)

    assert docs == []
    mock_load_markdown.assert_called_once()
