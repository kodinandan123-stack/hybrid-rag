"""
reindex.py

CLI script that runs the full ingestion pipeline (load -> chunk) and posts
the resulting chunks to the running API's /index endpoint, so the corpus
can be re-indexed without restarting the service.
"""

from __future__ import annotations

import argparse
import sys

import requests

from ingestion.chunker import chunk_documents
from ingestion.loader import load_corpus


def reindex(corpus_dir: str, api_url: str) -> int:
    """Load and chunk the corpus, then POST the chunks to the /index endpoint."""
    documents = load_corpus(corpus_dir)
    chunks = chunk_documents(documents)

    if not chunks:
        print(f"No chunks produced from '{corpus_dir}'; nothing to index.")
        return 0

    response = requests.post(f"{api_url.rstrip('/')}/index", json=chunks)
    response.raise_for_status()
    indexed = response.json().get("indexed", len(chunks))
    print(f"Indexed {indexed} chunks from {len(documents)} documents into {api_url}")
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser(description="Reindex the corpus into the hybrid RAG API.")
    parser.add_argument("corpus", nargs="?", default="corpus", help="Path to the corpus directory")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Base URL of the running API")
    args = parser.parse_args()

    try:
        reindex(args.corpus, args.api_url)
    except Exception as exc:
        print(f"Reindex failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
