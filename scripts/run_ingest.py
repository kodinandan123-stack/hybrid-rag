"""CLI script for ingesting documents into the hybrid-rag pipeline.

Usage:
    python scripts/run_ingest.py --corpus-dir data/corpus
    python scripts/run_ingest.py --corpus-dir data/corpus --reset
    python scripts/run_ingest.py --corpus-dir data/corpus --file report.pdf
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion.loader import DocumentLoader
from ingestion.chunker import RecursiveChunker
from retrieval.dense import DenseRetriever
from retrieval.sparse import SparseRetriever
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_ingest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest documents into hybrid-rag")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/corpus"),
        help="Directory containing PDF and markdown source documents",
    )
    parser.add_argument(
        "--file",
        type=Path,
        default=None,
        help="Ingest a single file instead of the full corpus directory",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the Qdrant collection before ingesting",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.CHUNK_SIZE,
        help="Token chunk size (default: %(default)s)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.CHUNK_OVERLAP,
        help="Chunk overlap (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    loader = DocumentLoader()
    chunker = RecursiveChunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    dense = DenseRetriever(
        collection_name=settings.QDRANT_COLLECTION,
        qdrant_url=settings.QDRANT_URL,
        model_name=settings.EMBED_MODEL,
    )
    sparse = SparseRetriever(index_path=settings.BM25_INDEX_PATH)

    if args.reset:
        logger.info("Resetting Qdrant collection '%s'", settings.QDRANT_COLLECTION)
        dense.reset_collection()

    if args.file:
        source_files = [args.file]
        logger.info("Ingesting single file: %s", args.file)
    else:
        source_files = list(args.corpus_dir.rglob("*.pdf")) + list(
            args.corpus_dir.rglob("*.md")
        )
        logger.info(
            "Found %d document(s) in %s", len(source_files), args.corpus_dir
        )

    if not source_files:
        logger.warning("No documents found. Exiting.")
        return

    documents = loader.load(source_files)
    chunks = chunker.chunk(documents)
    logger.info("Created %d chunk(s) from %d document(s)", len(chunks), len(documents))

    logger.info("Indexing into Qdrant...")
    dense.index(chunks)

    logger.info("Building BM25 index...")
    sparse.index(chunks)

    logger.info("Ingestion complete: %d chunks indexed.", len(chunks))


if __name__ == "__main__":
    main()
