"""CLI script for querying the hybrid RAG pipeline."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.main import app  # noqa: E402 – imported for pipeline access
from retrieval.hybrid import HybridRetriever
from generation.generator import Generator
from config.settings import Settings


def parse_args() -> argparse.Namespace:
      parser = argparse.ArgumentParser(description="Query the hybrid RAG pipeline")
      parser.add_argument("query", help="Question to ask the RAG system")
      parser.add_argument(
          "--top-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)"
      )
      parser.add_argument(
          "--output", choices=["text", "json"], default="text",
          help="Output format: plain text or JSON (default: text)"
      )
      parser.add_argument(
          "--no-rerank", action="store_true",
          help="Skip cross-encoder reranking step"
      )
      return parser.parse_args()


def run_query(query: str, top_k: int, rerank: bool) -> dict:
      settings = Settings()
      retriever = HybridRetriever(settings=settings)
      generator = Generator(settings=settings)

    chunks = retriever.search(query, top_k=top_k, rerank=rerank)
    answer = generator.generate(query=query, chunks=chunks)

    return {
              "query": query,
              "answer": answer,
              "chunks": [
                            {
                                              "text": c.text,
                                              "source": c.metadata.get("source", ""),
                                              "score": round(c.score, 4),
                            }
                            for c in chunks
              ],
    }


def main() -> None:
      args = parse_args()
      result = run_query(
          query=args.query,
          top_k=args.top_k,
          rerank=not args.no_rerank,
      )

    if args.output == "json":
              print(json.dumps(result, indent=2))
else:
          print(f"Answer:\n{result['answer']}\n")
          print("Sources:")
          for chunk in result["chunks"]:
                        print(f"  [{chunk['score']}] {chunk['source']}")


if __name__ == "__main__":
      main()
