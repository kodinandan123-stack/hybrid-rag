"""CLI script to export the FAISS dense index and BM25 sparse index to disk.

Usage:
    python scripts/export_index.py --output-dir ./index_export
        python scripts/export_index.py --output-dir ./index_export --compress
        """
from __future__ import annotations

import argparse
import gzip
import json
import logging
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s [%(levelname)s] %(message)s",
      datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> Path:
      path.mkdir(parents=True, exist_ok=True)
      return path


def export_faiss_index(
      index_path: Path,
      output_dir: Path,
      compress: bool = False,
) -> Path:
      """Copy (or compress) the FAISS index file to *output_dir*."""
      if not index_path.exists():
                raise FileNotFoundError(f"FAISS index not found: {index_path}")

      dest_name = index_path.name + (".gz" if compress else "")
      dest = output_dir / dest_name

    if compress:
              with index_path.open("rb") as src, gzip.open(dest, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        log.info("FAISS index compressed -> %s", dest)
else:
        shutil.copy2(index_path, dest)
        log.info("FAISS index copied -> %s", dest)

    return dest


def export_bm25_index(
      index_path: Path,
      output_dir: Path,
      compress: bool = False,
) -> Path:
      """Copy (or compress) the BM25 pickle file to *output_dir*."""
    if not index_path.exists():
              raise FileNotFoundError(f"BM25 index not found: {index_path}")

    dest_name = index_path.name + (".gz" if compress else "")
    dest = output_dir / dest_name

    if compress:
              with index_path.open("rb") as src, gzip.open(dest, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        log.info("BM25 index compressed -> %s", dest)
else:
        shutil.copy2(index_path, dest)
        log.info("BM25 index copied -> %s", dest)

    return dest


def write_manifest(
      output_dir: Path,
      faiss_dest: Optional[Path],
      bm25_dest: Optional[Path],
      compress: bool,
) -> Path:
      """Write a JSON manifest describing the exported artefacts."""
    manifest = {
              "exported_at": datetime.now(timezone.utc).isoformat(),
              "compressed": compress,
              "files": {
                            "faiss_index": faiss_dest.name if faiss_dest else None,
                            "bm25_index": bm25_dest.name if bm25_dest else None,
              },
    }
    dest = output_dir / "manifest.json"
    dest.write_text(json.dumps(manifest, indent=2))
    log.info("Manifest written -> %s", dest)
    return dest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
      parser = argparse.ArgumentParser(
                description="Export hybrid-RAG FAISS and BM25 indexes to a directory."
      )
    parser.add_argument(
              "--output-dir",
              required=True,
              type=Path,
              help="Destination directory for exported index files.",
    )
    parser.add_argument(
              "--faiss-path",
              type=Path,
              default=Path("data/index/faiss.index"),
              help="Path to the FAISS index file (default: data/index/faiss.index).",
    )
    parser.add_argument(
              "--bm25-path",
              type=Path,
              default=Path("data/index/bm25.pkl"),
              help="Path to the BM25 pickle file (default: data/index/bm25.pkl).",
    )
    parser.add_argument(
              "--compress",
              action="store_true",
              help="Gzip-compress the exported files.",
    )
    parser.add_argument(
              "--skip-faiss",
              action="store_true",
              help="Skip exporting the FAISS index.",
    )
    parser.add_argument(
              "--skip-bm25",
              action="store_true",
              help="Skip exporting the BM25 index.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
      args = parse_args(argv)
    output_dir = _ensure_dir(args.output_dir)
    log.info("Exporting indexes to %s", output_dir.resolve())

    faiss_dest: Optional[Path] = None
    bm25_dest: Optional[Path] = None

    if not args.skip_faiss:
              try:
                            faiss_dest = export_faiss_index(args.faiss_path, output_dir, args.compress)
except FileNotFoundError as exc:
            log.warning("%s — skipping FAISS export", exc)

    if not args.skip_bm25:
              try:
                            bm25_dest = export_bm25_index(args.bm25_path, output_dir, args.compress)
except FileNotFoundError as exc:
            log.warning("%s — skipping BM25 export", exc)

    if faiss_dest is None and bm25_dest is None:
              log.error("Nothing was exported.")
        return 1

    write_manifest(output_dir, faiss_dest, bm25_dest, args.compress)
    log.info("Export complete.")
    return 0


if __name__ == "__main__":
      sys.exit(main())
