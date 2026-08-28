"""scripts/run_cleanup.py

Remove stale cache entries and orphaned JSONL exports.
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.logging import setup_logging  # noqa: E402
from retrieval.cache import RetrievalCache  # noqa: E402

logger = logging.getLogger(__name__)


def clean_cache() -> None:
    cache = RetrievalCache()
    size = cache.size()
    cache.clear()
    logger.info("Cache cleared: %d entries removed.", size)


def clean_stale_exports(export_dir: Path, keep: int = 5) -> None:
    files = sorted(export_dir.glob("*.jsonl"), key=lambda f: f.stat().st_mtime, reverse=True)
    for f in files[keep:]:
        f.unlink()
        logger.info("Deleted: %s", f.name)
    logger.info("Kept %d, deleted %d.", min(len(files), keep), max(0, len(files) - keep))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean up stale caches and exports.")
    p.add_argument("--export-dir", type=Path, default=PROJECT_ROOT / "exports")
    p.add_argument("--keep-exports", type=int, default=5)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(level=args.log_level)
    logger.info("Cleanup started.")
    clean_cache()
    if args.export_dir.exists():
        clean_stale_exports(args.export_dir, keep=args.keep_exports)
    else:
        logger.warning("Export dir not found: %s", args.export_dir)
    logger.info("Cleanup complete.")


if __name__ == "__main__":
    main()
