"""
logging.py

Centralized logging configuration for the hybrid RAG pipeline.

Call ``configure_logging()`` once at process start (e.g. in api/main.py or
any CLI entry-point) to apply a consistent format across all modules.  All
pipeline modules obtain their logger via the standard ``logging.getLogger``
idiom — this module only sets up handlers and formatters; it does not own
any loggers itself.

Usage
-----
    from config.logging import configure_logging
    configure_logging()          # INFO to stdout, default format
    configure_logging(level="DEBUG", json_format=True)  # structured JSON
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Optional


_DEFAULT_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%dT%H:%M:%S"


class _JsonFormatter(logging.Formatter):
    """Emit each log record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003
        payload = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(
    level: str = "INFO",
    json_format: bool = False,
    datefmt: Optional[str] = None,
) -> None:
    """Configure the root logger with a stream handler writing to stdout.

    Parameters
    ----------
    level:
        Logging level string accepted by ``logging.getLevelName``
        (e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``).
    json_format:
        When ``True`` each record is emitted as a JSON object suitable for
        log-aggregation systems (Datadog, Loki, CloudWatch, ...).  When
        ``False`` a human-readable text format is used.
    datefmt:
        ``strftime``-compatible date format string.  Defaults to ISO-8601
        without the timezone suffix.
    """
    numeric_level = logging.getLevelName(level.upper())
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unknown log level: {level!r}")

    datefmt = datefmt or _DEFAULT_DATEFMT

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        formatter: logging.Formatter = _JsonFormatter(datefmt=datefmt)
    else:
        formatter = logging.Formatter(fmt=_DEFAULT_FMT, datefmt=datefmt)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Remove pre-existing handlers so re-calling configure_logging is idempotent.
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(numeric_level)
