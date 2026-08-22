"""api/middleware.py

FastAPI middleware for request logging, timing, and error handling.
"""
from __future__ import annotations

import time
import uuid
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
      """Log each request with a unique ID, method, path, status, and latency."""

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
              request_id = str(uuid.uuid4())[:8]
              start = time.perf_counter()

        logger.info(
                      "request_start",
                      extra={
                                        "request_id": request_id,
                                        "method": request.method,
                                        "path": request.url.path,
                      },
        )

        try:
                      response: Response = await call_next(request)
except Exception as exc:  # noqa: BLE001
              logger.exception(
                                "request_error",
                                extra={"request_id": request_id, "error": str(exc)},
              )
              raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"

        logger.info(
                      "request_end",
                      extra={
                                        "request_id": request_id,
                                        "status_code": response.status_code,
                                        "elapsed_ms": round(elapsed_ms, 1),
                      },
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
      """Simple in-memory rate limiter: max N requests per window per client IP."""

    def __init__(self, app, max_requests: int = 60, window_seconds: int = 60) -> None:
              super().__init__(app)
              self.max_requests = max_requests
              self.window_seconds = window_seconds
              self._buckets: dict[str, list[float]] = {}

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
              client_ip = request.client.host if request.client else "unknown"
              now = time.time()
              window_start = now - self.window_seconds

        hits = self._buckets.get(client_ip, [])
        hits = [t for t in hits if t > window_start]
        hits.append(now)
        self._buckets[client_ip] = hits

        if len(hits) > self.max_requests:
                      from starlette.responses import JSONResponse

            return JSONResponse(
                              status_code=429,
                              content={"detail": "Rate limit exceeded. Please slow down."},
            )

        return await call_next(request)
