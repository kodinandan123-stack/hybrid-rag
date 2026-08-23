"""Health check endpoint for the Hybrid RAG API."""

from fastapi import APIRouter
from datetime import datetime, timezone
import platform
import psutil

router = APIRouter()


@router.get("/health", tags=["health"])
def health_check() -> dict:
      """Return service liveness status."""
      return {
          "status": "ok",
          "timestamp": datetime.now(timezone.utc).isoformat(),
      }


@router.get("/health/ready", tags=["health"])
def readiness_check() -> dict:
      """Return service readiness with system resource info."""
      mem = psutil.virtual_memory()
      disk = psutil.disk_usage("/")
      return {
          "status": "ready",
          "timestamp": datetime.now(timezone.utc).isoformat(),
          "system": {
              "python": platform.python_version(),
              "os": platform.system(),
              "cpu_count": psutil.cpu_count(),
              "memory_used_pct": round(mem.percent, 1),
              "disk_used_pct": round(disk.percent, 1),
          },
      }
  
