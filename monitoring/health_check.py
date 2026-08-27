"""Health check endpoints and system status monitoring for hybrid-RAG pipeline."""

  import time
import psutil
import logging
from typing import Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


  @dataclass
class ComponentStatus:
    name: str
    healthy: bool
      latency_ms: float = 0.0
      details: Dict[str, Any] = field(default_factory=dict)
    error: str = ""


  class HealthChecker:
    """Aggregates health checks across all RAG pipeline components."""

      def __init__(self, qdrant_client=None, redis_client=None, ollama_base_url: str = "http://localhost:11434"):
        self.qdrant_client = qdrant_client
          self.redis_client = redis_client
          self.ollama_base_url = ollama_base_url

      def check_qdrant(self) -> ComponentStatus:
        start = time.monotonic()
          try:
            if self.qdrant_client is None:
                return ComponentStatus(name="qdrant", healthy=False, error="client not initialised")
                              info = self.qdrant_client.get_collections()
              latency = (time.monotonic() - start) * 1000
              return ComponentStatus(
                  name="qdrant",
                  healthy=True,
                  latency_ms=round(latency, 2),
                  details={"collections": len(info.collections)},
              )
          except Exception as exc:
            return ComponentStatus(name="qdrant", healthy=False, error=str(exc))

      def check_redis(self) -> ComponentStatus:
        start = time.monotonic()
          try:
            if self.redis_client is None:
                return ComponentStatus(name="redis", healthy=False, error="client not initialised")
                              self.redis_client.ping()
              latency = (time.monotonic() - start) * 1000
              info = self.redis_client.info("server")
              return ComponentStatus(
                  name="redis",
                  healthy=True,
                  latency_ms=round(latency, 2),
                  details={"version": info.get("redis_version", "unknown")},
              )
          except Exception as exc:
            return ComponentStatus(name="redis", healthy=False, error=str(exc))

      def check_ollama(self) -> ComponentStatus:
        import urllib.request
        start = time.monotonic()
          try:
            with urllib.request.urlopen(f"{self.ollama_base_url}/api/tags", timeout=5) as resp:
                latency = (time.monotonic() - start) * 1000
                                  return ComponentStatus(
                      name="ollama",
                      healthy=resp.status == 200,
                      latency_ms=round(latency, 2),
                  )
          except Exception as exc:
            return ComponentStatus(name="ollama", healthy=False, error=str(exc))

      def check_system(self) -> ComponentStatus:
        cpu = psutil.cpu_percent(interval=0.1)
          mem = psutil.virtual_memory()
          disk = psutil.disk_usage("/")
          healthy = cpu < 90 and mem.percent < 90 and disk.percent < 90
          return ComponentStatus(
              name="system",
              healthy=healthy,
              details={
                  "cpu_percent": cpu,
                  "memory_percent": mem.percent,
                  "disk_percent": disk.percent,
  },
          )

      def run_all(self) -> Dict[str, Any]:
          checks = [
              self.check_system(),
              self.check_qdrant(),
              self.check_redis(),
              self.check_ollama(),
          ]
          overall = all(c.healthy for c in checks)
          return {
            "status": "healthy" if overall else "degraded",
                          "timestamp": time.time(),
                          "components": [
              {
                                  "name": c.name,
                                  "healthy": c.healthy,
                                  "latency_ms": c.latency_ms,
                                  "details": c.details,
                                  "error": c.error,
              }
                              for c in checks
            ],
  }


  def get_health_checker(**kwargs) -> HealthChecker:
    return HealthChecker(**kwargs)
      
