# Response Cache Guide

The `ResponseCache` class (`generation/response_cache.py`) stores generated
answers keyed by a hash of the query and compressed context, preventing
redundant calls to the LLM API when identical or near-identical queries arrive
in quick succession.

## Why response caching matters

Each LLM API call incurs latency and cost.  In production workloads a
significant share of queries are repeated verbatim or produce the same top
retrieved chunks.  The response cache intercepts those requests before they
reach the generation step, returning a stored answer in sub-millisecond time.

## Configuration

All settings live in `config/response_cache_config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `max_size` | `256` | Maximum entries held in memory |
| `ttl` | `3600` | Entry lifetime in seconds (`0` = never expire) |
| `enabled` | `true` | Toggle the cache on or off |

## Cache key design

The cache key is a SHA-256 digest of the normalised query joined with a hash
of the context chunks.  Normalisation lower-cases and strips the query so
minor capitalisation differences share the same cache slot.  A query reused
with a different set of retrieved chunks misses the cache and generates a
fresh answer.

## Usage example

```python
from generation.response_cache import ResponseCache

cache = ResponseCache(max_size=256, ttl=3600)

cached = cache.get(query, compressed_context.chunks)
if cached is not None:
    return cached

answer = generator.generate(query=query, context=compressed_context.chunks)
cache.set(query, compressed_context.chunks, answer)
return answer
```

## Tuning tips

- Increase `max_size` for high-traffic deployments where memory permits.
- - Decrease `ttl` (e.g. `600`) when the document corpus is updated frequently.
  - - Set `ttl: 0` in a read-only deployment to keep popular answers indefinitely.
    - - Set `enabled: false` during ragas evaluation runs to ensure every query
      -   exercises the full pipeline and metrics reflect true generation quality.
     
      -   ## Monitoring
     
      -   Call `cache.stats()` to inspect runtime state:
     
      -   ```python
          stats = cache.stats()
          # {'size': 42, 'max_size': 256, 'ttl': 3600, 'total_hits': 187}
          ```

          A hit rate below 20 % on a production workload usually means `ttl` is too
          short or `max_size` is too small relative to the query distribution.

          ## Integration with the pipeline

          The cache sits between retrieval and generation inside `api/main.py`:

          ```python
          compressed = compressor.compress([r.text for r in reranked])
          answer = cache.get(query, compressed.chunks)
          if answer is None:
              answer = generator.generate(query=query, context=compressed.chunks)
              cache.set(query, compressed.chunks, answer)
          ```
