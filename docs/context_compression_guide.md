# Context Compression Guide

The `ContextCompressor` class (`retrieval/context_compressor.py`) trims
the ranked list of retrieved chunks to a configurable token budget before
they are passed to the generation step. This prevents context-window
overflow and keeps generation latency predictable.

## Why compression matters

Large language models have a fixed context window. Passing more tokens than
the window allows causes an error; passing far fewer wastes retrieval quality.
The compressor sits between reranking and generation to enforce the budget
automatically.

## Configuration

All settings live in `config/context_config.yaml`:

| Key | Default | Description |
|-----|---------|-------------|
| `max_tokens` | `2048` | Hard ceiling on tokens sent to the LLM |
| `tokens_per_char` | `0.25` | Approximation factor (1 token ≈ 4 chars) |
| `strategy` | `greedy` | Selection strategy — see below |
| `min_partial_chars` | `50` | Minimum chars for a partial chunk to be included |

## Strategies

### greedy (default)

Chunks are processed in relevance rank order. Each chunk is included in full
until the budget is exhausted. The last chunk that exceeds the budget is
truncated to the remaining character allowance if it is larger than
`min_partial_chars`; otherwise it is dropped.

Best for: scenarios where the top-ranked chunks are most important and later
chunks are acceptable to lose.

### equal

The token budget is divided equally across all retrieved chunks. Each chunk
is truncated to its per-chunk character limit. No chunks are dropped.

Best for: scenarios where coverage across all chunks matters more than depth
in any single chunk (e.g. multi-document summarisation).

## Usage example

```python
from retrieval.context_compressor import ContextCompressor

compressor = ContextCompressor(max_tokens=2048, strategy="greedy")
result = compressor.compress(reranked_chunks)

print(f"Kept {len(result.chunks)} chunks, ~{result.total_tokens} tokens")
context = "\n\n".join(result.chunks)
```

## Tuning tips

- Increase `max_tokens` when using a model with a larger context window
  (e.g. 8 192 for Claude 3 Haiku or 200 000 for Claude 3 Opus).
- Lower `tokens_per_char` (e.g. `0.30`) for code-heavy or multilingual
  corpora where tokens are shorter on average.
- Switch to `equal` strategy when ragas context recall drops below 0.75 —
  it often indicates that relevant information is spread across many chunks.
- Monitor `CompressedContext.dropped` in production logs; a consistently
  high value suggests the retrieval pipeline is returning too many chunks
  relative to the budget.

## Integration with the pipeline

The compressor is called inside `retrieval/hybrid.py` after reranking:

```python
compressed = compressor.compress([r.text for r in reranked])
answer = generator.generate(query=query, context=compressed.chunks)
```
