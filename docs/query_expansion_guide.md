# Query Expansion Guide

Query expansion improves retrieval recall by generating alternative phrasings of a user query before searching the vector and keyword indexes.

## Overview

The `QueryExpander` class in `retrieval/query_expansion.py` supports three complementary strategies:

| Strategy | Module | Notes |
|----------|--------|-------|
| Synonym substitution | WordNet (NLTK) | Replaces tokens with WordNet synonyms |
| Hyponym expansion | WordNet (NLTK) | Adds more specific sub-terms |
| LLM expansion | OpenAI chat completions | Generates semantically varied rewrites |

All strategies are independently toggleable via `QueryExpansionConfig` or `config/query_expansion_config.yaml`.

## Quick Start

```python
from retrieval.query_expansion import QueryExpander, QueryExpansionConfig

cfg = QueryExpansionConfig(enable_synonyms=True, enable_llm_expansion=True, max_expansions=5)
expander = QueryExpander(config=cfg)

expanded = expander.expand("what is hybrid retrieval?")
print(expanded.all_queries)
# ['what is hybrid retrieval?', 'what is combined retrieval?', ...]
```

## Configuration Reference

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enable_synonyms` | bool | `true` | Enable WordNet synonym substitution |
| `enable_hyponyms` | bool | `false` | Enable WordNet hyponym expansion |
| `enable_llm_expansion` | bool | `true` | Enable LLM-based rewrite generation |
| `max_expansions` | int | `5` | Max expanded queries (excluding original) |
| `llm_temperature` | float | `0.3` | LLM sampling temperature |
| `deduplicate` | bool | `true` | Remove duplicate expansions |
| `llm_model` | str | `gpt-4o-mini` | OpenAI model for expansion |
| `llm_max_tokens` | int | `256` | Max tokens for LLM response |

## Integration with Hybrid Retrieval

Pass all queries from `ExpandedQuery.all_queries` to both the dense and sparse retrievers, then merge results with your `HybridScorer`:

```python
from retrieval.query_expansion import QueryExpander
from retrieval.hybrid import HybridRetriever

expander = QueryExpander()
hybrid = HybridRetriever()

expanded = expander.expand(user_query)
results = []
for q in expanded.all_queries:
    results.extend(hybrid.retrieve(q, top_k=10))

# Deduplicate and re-rank the merged results
```

## Performance Considerations

- **Synonym expansion** is CPU-bound and fast; first call downloads the WordNet corpus (~3 MB).
- **Hyponym expansion** can produce many variants; keep `max_expansions` low when enabled.
- **LLM expansion** adds latency (~200–800 ms per call). Cache results with `RerankCache` for repeated queries.
- Set `enable_llm_expansion: false` in latency-sensitive paths and rely on synonym expansion only.

## Tuning Tips

- Start with `max_expansions: 3` and increase only if recall metrics improve.
- Use a low `llm_temperature` (0.1–0.3) for more focused expansions.
- Monitor token spend: each LLM expansion call costs ~50–150 input tokens + completion.
- Combine with the `RerankCache` (see `retrieval/rerank_cache.py`) to avoid redundant LLM calls for frequently repeated queries.
