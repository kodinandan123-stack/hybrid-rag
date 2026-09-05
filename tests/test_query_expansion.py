"""tests/test_query_expansion.py -- Unit tests for QueryExpander."""

import pytest
from unittest.mock import MagicMock, patch
from retrieval.query_expansion import QueryExpander, QueryExpansionConfig, ExpandedQuery


# ---------------------------------------------------------------------------
# ExpandedQuery
# ---------------------------------------------------------------------------

class TestExpandedQuery:
    def test_all_queries_includes_original(self):
        eq = ExpandedQuery(original="test query")
        assert eq.all_queries == ["test query"]

    def test_all_queries_deduplicates(self):
        eq = ExpandedQuery(original="foo", expansions=["bar", "foo", "baz", "bar"])
        result = eq.all_queries
        assert result[0] == "foo"
        assert result.count("foo") == 1
        assert result.count("bar") == 1

    def test_all_queries_preserves_order(self):
        eq = ExpandedQuery(original="q", expansions=["a", "b", "c"])
        assert eq.all_queries == ["q", "a", "b", "c"]


# ---------------------------------------------------------------------------
# QueryExpansionConfig defaults
# ---------------------------------------------------------------------------

class TestQueryExpansionConfig:
    def test_defaults(self):
        cfg = QueryExpansionConfig()
        assert cfg.enable_synonyms is True
        assert cfg.enable_hyponyms is False
        assert cfg.enable_llm_expansion is True
        assert cfg.max_expansions == 5
        assert cfg.deduplicate is True

    def test_custom_values(self):
        cfg = QueryExpansionConfig(max_expansions=3, llm_temperature=0.7)
        assert cfg.max_expansions == 3
        assert cfg.llm_temperature == 0.7


# ---------------------------------------------------------------------------
# QueryExpander.expand
# ---------------------------------------------------------------------------

class TestQueryExpander:
    def _make_expander(self, **cfg_kwargs):
        cfg = QueryExpansionConfig(**cfg_kwargs)
        return QueryExpander(config=cfg)

    def test_returns_expanded_query(self):
        expander = self._make_expander(
            enable_synonyms=False, enable_hyponyms=False, enable_llm_expansion=False
        )
        result = expander.expand("what is RAG?")
        assert isinstance(result, ExpandedQuery)
        assert result.original == "what is RAG?"

    def test_no_strategies_gives_empty_expansions(self):
        expander = self._make_expander(
            enable_synonyms=False, enable_hyponyms=False, enable_llm_expansion=False
        )
        result = expander.expand("hello world")
        assert result.expansions == []

    def test_max_expansions_respected(self):
        expander = self._make_expander(
            enable_synonyms=False, enable_hyponyms=False, enable_llm_expansion=False,
            max_expansions=2
        )
        expander._synonym_expansions = lambda q: ["a", "b", "c", "d"]
        expander.config.enable_synonyms = True
        result = expander.expand("test")
        assert len(result.expansions) <= 2

    def test_deduplication_applied(self):
        expander = self._make_expander(
            enable_synonyms=False, enable_hyponyms=False, enable_llm_expansion=False,
            max_expansions=10, deduplicate=True
        )
        expander._synonym_expansions = lambda q: ["dup", "dup", "unique"]
        expander.config.enable_synonyms = True
        result = expander.expand("test")
        assert result.expansions.count("dup") == 1

    def test_deduplication_disabled(self):
        expander = self._make_expander(
            enable_synonyms=False, enable_hyponyms=False, enable_llm_expansion=False,
            max_expansions=10, deduplicate=False
        )
        expander._synonym_expansions = lambda q: ["dup", "dup"]
        expander.config.enable_synonyms = True
        result = expander.expand("test")
        assert result.expansions.count("dup") == 2


# ---------------------------------------------------------------------------
# LLM expansion
# ---------------------------------------------------------------------------

class TestLLMExpansion:
    def test_llm_expansion_parses_lines(self):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "variant one\nvariant two\nvariant three"

        expander = QueryExpander(config=QueryExpansionConfig(
            enable_synonyms=False, enable_hyponyms=False, enable_llm_expansion=True
        ))
        with patch("retrieval.query_expansion.openai") as mock_openai:
            mock_openai.chat.completions.create.return_value = mock_response
            result = expander._llm_expansions("what is hybrid retrieval?")

        assert result == ["variant one", "variant two", "variant three"]

    def test_llm_expansion_returns_empty_on_error(self):
        expander = QueryExpander(config=QueryExpansionConfig(
            enable_synonyms=False, enable_hyponyms=False, enable_llm_expansion=True
        ))
        with patch("retrieval.query_expansion.openai", side_effect=ImportError):
            result = expander._llm_expansions("test")
        assert result == []
