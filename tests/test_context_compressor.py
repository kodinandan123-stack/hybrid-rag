"""test_context_compressor.py

Unit tests for retrieval.context_compressor.ContextCompressor.
"""

import pytest
from retrieval.context_compressor import CompressedContext, ContextCompressor


@pytest.fixture()
def short_chunks():
    return ["chunk one " * 10, "chunk two " * 10, "chunk three " * 10]


class TestContextCompressorInit:
    def test_default_params(self):
        cc = ContextCompressor()
        assert cc.max_tokens == 2048
        assert cc.strategy == "greedy"

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            ContextCompressor(strategy="random")


class TestGreedyStrategy:
    def test_all_chunks_fit(self):
        cc = ContextCompressor(max_tokens=4096)
        chunks = ["hello world"] * 5
        result = cc.compress(chunks)
        assert result.dropped == 0
        assert len(result.chunks) == 5

    def test_budget_exceeded_drops_chunks(self):
        cc = ContextCompressor(max_tokens=20)
        # Each chunk ~25 tokens (100 chars * 0.25)
        chunks = ["x" * 100] * 4
        result = cc.compress(chunks)
        assert result.dropped > 0
        assert result.total_tokens <= cc.max_tokens

    def test_partial_chunk_included_when_budget_allows(self):
        cc = ContextCompressor(max_tokens=50)
        # First chunk uses 10 tokens, second is large but partially fits
        chunks = ["a" * 40, "b" * 400]
        result = cc.compress(chunks)
        # Second chunk should be truncated, not dropped entirely
        assert len(result.chunks) == 2
        assert len(result.chunks[1]) < 400

    def test_empty_input(self):
        cc = ContextCompressor()
        result = cc.compress([])
        assert result.chunks == []
        assert result.total_tokens == 0
        assert result.dropped == 0


class TestEqualStrategy:
    def test_all_chunks_present(self, short_chunks):
        cc = ContextCompressor(max_tokens=2048, strategy="equal")
        result = cc.compress(short_chunks)
        assert len(result.chunks) == len(short_chunks)
        assert result.dropped == 0

    def test_chunks_truncated_to_budget(self):
        cc = ContextCompressor(max_tokens=40, strategy="equal")
        chunks = ["z" * 200, "y" * 200]
        result = cc.compress(chunks)
        # Each chunk gets max_tokens/2 = 20 tokens => 80 chars
        for chunk in result.chunks:
            assert len(chunk) <= 80

    def test_empty_input(self):
        cc = ContextCompressor(strategy="equal")
        result = cc.compress([])
        assert result == CompressedContext(chunks=[], total_tokens=0, dropped=0)


class TestReturnType:
    def test_returns_compressed_context(self):
        cc = ContextCompressor()
        result = cc.compress(["some text"])
        assert isinstance(result, CompressedContext)
        assert isinstance(result.chunks, list)
        assert isinstance(result.total_tokens, int)
        assert isinstance(result.dropped, int)
