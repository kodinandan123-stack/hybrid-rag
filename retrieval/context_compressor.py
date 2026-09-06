"""context_compressor.py

Selects and truncates retrieved chunks to fit within a token budget
before passing context to the generation step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class CompressedContext:
    chunks: List[str]
    total_tokens: int
    dropped: int


class ContextCompressor:
    """Trims a ranked list of chunks to a maximum token budget.

    Args:
        max_tokens: Hard upper bound on total tokens across all selected chunks.
        tokens_per_char: Approximation (default 0.25, ~4 chars per token).
        strategy: 'greedy' fills budget in rank order; 'equal' splits evenly.
    """

    def __init__(
        self,
        max_tokens: int = 2048,
        tokens_per_char: float = 0.25,
        strategy: str = "greedy",
    ) -> None:
        if strategy not in {"greedy", "equal"}:
            raise ValueError(f"Unknown strategy '{strategy}'. Use 'greedy' or 'equal'.")
        self.max_tokens = max_tokens
        self.tokens_per_char = tokens_per_char
        self.strategy = strategy

    def compress(self, chunks: List[str]) -> CompressedContext:
        if self.strategy == "greedy":
            return self._greedy(chunks)
        return self._equal(chunks)

    def _estimate_tokens(self, text: str) -> int:
        return max(1, round(len(text) * self.tokens_per_char))

    def _greedy(self, chunks: List[str]) -> CompressedContext:
        selected = []
        budget = self.max_tokens
        dropped = 0
        for chunk in chunks:
            tokens = self._estimate_tokens(chunk)
            if tokens <= budget:
                selected.append(chunk)
                budget -= tokens
            else:
                allowed_chars = int(budget / self.tokens_per_char)
                if allowed_chars > 50:
                    selected.append(chunk[:allowed_chars].rstrip())
                    budget = 0
                else:
                    dropped += 1
                if budget == 0:
                    dropped += len(chunks) - len(selected) - dropped
                    break
        total_used = self.max_tokens - budget
        logger.info("Compression: %d/%d chunks kept, ~%d tokens, %d dropped.",
                    len(selected), len(chunks), total_used, dropped)
        return CompressedContext(chunks=selected, total_tokens=total_used, dropped=dropped)

    def _equal(self, chunks: List[str]) -> CompressedContext:
        if not chunks:
            return CompressedContext(chunks=[], total_tokens=0, dropped=0)
        per_chunk = self.max_tokens // len(chunks)
        allowed_chars = int(per_chunk / self.tokens_per_char)
        selected = [c[:allowed_chars].rstrip() for c in chunks]
        total = sum(self._estimate_tokens(c) for c in selected)
        return CompressedContext(chunks=selected, total_tokens=total, dropped=0)
