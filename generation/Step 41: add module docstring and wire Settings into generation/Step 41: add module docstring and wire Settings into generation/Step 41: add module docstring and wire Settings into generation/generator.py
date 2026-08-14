"""
generator.py

Grounded answer generation using the Anthropic Messages API. Builds a
context block from retrieved chunks and prompts Claude to answer using
ONLY the provided context, citing sources in its response.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import anthropic

from config.settings import get_settings


SYSTEM_PROMPT = (
    "You are a precise assistant that answers questions using ONLY the provided "
    "context. If the answer is not contained in the context, say you don't know. "
    "Always cite the source of each claim using the provided chunk metadata."
)


class Generator:
    """Synthesizes grounded answers from a query and retrieved context chunks."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        settings = get_settings()
        self.model = model or settings.anthropic_model
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        blocks = []
        for i, chunk in enumerate(chunks):
            source = chunk.get("source", "unknown")
            blocks.append(f"[{i + 1}] (source: {source})\n{chunk['text']}")
        return "\n\n".join(blocks)

    def generate(self, query: str, chunks: List[Dict[str, Any]], max_tokens: int = 1024) -> str:
        """Generate a grounded answer for query using the provided context chunks."""
        context = self._format_context(chunks)
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text
