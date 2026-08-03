"""Answer generation grounded in retrieved context, using the Anthropic API."""

from typing import List, Dict, Any
import os
import anthropic


SYSTEM_PROMPT = (
    "You are a precise assistant that answers questions using ONLY the provided "
    "context. If the answer is not contained in the context, say you don't know. "
    "Always cite the source of each claim using the provided chunk metadata."
)

class Generator:
    """Synthesizes grounded answers from a query and retrieved context chunks."""

    def __init__(self, model: str = "claude-sonnet-4-5", api_key: str = None):
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        blocks = []
        for i, chunk in enumerate(chunks):
            source = chunk.get("source", "unknown")
            blocks.append(f"[{i + 1}] (source: {source})\n{chunk['text']}")
        return "\n\n".join(blocks)

    def generate(self, query: str, chunks: List[Dict[str, Any]], max_tokens: int = 1024) -> str:
        context = self._format_context(chunks)
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text
