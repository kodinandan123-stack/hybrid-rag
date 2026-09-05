"""retrieval/query_expansion.py -- Multi-strategy query expansion for hybrid RAG."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QueryExpansionConfig:
    enable_synonyms: bool = True
    enable_hyponyms: bool = False
    enable_llm_expansion: bool = True
    max_expansions: int = 5
    llm_temperature: float = 0.3
    deduplicate: bool = True


@dataclass
class ExpandedQuery:
    original: str
    expansions: List[str] = field(default_factory=list)

    @property
    def all_queries(self) -> List[str]:
        seen = {self.original}
        result = [self.original]
        for q in self.expansions:
            if q not in seen:
                seen.add(q)
                result.append(q)
        return result


class QueryExpander:
    def __init__(self, config=None):
        self.config = config or QueryExpansionConfig()

    def expand(self, query: str) -> ExpandedQuery:
        result = ExpandedQuery(original=query)
        if self.config.enable_synonyms:
            result.expansions.extend(self._synonym_expansions(query))
        if self.config.enable_hyponyms:
            result.expansions.extend(self._hyponym_expansions(query))
        if self.config.enable_llm_expansion:
            result.expansions.extend(self._llm_expansions(query))
        if self.config.deduplicate:
            result.expansions = list(dict.fromkeys(result.expansions))
        result.expansions = result.expansions[: self.config.max_expansions]
        logger.debug("Expanded %r -> %d variants", query, len(result.expansions))
        return result

    def _synonym_expansions(self, query: str) -> List[str]:
        try:
            import nltk
            from nltk.corpus import wordnet
            nltk.download("wordnet", quiet=True)
            tokens = query.split()
            variants = []
            for i, token in enumerate(tokens):
                for syn in wordnet.synsets(token):
                    for lemma in syn.lemmas():
                        candidate = lemma.name().replace("_", " ")
                        if candidate.lower() != token.lower():
                            new_tokens = tokens[:i] + [candidate] + tokens[i + 1:]
                            variants.append(" ".join(new_tokens))
            return variants
        except Exception as exc:
            logger.warning("Synonym expansion failed: %s", exc)
            return []

    def _hyponym_expansions(self, query: str) -> List[str]:
        try:
            import nltk
            from nltk.corpus import wordnet
            nltk.download("wordnet", quiet=True)
            tokens = query.split()
            variants = []
            for i, token in enumerate(tokens):
                for syn in wordnet.synsets(token):
                    for hypo in syn.hyponyms():
                        for lemma in hypo.lemmas():
                            candidate = lemma.name().replace("_", " ")
                            new_tokens = tokens[:i] + [candidate] + tokens[i + 1:]
                            variants.append(" ".join(new_tokens))
            return variants
        except Exception as exc:
            logger.warning("Hyponym expansion failed: %s", exc)
            return []

    def _llm_expansions(self, query: str) -> List[str]:
        try:
            import openai
            prompt = (
                f"Generate {self.config.max_expansions} alternative search queries "
                f"for the following question. Return one per line, no numbering.\n\nQuestion: {query}"
            )
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.llm_temperature,
                max_tokens=256,
            )
            raw = response.choices[0].message.content or ""
            return [line.strip() for line in raw.splitlines() if line.strip()]
        except Exception as exc:
            logger.warning("LLM expansion failed: %s", exc)
            return []
