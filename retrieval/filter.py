"""Metadata filtering for retrieval results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MetadataFilter:
    """A single key-value metadata filter."""

        key: str
            value: Any
                operator: str = "eq"  # eq | ne | in | not_in

                    def matches(self, metadata: dict[str, Any]) -> bool:
                            """Return True if *metadata* satisfies this filter."""
                                    actual = metadata.get(self.key)
                                            if self.operator == "eq":
                                                        return actual == self.value
                                                                if self.operator == "ne":
                                                                            return actual != self.value
                                                                                    if self.operator == "in":
                                                                                                return actual in self.value
                                                                                                        if self.operator == "not_in":
                                                                                                                    return actual not in self.value
                                                                                                                            raise ValueError(f"Unknown operator: {self.operator}")
                                                                                                                            
                                                                                                                            
                                                                                                                            def apply_filters(
                                                                                                                                chunks: list[dict[str, Any]],
                                                                                                                                    filters: list[MetadataFilter],
                                                                                                                                    ) -> list[dict[str, Any]]:
                                                                                                                                        """Return only chunks whose metadata satisfies all *filters*.
                                                                                                                                        
                                                                                                                                            Args:
                                                                                                                                                    chunks: List of chunk dicts, each containing a ``metadata`` key.
                                                                                                                                                            filters: Filters to apply (AND semantics).
                                                                                                                                                            
                                                                                                                                                                Returns:
                                                                                                                                                                        Filtered list of chunks.
                                                                                                                                                                            """
                                                                                                                                                                                if not filters:
                                                                                                                                                                                        return chunks
                                                                                                                                                                                            return [
                                                                                                                                                                                                    c
                                                                                                                                                                                                            for c in chunks
                                                                                                                                                                                                                    if all(f.matches(c.get("metadata", {})) for f in filters)
                                                                                                                                                                                                                        ]
