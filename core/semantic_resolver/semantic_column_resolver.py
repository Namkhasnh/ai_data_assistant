from __future__ import annotations

from collections.abc import Sequence

from models.semantic_column_registry import SemanticColumnRegistry


class SemanticColumnResolver:
    """Quiet resolver for semantic type to physical column relationships."""

    def __init__(self, registry: SemanticColumnRegistry | None = None) -> None:
        self.registry = registry or SemanticColumnRegistry()

    def get_column(self, semantic_type: str) -> str | None:
        """Return the first deterministic column for one semantic type."""

        columns = self.registry.columns_by_semantic_type.get(semantic_type, [])
        if not columns:
            return None
        return columns[0]

    def get_columns(self, semantic_types: Sequence[str]) -> list[str]:
        """Return sorted unique columns for one or more semantic types."""

        requested_types = sorted({semantic_type for semantic_type in semantic_types if semantic_type})
        columns: set[str] = set()
        for semantic_type in requested_types:
            columns.update(self.registry.columns_by_semantic_type.get(semantic_type, []))
        return sorted(columns)

    def has_semantic_type(self, semantic_type: str) -> bool:
        """Return whether the registry contains at least one column for a semantic type."""

        return bool(self.registry.columns_by_semantic_type.get(semantic_type))

    def available_semantic_types(self) -> list[str]:
        """Return available semantic types in deterministic order."""

        return sorted(self.registry.columns_by_semantic_type)
