from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SemanticColumnRegistry(BaseModel):
    """Normalized mapping from semantic types to physical column names."""

    model_config = ConfigDict(extra="forbid")

    columns_by_semantic_type: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("columns_by_semantic_type")
    @classmethod
    def _normalize_registry(
        cls,
        value: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        return {
            semantic_type: sorted({column for column in columns if column})
            for semantic_type, columns in sorted(value.items())
            if semantic_type
        }
