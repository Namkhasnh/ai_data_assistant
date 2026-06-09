from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AISuggestion(BaseModel):
    """LLM-proposed cleaning or standardization suggestion pending review."""

    model_config = ConfigDict(extra="forbid")

    suggestion_id: str | None = None
    type: str = Field(min_length=1)
    column: str | None = None
    semantic_type: str | None = None
    source_value: str | None = None
    target_value: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    confidence: float
    reason: str = Field(min_length=1)
    created_by: str = "llm"


class AISuggestionReport(BaseModel):
    """Validated AI suggestions and audit-friendly validation metadata."""

    model_config = ConfigDict(extra="forbid")

    suggestions: list[AISuggestion] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provider: str | None = None
    model: str | None = None
    minimum_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    duplicate_suggestions_removed: int = Field(ge=0, default=0)
    rejected_suggestions_count: int = Field(ge=0, default=0)
