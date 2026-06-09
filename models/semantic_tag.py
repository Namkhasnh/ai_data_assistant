from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class SemanticTag(BaseModel):
    """Detected semantic meaning for one dataset column."""

    model_config = ConfigDict(extra="forbid")

    column_name: str = Field(min_length=1)
    semantic_type: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    detector_name: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)


class SemanticDetectionReport(BaseModel):
    """Semantic detection artifact for a profiled dataset."""

    model_config = ConfigDict(extra="forbid")

    source_file: str = Field(min_length=1)
    column_count: int = Field(ge=0)
    columns: list[SemanticTag] = Field(default_factory=list)
