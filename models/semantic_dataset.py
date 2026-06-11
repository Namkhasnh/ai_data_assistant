from __future__ import annotations

from datetime import datetime

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from models.audit import isoformat_utc, to_utc


class SemanticDatasetReport(BaseModel):
    """Metadata for one persisted semantic dataset artifact."""

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    source_column_count: int = Field(ge=0)
    semantic_column_count: int = Field(ge=0)
    total_columns: int = Field(ge=0)
    source_columns: list[str] = Field(default_factory=list)
    semantic_columns: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("generated_at")
    @classmethod
    def _generated_at_to_utc(cls, value: datetime) -> datetime:
        return to_utc(value)

    @field_serializer("generated_at")
    def _serialize_generated_at(self, value: datetime) -> str:
        return isoformat_utc(value)


class SemanticDataset(BaseModel):
    """Semantic dataframe plus its persistence report."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    dataframe: pd.DataFrame
    report: SemanticDatasetReport
