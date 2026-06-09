from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TopValue(BaseModel):
    """Frequency count for a non-null column value."""

    model_config = ConfigDict(extra="forbid")

    value: Any
    count: int = Field(ge=0)


class ColumnProfile(BaseModel):
    """Profiling metadata for one dataset column."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    data_type: str = Field(min_length=1)
    null_count: int = Field(ge=0)
    null_percentage: float = Field(ge=0.0, le=100.0)
    unique_value_count: int = Field(ge=0)
    top_values: list[TopValue] = Field(default_factory=list)
    sample_values: list[Any] = Field(default_factory=list)
