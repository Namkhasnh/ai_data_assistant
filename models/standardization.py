from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StandardizationReport(BaseModel):
    """Report for deterministic standardization runs."""

    model_config = ConfigDict(extra="forbid")

    total_standardized_columns: int = Field(ge=0)
    standardized_by_standardizer: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    skipped_columns: list[str] = Field(default_factory=list)
    standardizer_warnings: dict[str, list[str]] = Field(default_factory=dict)
