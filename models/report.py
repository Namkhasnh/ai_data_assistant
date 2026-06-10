from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from models.audit import isoformat_utc, to_utc, utc_now


class ReportSection(BaseModel):
    """A structured section rendered by the HTML report templates."""

    model_config = ConfigDict(extra="forbid")

    section_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    summary: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class ReportChart(BaseModel):
    """Metadata for one external chart asset."""

    model_config = ConfigDict(extra="forbid")

    chart_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    chart_type: str = Field(min_length=1)
    path: str = Field(min_length=1)
    generated_at: datetime = Field(default_factory=utc_now)

    @field_validator("generated_at")
    @classmethod
    def _generated_at_to_utc(cls, value: datetime) -> datetime:
        return to_utc(value)

    @field_serializer("generated_at")
    def _serialize_generated_at(self, value: datetime) -> str:
        return isoformat_utc(value)


class ReportMetadata(BaseModel):
    """Top-level metadata for a generated pipeline HTML report."""

    model_config = ConfigDict(extra="forbid")

    title: str = "Pipeline Report"
    generated_at: datetime = Field(default_factory=utc_now)
    source_file: str | None = None

    @field_validator("generated_at")
    @classmethod
    def _generated_at_to_utc(cls, value: datetime) -> datetime:
        return to_utc(value)

    @field_serializer("generated_at")
    def _serialize_generated_at(self, value: datetime) -> str:
        return isoformat_utc(value)


class PipelineReport(BaseModel):
    """Serializable report context independent of external chart assets."""

    model_config = ConfigDict(extra="forbid")

    metadata: ReportMetadata = Field(default_factory=ReportMetadata)
    sections: list[ReportSection] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
