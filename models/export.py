from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from models.audit import isoformat_utc, to_utc, utc_now


class ExportArtifact(BaseModel):
    """Metadata for one generated export artifact."""

    model_config = ConfigDict(extra="forbid")

    artifact: str = Field(min_length=1)
    format: str = Field(min_length=1)
    path: str = Field(min_length=1)
    exists: bool
    size_bytes: int | None = Field(default=None, ge=0)
    warning: str | None = None


class ExportReport(BaseModel):
    """Summary of a passive export run."""

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime = Field(default_factory=utc_now)
    warnings: list[str] = Field(default_factory=list)
    artifacts: list[ExportArtifact] = Field(default_factory=list)
    total_exports: int = Field(ge=0)
    successful_exports: int = Field(ge=0)
    failed_exports: int = Field(ge=0)

    @field_validator("generated_at")
    @classmethod
    def _generated_at_to_utc(cls, value: datetime) -> datetime:
        return to_utc(value)

    @field_serializer("generated_at")
    def _serialize_generated_at(self, value: datetime) -> str:
        return isoformat_utc(value)
