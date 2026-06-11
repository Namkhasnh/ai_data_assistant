from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from models.audit import isoformat_utc, to_utc, utc_now


class BusinessDatasetReport(BaseModel):
    """Summary of a deterministic business dataset generation run."""

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime = Field(default_factory=utc_now)
    total_columns: int = Field(ge=0)
    excluded_columns: list[str] = Field(default_factory=list)
    included_columns: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("generated_at")
    @classmethod
    def _generated_at_to_utc(cls, value: datetime) -> datetime:
        return to_utc(value)

    @field_serializer("generated_at")
    def _serialize_generated_at(self, value: datetime) -> str:
        return isoformat_utc(value)
