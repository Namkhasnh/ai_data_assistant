from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from models.audit import isoformat_utc, to_utc, utc_now


ValidationStatus = Literal["PASS", "PASS_WITH_WARNINGS", "FAIL"]


class CrossDomainValidationRecord(BaseModel):
    """Validation result for one representative domain pipeline run."""

    model_config = ConfigDict(extra="forbid")

    domain: str = Field(min_length=1)
    status: ValidationStatus
    warnings: list[str] = Field(default_factory=list)
    missing_knowledge: list[str] = Field(default_factory=list)
    generated_artifacts: list[str] = Field(default_factory=list)
    column_leakage_detected: bool = False
    unexpected_column_loss: bool = False


class CrossDomainValidationReport(BaseModel):
    """Cross-domain validation report for all representative domains."""

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime = Field(default_factory=utc_now)
    records: list[CrossDomainValidationRecord] = Field(default_factory=list)

    @field_validator("generated_at")
    @classmethod
    def _generated_at_to_utc(cls, value: datetime) -> datetime:
        return to_utc(value)

    @field_serializer("generated_at")
    def _serialize_generated_at(self, value: datetime) -> str:
        return isoformat_utc(value)
