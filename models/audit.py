from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


def to_utc(value: datetime) -> datetime:
    """Normalize a datetime to timezone-aware UTC."""

    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def isoformat_utc(value: datetime) -> str:
    """Serialize a datetime as ISO 8601 UTC with a Z suffix."""

    return to_utc(value).isoformat().replace("+00:00", "Z")


class AuditRecord(BaseModel):
    """One append-only module artifact event."""

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime = Field(default_factory=utc_now)
    module: str = Field(min_length=1)
    artifact: str = Field(min_length=1)
    status: str = Field(min_length=1)
    message: str = Field(min_length=1)

    @field_validator("timestamp")
    @classmethod
    def _timestamp_to_utc(cls, value: datetime) -> datetime:
        return to_utc(value)

    @field_serializer("timestamp")
    def _serialize_timestamp(self, value: datetime) -> str:
        return isoformat_utc(value)


class ArtifactRecord(BaseModel):
    """Filesystem metadata for a generated artifact."""

    model_config = ConfigDict(extra="forbid")

    artifact: str = Field(min_length=1)
    path: str = Field(min_length=1)
    module: str = Field(min_length=1)
    exists: bool
    content_hash: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    modified_at: datetime | None = None

    @field_validator("modified_at")
    @classmethod
    def _modified_at_to_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return to_utc(value)

    @field_serializer("modified_at")
    def _serialize_modified_at(self, value: datetime | None) -> str | None:
        if value is None:
            return None
        return isoformat_utc(value)


class ChangeLog(BaseModel):
    """Append-only audit records ordered by timestamp."""

    model_config = ConfigDict(extra="forbid")

    records: list[AuditRecord] = Field(default_factory=list)


class RuleHistoryRecord(BaseModel):
    """Aggregated rule history metadata."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(min_length=1)
    rule_type: str = Field(min_length=1)
    created_by: str = Field(min_length=1)
    enabled: bool
    execution_count: int = Field(ge=0)
    first_seen: datetime = Field(default_factory=utc_now)
    last_seen: datetime = Field(default_factory=utc_now)

    @field_validator("first_seen", "last_seen")
    @classmethod
    def _seen_to_utc(cls, value: datetime) -> datetime:
        return to_utc(value)

    @field_serializer("first_seen", "last_seen")
    def _serialize_seen(self, value: datetime) -> str:
        return isoformat_utc(value)


class RuleHistory(BaseModel):
    """Rule history records ordered by rule ID."""

    model_config = ConfigDict(extra="forbid")

    records: list[RuleHistoryRecord] = Field(default_factory=list)


class AuditReport(BaseModel):
    """Complete passive audit report assembled from existing outputs."""

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime = Field(default_factory=utc_now)
    warnings: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    change_log: ChangeLog = Field(default_factory=ChangeLog)
    rule_history: RuleHistory = Field(default_factory=RuleHistory)

    @field_validator("generated_at")
    @classmethod
    def _generated_at_to_utc(cls, value: datetime) -> datetime:
        return to_utc(value)

    @field_serializer("generated_at")
    def _serialize_generated_at(self, value: datetime) -> str:
        return isoformat_utc(value)
