from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Rule(BaseModel):
    """Serializable deterministic rule definition."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    type: str = Field(min_length=1)
    column: str = Field(min_length=1)
    parameters: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    priority: int = 100
    description: str | None = None
    created_by: str = "manual"

    @field_validator("type")
    @classmethod
    def _normalize_type(cls, value: str) -> str:
        return value.strip().lower()


class RuleSet(BaseModel):
    """Collection of deterministic rules."""

    model_config = ConfigDict(extra="forbid")

    rules: list[Rule] = Field(default_factory=list)


class RuleExecutionResult(BaseModel):
    """Execution outcome for one rule."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(min_length=1)
    rule_type: str = Field(min_length=1)
    status: Literal["applied", "skipped", "failed"]
    affected_rows: int = Field(ge=0)
    execution_time_ms: float = Field(ge=0.0)
    message: str = ""


class ExecutionReport(BaseModel):
    """Execution report for a complete rule set run."""

    model_config = ConfigDict(extra="forbid")

    results: list[RuleExecutionResult] = Field(default_factory=list)
