from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RuleGenerationReport(BaseModel):
    """Report for deterministic rule generation."""

    model_config = ConfigDict(extra="forbid")

    total_generated_rules: int = Field(ge=0)
    generated_by_generator: dict[str, int] = Field(default_factory=dict)
    skipped_columns: list[str] = Field(default_factory=list)
    duplicate_rules_removed: int = Field(ge=0, default=0)
    warnings: list[str] = Field(default_factory=list)
    generator_warnings: dict[str, list[str]] = Field(default_factory=dict)
