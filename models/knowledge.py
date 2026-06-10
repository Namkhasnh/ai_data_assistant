from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class KnowledgeEntry(BaseModel):
    """One normalized deterministic knowledge-base entry."""

    model_config = ConfigDict(extra="forbid")

    canonical_value: str = Field(min_length=1)
    match_type: str = "exact"
    aliases: list[str] = Field(default_factory=list)
    patterns: list[str] = Field(default_factory=list)
    outputs: dict[str, Any] = Field(default_factory=dict)
    priority: int = 100
