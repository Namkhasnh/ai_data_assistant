from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class KnowledgePackageMetadata(BaseModel):
    """Public metadata exposed by one optional knowledge package."""

    model_config = ConfigDict(extra="forbid")

    package_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = ""
    version: str = "1.0"
    enabled: bool = True
    priority: int = 100
    warnings: list[str] = Field(default_factory=list)
    required_columns: list[str] = Field(default_factory=list)
    produced_columns: list[str] = Field(default_factory=list)


class KnowledgePackageReport(BaseModel):
    """Summary of one deterministic knowledge package engine run."""

    model_config = ConfigDict(extra="forbid")

    applied_packages: list[str] = Field(default_factory=list)
    skipped_packages: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    produced_columns: list[str] = Field(default_factory=list)
    produced_columns_by_package: dict[str, list[str]] = Field(default_factory=dict)
    unknown_values_by_package: dict[str, list[str]] = Field(default_factory=dict)


class KnowledgePackageResult(BaseModel):
    """Dataframe plus report returned by the knowledge package engine."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    dataframe: pd.DataFrame
    report: KnowledgePackageReport
