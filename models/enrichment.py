from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from models.knowledge import KnowledgeEntry


class EnrichmentConfigItem(BaseModel):
    """One deterministic enrichment configuration item."""

    model_config = ConfigDict(extra="forbid")

    enricher_id: str = Field(min_length=1)
    enricher: str = Field(min_length=1)
    source_column: str = Field(min_length=1)
    knowledge_file: str = Field(min_length=1)
    output_columns: list[str] = Field(min_length=1)
    enabled: bool = True
    priority: int = 100


class EnrichmentConfig(BaseModel):
    """Config container for deterministic enrichment items."""

    model_config = ConfigDict(extra="forbid")

    enrichments: list[EnrichmentConfigItem] = Field(default_factory=list)


class EnrichmentReport(BaseModel):
    """Report for deterministic knowledge-based enrichment runs."""

    model_config = ConfigDict(extra="forbid")

    total_enriched_columns: int = Field(ge=0)
    enriched_by_enricher: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    skipped_columns: list[str] = Field(default_factory=list)
    enricher_warnings: dict[str, list[str]] = Field(default_factory=dict)
