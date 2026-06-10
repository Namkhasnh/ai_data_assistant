from __future__ import annotations

import pandas as pd

from core.enrichment.base_enricher import EnrichmentError
from core.enrichment.enricher_registry import EnricherRegistry
from models.enrichment import EnrichmentConfig, EnrichmentConfigItem, EnrichmentReport
from models.semantic_tag import SemanticDetectionReport


class EnrichmentEngine:
    """Sequential deterministic enrichment engine."""

    def __init__(self, registry: EnricherRegistry | None = None) -> None:
        self.registry = registry or EnricherRegistry.default()

    def enrich(
        self,
        dataframe: pd.DataFrame,
        semantic_report: SemanticDetectionReport,
        enrichment_config: EnrichmentConfig,
    ) -> tuple[pd.DataFrame, EnrichmentReport]:
        """Apply enabled enrichment items in priority order."""

        result = dataframe.copy(deep=True)
        enriched_by_enricher: dict[str, int] = {}
        enricher_warnings: dict[str, list[str]] = {}
        warnings: list[str] = []
        enriched_columns: set[str] = set()
        processed_columns: set[str] = set()

        for config_item in self._ordered_items(enrichment_config):
            if not config_item.enabled:
                warnings.append(f"Enrichment disabled: {config_item.enricher_id}")
                continue

            try:
                enricher = self.registry.create(config_item.enricher)
            except EnrichmentError as exc:
                warnings.append(f"{config_item.enricher_id}: {exc}")
                continue

            result = enricher.enrich(
                dataframe=result,
                semantic_report=semantic_report,
                enrichment_config=config_item,
            )
            enriched_by_enricher[enricher.name] = (
                enriched_by_enricher.get(enricher.name, 0)
                + len(enricher.enriched_columns)
            )
            enriched_columns.update(enricher.enriched_columns)
            processed_columns.update(enricher.processed_columns)
            if enricher.warnings:
                enricher_warnings[config_item.enricher_id] = enricher.warnings
                warnings.extend(enricher.warnings)

        skipped_columns = [
            tag.column_name
            for tag in semantic_report.columns
            if tag.column_name not in processed_columns
        ]

        report = EnrichmentReport(
            total_enriched_columns=len(enriched_columns),
            enriched_by_enricher=enriched_by_enricher,
            warnings=warnings,
            skipped_columns=skipped_columns,
            enricher_warnings=enricher_warnings,
        )
        return result, report

    @staticmethod
    def _ordered_items(enrichment_config: EnrichmentConfig) -> list[EnrichmentConfigItem]:
        return sorted(
            enrichment_config.enrichments,
            key=lambda item: (item.priority, item.enricher_id),
        )
