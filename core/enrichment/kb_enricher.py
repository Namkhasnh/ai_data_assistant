from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from core.enrichment.base_enricher import BaseEnricher
from core.matching.matcher import MatcherError, MatcherRegistry
from models.enrichment import EnrichmentConfigItem
from models.knowledge import KnowledgeEntry
from models.semantic_tag import SemanticDetectionReport


class KBEnricher(BaseEnricher):
    """Enrich rows from explicit source-column values and JSON knowledge files."""

    def __init__(self, matcher_registry: MatcherRegistry | None = None) -> None:
        super().__init__(name="knowledge")
        self.matcher_registry = matcher_registry or MatcherRegistry.default()

    def enrich(
        self,
        dataframe: pd.DataFrame,
        semantic_report: SemanticDetectionReport,
        enrichment_config: EnrichmentConfigItem,
    ) -> pd.DataFrame:
        result = self.copy_dataframe(dataframe)
        source_column = enrichment_config.source_column

        if source_column not in result.columns:
            self.warnings.append(
                f"Source column not found for enrichment {enrichment_config.enricher_id}: {source_column}"
            )
            return result

        conflicting_columns = [
            column
            for column in enrichment_config.output_columns
            if column == source_column or column in result.columns
        ]
        if conflicting_columns:
            self.warnings.append(
                "Enrichment output columns would overwrite existing columns "
                f"for {enrichment_config.enricher_id}: {conflicting_columns}"
            )
            return result

        knowledge = self._load_knowledge(enrichment_config)
        if knowledge is None:
            return result

        for output_column in enrichment_config.output_columns:
            result[output_column] = result[source_column].map(
                lambda value: self._lookup_output(value, knowledge, output_column)
            )
            self.enriched_columns.add(output_column)

        self.processed_columns.add(source_column)
        return result

    def _load_knowledge(
        self,
        enrichment_config: EnrichmentConfigItem,
    ) -> list[KnowledgeEntry] | None:
        knowledge_path = Path(enrichment_config.knowledge_file)
        try:
            text = knowledge_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            self.warnings.append(
                f"Knowledge file not found for enrichment {enrichment_config.enricher_id}: {knowledge_path}"
            )
            return None

        if not text:
            self.warnings.append(
                f"Knowledge file is empty for enrichment {enrichment_config.enricher_id}: {knowledge_path}"
            )
            return None

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            self.warnings.append(
                f"Knowledge file is invalid JSON for enrichment {enrichment_config.enricher_id}: {knowledge_path}"
            )
            return None

        if not isinstance(payload, dict):
            self.warnings.append(
                f"Knowledge file must contain an object for enrichment {enrichment_config.enricher_id}: {knowledge_path}"
            )
            return None

        knowledge: list[KnowledgeEntry] = []
        for source_value, attributes in payload.items():
            entry = self._knowledge_entry(source_value, attributes, enrichment_config)
            if entry is None:
                continue
            if not self._matcher_supported(entry, enrichment_config):
                continue
            knowledge.append(entry)

        return sorted(
            knowledge,
            key=lambda entry: (entry.priority, entry.canonical_value.casefold()),
        )

    def _knowledge_entry(
        self,
        source_value: Any,
        attributes: Any,
        enrichment_config: EnrichmentConfigItem,
    ) -> KnowledgeEntry | None:
        if not isinstance(attributes, dict):
            self.warnings.append(
                f"Knowledge entry must contain an object for enrichment {enrichment_config.enricher_id}: {source_value}"
            )
            return None

        if "outputs" in attributes:
            payload = {
                "canonical_value": str(source_value),
                "match_type": str(attributes.get("match_type", "exact")).strip().lower(),
                "aliases": attributes.get("aliases", []),
                "patterns": attributes.get("patterns", []),
                "outputs": attributes.get("outputs", {}),
                "priority": attributes.get("priority", 100),
            }
        else:
            payload = {
                "canonical_value": str(source_value),
                "match_type": "exact",
                "aliases": [],
                "patterns": [],
                "outputs": attributes,
                "priority": 100,
            }

        try:
            return KnowledgeEntry.model_validate(payload)
        except ValidationError as exc:
            self.warnings.append(
                f"Invalid knowledge entry for enrichment {enrichment_config.enricher_id}: {source_value}: {exc}"
            )
            return None

    def _matcher_supported(
        self,
        entry: KnowledgeEntry,
        enrichment_config: EnrichmentConfigItem,
    ) -> bool:
        try:
            self.matcher_registry.create(entry.match_type)
        except MatcherError:
            self.warnings.append(
                f"Unsupported match_type for enrichment {enrichment_config.enricher_id}: {entry.match_type}"
            )
            return False
        return True

    def _lookup_output(
        self,
        value: Any,
        knowledge: list[KnowledgeEntry],
        output_column: str,
    ) -> Any:
        matched_entry = self._match_entry(value, knowledge)
        if matched_entry is None:
            return pd.NA
        return matched_entry.outputs.get(output_column, pd.NA)

    def _match_entry(
        self,
        value: Any,
        knowledge: list[KnowledgeEntry],
    ) -> KnowledgeEntry | None:
        if pd.isna(value):
            return None

        for entry in knowledge:
            matcher = self.matcher_registry.create(entry.match_type)
            if matcher.matches(value, entry):
                return entry
        return None
