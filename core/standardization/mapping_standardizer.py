from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from core.matching.matcher import MatcherError, MatcherRegistry
from core.standardization.base_standardizer import BaseStandardizer
from models.knowledge import KnowledgeEntry
from models.semantic_tag import SemanticDetectionReport


class MappingStandardizer(BaseStandardizer):
    """Map aliases to canonical values using external knowledge files."""

    def __init__(self, matcher_registry: MatcherRegistry | None = None) -> None:
        super().__init__(name="mapping")
        self.matcher_registry = matcher_registry or MatcherRegistry.default()

    def standardize(
        self,
        dataframe: pd.DataFrame,
        semantic_report: SemanticDetectionReport,
        standardization_config: dict[str, Any],
    ) -> pd.DataFrame:
        result = self.copy_dataframe(dataframe)
        semantic_columns = self.semantic_columns_by_type(semantic_report)

        for semantic_type, config in standardization_config.items():
            if config.get("standardizer") != self.name:
                continue

            knowledge = self._load_mapping(config)
            if not knowledge:
                self.warnings.append(f"No mapping knowledge available for {semantic_type}")
                continue

            for column in self.configured_source_columns(config, semantic_type, semantic_columns):
                if column not in result.columns:
                    self.warnings.append(f"Column not found for mapping standardization: {column}")
                    continue

                target_column = self._target_column(config, column)
                if target_column == column or target_column in result.columns:
                    self.warnings.append(
                        f"Mapping standardization would overwrite existing column: {target_column}"
                    )
                    continue

                result[target_column] = result[column].map(
                    lambda value: self._canonicalize(value, knowledge)
                )
                self.standardized_columns.add(target_column)
                self.processed_columns.add(column)

        return result

    def _load_mapping(self, config: dict[str, Any]) -> list[KnowledgeEntry]:
        knowledge_file = config.get("knowledge_file")
        if not isinstance(knowledge_file, str):
            self.warnings.append("Mapping standardizer requires knowledge_file")
            return []

        knowledge_path = Path(knowledge_file)
        try:
            text = knowledge_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            self.warnings.append(f"Knowledge file not found: {knowledge_path}")
            return []

        if not text:
            self.warnings.append(f"Knowledge file is empty: {knowledge_path}")
            return []

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            self.warnings.append(f"Knowledge file is invalid JSON: {knowledge_path}")
            return []

        if not isinstance(payload, dict):
            self.warnings.append(f"Knowledge file must contain an object: {knowledge_path}")
            return []

        knowledge: list[KnowledgeEntry] = []
        for canonical_key, entry in payload.items():
            knowledge_entry = self._knowledge_entry(canonical_key, entry)
            if knowledge_entry is None:
                continue
            if not self._matcher_supported(knowledge_entry):
                continue
            knowledge.append(knowledge_entry)

        return sorted(
            knowledge,
            key=lambda item: (item.priority, item.canonical_value.casefold()),
        )

    def _knowledge_entry(self, canonical_key: Any, entry: Any) -> KnowledgeEntry | None:
        if isinstance(entry, dict):
            aliases = self._extract_aliases(entry)
            has_new_format = any(
                field in entry
                for field in ("canonical_value", "match_type", "patterns", "priority")
            )
            match_type = entry.get("match_type")
            if not isinstance(match_type, str):
                match_type = "alias" if aliases else "exact"
            payload = {
                "canonical_value": str(entry.get("canonical_value", canonical_key)),
                "match_type": match_type.strip().lower(),
                "aliases": aliases,
                "patterns": entry.get("patterns", []),
                "outputs": {},
                "priority": entry.get("priority", 100),
            }
            if not has_new_format and aliases:
                payload["match_type"] = "alias"
        else:
            payload = {
                "canonical_value": str(canonical_key),
                "match_type": "alias",
                "aliases": self._extract_aliases(entry),
                "patterns": [],
                "outputs": {},
                "priority": 100,
            }

        try:
            return KnowledgeEntry.model_validate(payload)
        except ValidationError as exc:
            self.warnings.append(f"Invalid mapping knowledge entry: {canonical_key}: {exc}")
            return None

    def _matcher_supported(self, entry: KnowledgeEntry) -> bool:
        try:
            self.matcher_registry.create(entry.match_type)
        except MatcherError:
            self.warnings.append(f"Unsupported mapping match_type: {entry.match_type}")
            return False
        return True

    def _canonicalize(self, value: Any, knowledge: list[KnowledgeEntry]) -> Any:
        if pd.isna(value):
            return pd.NA

        for entry in knowledge:
            matcher = self.matcher_registry.create(entry.match_type)
            if matcher.matches(value, entry):
                return entry.canonical_value
        return pd.NA

    @staticmethod
    def _extract_aliases(entry: Any) -> list[str]:
        if isinstance(entry, dict):
            aliases = entry.get("aliases", [])
            if isinstance(aliases, list):
                return [str(alias) for alias in aliases]
        if isinstance(entry, list):
            return [str(alias) for alias in entry]
        return []

    @staticmethod
    def _target_column(config: dict[str, Any], source_column: str) -> str:
        output_column = config.get("output_column")
        if isinstance(output_column, str) and output_column:
            return output_column
        return f"standardized_{source_column}"
