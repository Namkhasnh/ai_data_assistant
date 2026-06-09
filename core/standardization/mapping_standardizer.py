from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from core.standardization.base_standardizer import BaseStandardizer
from models.semantic_tag import SemanticDetectionReport


class MappingStandardizer(BaseStandardizer):
    """Map aliases to canonical values using external knowledge files."""

    def __init__(self) -> None:
        super().__init__(name="mapping")

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

            mapping = self._load_mapping(config)
            if not mapping:
                self.warnings.append(f"No mapping knowledge available for {semantic_type}")
                continue

            for column in self.configured_source_columns(config, semantic_type, semantic_columns):
                if column not in result.columns:
                    self.warnings.append(f"Column not found for mapping standardization: {column}")
                    continue

                target_column = self.configured_target_column(config, column)
                result[target_column] = result[column].map(
                    lambda value: self._map_value(value, mapping)
                )
                self.standardized_columns.add(target_column)
                self.processed_columns.add(column)

        return result

    def _load_mapping(self, config: dict[str, Any]) -> dict[str, str]:
        knowledge_file = config.get("knowledge_file")
        if not isinstance(knowledge_file, str):
            self.warnings.append("Mapping standardizer requires knowledge_file")
            return {}

        knowledge_path = Path(knowledge_file)
        try:
            text = knowledge_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            self.warnings.append(f"Knowledge file not found: {knowledge_path}")
            return {}

        if not text:
            self.warnings.append(f"Knowledge file is empty: {knowledge_path}")
            return {}

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            self.warnings.append(f"Knowledge file is invalid JSON: {knowledge_path}")
            return {}

        if not isinstance(payload, dict):
            self.warnings.append(f"Knowledge file must contain an object: {knowledge_path}")
            return {}

        mapping: dict[str, str] = {}
        for canonical_value, entry in payload.items():
            mapping[str(canonical_value).casefold()] = str(canonical_value)
            for alias in self._extract_aliases(entry):
                mapping[alias.casefold()] = str(canonical_value)
        return mapping

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
    def _map_value(value: Any, mapping: dict[str, str]) -> Any:
        if pd.isna(value):
            return value
        return mapping.get(str(value).casefold(), value)
