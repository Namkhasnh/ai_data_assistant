from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.rule_generation.base_generator import BaseRuleGenerator
from models.column_profile import ColumnProfile
from models.dataset import DatasetMetadata
from models.rule import Rule
from models.semantic_tag import SemanticDetectionReport


class MappingRuleGenerator(BaseRuleGenerator):
    """Suggest mapping rules only when deterministic knowledge aliases exist."""

    KNOWLEDGE_FILES = {
        "LOCATION": "cities.json",
        "JOB_TITLE": "job_titles.json",
    }

    def __init__(self, knowledge_dir: str | Path = "knowledge") -> None:
        super().__init__(name="mapping")
        self.knowledge_dir = Path(knowledge_dir)

    def generate(
        self,
        metadata: DatasetMetadata,
        semantic_report: SemanticDetectionReport,
    ) -> list[Rule]:
        semantic_by_column = self.semantic_by_column(semantic_report)
        rules: list[Rule] = []

        for column in metadata.columns:
            semantic_tag = semantic_by_column.get(column.name)
            if semantic_tag is None:
                continue

            knowledge_file = self.KNOWLEDGE_FILES.get(semantic_tag.semantic_type)
            if knowledge_file is None:
                continue

            knowledge_mapping = self._load_alias_mapping(knowledge_file)
            if not knowledge_mapping:
                self.warnings.append(
                    f"No deterministic mapping knowledge available for {semantic_tag.semantic_type}"
                )
                continue

            observed_mapping = self._observed_mapping(column, knowledge_mapping)
            if not observed_mapping:
                continue

            rules.append(
                Rule(
                    id=f"rule_mapping_{column.name}_001",
                    type="mapping",
                    column=column.name,
                    parameters={
                        "mapping": observed_mapping,
                        "case_sensitive": False,
                    },
                    enabled=True,
                    priority=20,
                    description=f"Map known aliases in {column.name} to canonical values.",
                    created_by="semantic",
                )
            )

        return rules

    def _load_alias_mapping(self, filename: str) -> dict[str, str]:
        knowledge_path = self.knowledge_dir / filename
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

        alias_mapping: dict[str, str] = {}
        for canonical_value, entry in payload.items():
            aliases = self._extract_aliases(entry)
            for alias in aliases:
                alias_mapping[alias.casefold()] = str(canonical_value)
        return alias_mapping

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
    def _observed_mapping(
        column: ColumnProfile,
        knowledge_mapping: dict[str, str],
    ) -> dict[str, str]:
        observed_values = [
            *[top_value.value for top_value in column.top_values],
            *column.sample_values,
        ]
        result: dict[str, str] = {}
        for value in observed_values:
            value_as_string = str(value)
            canonical_value = knowledge_mapping.get(value_as_string.casefold())
            if canonical_value is not None and value_as_string != canonical_value:
                result[value_as_string] = canonical_value
        return dict(sorted(result.items()))
