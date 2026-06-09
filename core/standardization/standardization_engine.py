from __future__ import annotations

from typing import Any

import pandas as pd

from core.standardization.standardizer_registry import StandardizerRegistry
from models.semantic_tag import SemanticDetectionReport
from models.standardization import StandardizationReport


class StandardizationEngine:
    """Sequential deterministic standardization engine."""

    def __init__(self, registry: StandardizerRegistry | None = None) -> None:
        self.registry = registry or StandardizerRegistry.default()

    def standardize(
        self,
        dataframe: pd.DataFrame,
        semantic_report: SemanticDetectionReport,
        standardization_config: dict[str, Any],
    ) -> tuple[pd.DataFrame, StandardizationReport]:
        result = dataframe.copy(deep=True)
        standardized_by_standardizer: dict[str, int] = {}
        standardizer_warnings: dict[str, list[str]] = {}
        warnings: list[str] = []
        standardized_columns: set[str] = set()
        processed_columns: set[str] = set()

        for standardizer_name in self._configured_standardizers(standardization_config):
            standardizer = self.registry.create(standardizer_name)
            result = standardizer.standardize(
                dataframe=result,
                semantic_report=semantic_report,
                standardization_config=standardization_config,
            )
            standardized_by_standardizer[standardizer.name] = len(standardizer.standardized_columns)
            standardized_columns.update(standardizer.standardized_columns)
            processed_columns.update(standardizer.processed_columns)
            if standardizer.warnings:
                standardizer_warnings[standardizer.name] = standardizer.warnings
                warnings.extend(standardizer.warnings)

        skipped_columns = [
            tag.column_name
            for tag in semantic_report.columns
            if tag.column_name not in processed_columns
        ]

        report = StandardizationReport(
            total_standardized_columns=len(standardized_columns),
            standardized_by_standardizer=standardized_by_standardizer,
            warnings=warnings,
            skipped_columns=skipped_columns,
            standardizer_warnings=standardizer_warnings,
        )
        return result, report

    @staticmethod
    def _configured_standardizers(standardization_config: dict[str, Any]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for config in standardization_config.values():
            if not isinstance(config, dict):
                continue
            standardizer_name = config.get("standardizer")
            if not isinstance(standardizer_name, str):
                continue
            if standardizer_name in seen:
                continue
            seen.add(standardizer_name)
            ordered.append(standardizer_name)
        return ordered
