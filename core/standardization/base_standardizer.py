from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from models.semantic_tag import SemanticDetectionReport


class StandardizationError(ValueError):
    """Raised when a standardization config cannot be applied safely."""


class BaseStandardizer(ABC):
    """Base interface for deterministic standardizers."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name
        self.warnings: list[str] = []
        self.standardized_columns: set[str] = set()
        self.processed_columns: set[str] = set()

    @abstractmethod
    def standardize(
        self,
        dataframe: pd.DataFrame,
        semantic_report: SemanticDetectionReport,
        standardization_config: dict[str, Any],
    ) -> pd.DataFrame:
        """Return a standardized dataframe without mutating the input."""

    @staticmethod
    def copy_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.copy(deep=True)

    @staticmethod
    def semantic_types(semantic_report: SemanticDetectionReport) -> dict[str, str]:
        return {
            tag.column_name: tag.semantic_type
            for tag in semantic_report.columns
        }

    @staticmethod
    def semantic_columns_by_type(semantic_report: SemanticDetectionReport) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for tag in semantic_report.columns:
            result.setdefault(tag.semantic_type, []).append(tag.column_name)
        return result

    @staticmethod
    def configured_source_columns(
        config: dict[str, Any],
        semantic_type: str,
        semantic_columns: dict[str, list[str]],
    ) -> list[str]:
        source_column = config.get("source_column")
        if isinstance(source_column, str):
            return [source_column]
        if isinstance(source_column, list):
            return [column for column in source_column if isinstance(column, str)]
        return semantic_columns.get(semantic_type, [])

    @staticmethod
    def configured_target_column(config: dict[str, Any], source_column: str) -> str:
        output_column = config.get("output_column")
        return output_column if isinstance(output_column, str) else source_column
