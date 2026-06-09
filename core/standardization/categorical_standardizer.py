from __future__ import annotations

from typing import Any

import pandas as pd

from core.standardization.base_standardizer import BaseStandardizer
from models.semantic_tag import SemanticDetectionReport


class CategoricalStandardizer(BaseStandardizer):
    """Normalize categorical values from deterministic config mappings."""

    def __init__(self) -> None:
        super().__init__(name="categorical")

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

            categories = config.get("categories", {})
            if not isinstance(categories, dict) or not categories:
                self.warnings.append(f"No category mapping configured for {semantic_type}")
                continue

            mapping = {
                str(source).casefold(): target
                for source, target in categories.items()
            }

            for column in self.configured_source_columns(config, semantic_type, semantic_columns):
                if column not in result.columns:
                    self.warnings.append(f"Column not found for categorical standardization: {column}")
                    continue

                target_column = self.configured_target_column(config, column)
                result[target_column] = result[column].map(
                    lambda value: self._map_value(value, mapping)
                )
                self.standardized_columns.add(target_column)
                self.processed_columns.add(column)

        return result

    @staticmethod
    def _map_value(value: Any, mapping: dict[str, Any]) -> Any:
        if pd.isna(value):
            return value
        return mapping.get(str(value).casefold(), value)
