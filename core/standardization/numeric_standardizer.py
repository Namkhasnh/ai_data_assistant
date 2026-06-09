from __future__ import annotations

from typing import Any

import pandas as pd

from core.standardization.base_standardizer import BaseStandardizer
from models.semantic_tag import SemanticDetectionReport


class NumericStandardizer(BaseStandardizer):
    """Bucket numeric values using thresholds supplied by config."""

    def __init__(self) -> None:
        super().__init__(name="numeric_bucket")

    def standardize(
        self,
        dataframe: pd.DataFrame,
        semantic_report: SemanticDetectionReport,
        standardization_config: dict[str, Any],
    ) -> pd.DataFrame:
        result = self.copy_dataframe(dataframe)

        for semantic_type, config in standardization_config.items():
            if config.get("standardizer") != self.name:
                continue

            source_column = config.get("source_column")
            output_column = config.get("output_column")
            buckets = config.get("buckets", [])

            if not isinstance(source_column, str) or source_column not in result.columns:
                self.warnings.append(f"Numeric bucket source column not found for {semantic_type}: {source_column}")
                continue
            if not isinstance(output_column, str):
                self.warnings.append(f"Numeric bucket output_column missing for {semantic_type}")
                continue
            if not isinstance(buckets, list) or not buckets:
                self.warnings.append(f"Numeric bucket thresholds missing for {semantic_type}")
                continue

            numeric_values = pd.to_numeric(result[source_column], errors="coerce")
            result[output_column] = numeric_values.map(
                lambda value: self._bucket_value(value, buckets)
            )
            self.standardized_columns.add(output_column)
            self.processed_columns.add(source_column)

        return result

    @staticmethod
    def _bucket_value(value: Any, buckets: list[dict[str, Any]]) -> Any:
        if pd.isna(value):
            return pd.NA

        numeric_value = float(value)
        for bucket in buckets:
            label = bucket.get("label")
            minimum = bucket.get("min")
            maximum = bucket.get("max")

            if label is None:
                continue
            if minimum is not None and numeric_value < float(minimum):
                continue
            if maximum is not None and numeric_value > float(maximum):
                continue
            return label

        return pd.NA
