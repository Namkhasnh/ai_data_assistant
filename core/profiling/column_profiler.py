from __future__ import annotations

import json
import logging
import math
from datetime import date, datetime
from typing import Any, Literal

import pandas as pd

from models.column_profile import ColumnProfile, TopValue


DataFrameBackend = Literal["polars", "pandas"]


logger = logging.getLogger(__name__)


class ColumnProfiler:
    """Build column-level metadata for supported dataframe backends."""

    def __init__(self, top_n: int = 5) -> None:
        if top_n < 1:
            raise ValueError("top_n must be at least 1")
        self.top_n = top_n

    def profile(
        self,
        name: str,
        series: object,
        row_count: int,
        backend: DataFrameBackend,
    ) -> ColumnProfile:
        """Profile one column from a Pandas or Polars dataframe."""

        if backend == "polars":
            return self._profile_polars_series(
                name=name,
                series=series,
                row_count=row_count,
            )
        if backend == "pandas":
            if not isinstance(series, pd.Series):
                raise TypeError("Pandas backend requires a pandas.Series")
            return self._profile_pandas_series(name=name, series=series, row_count=row_count)
        raise ValueError(f"Unsupported dataframe backend: {backend}")

    def _profile_pandas_series(
        self,
        name: str,
        series: pd.Series,
        row_count: int,
    ) -> ColumnProfile:
        null_count = int(series.isna().sum())
        top_values = self._pandas_top_values(series)

        return ColumnProfile(
            name=name,
            data_type=str(series.dtype),
            null_count=null_count,
            null_percentage=self._null_percentage(null_count=null_count, row_count=row_count),
            unique_value_count=int(series.nunique(dropna=True)),
            top_values=top_values,
            sample_values=self._pandas_sample_values(series),
        )

    def _profile_polars_series(
        self,
        name: str,
        series: object,
        row_count: int,
    ) -> ColumnProfile:
        null_count = int(series.null_count())  # type: ignore[attr-defined]
        non_null_series = series.drop_nulls()  # type: ignore[attr-defined]

        return ColumnProfile(
            name=name,
            data_type=str(series.dtype),  # type: ignore[attr-defined]
            null_count=null_count,
            null_percentage=self._null_percentage(null_count=null_count, row_count=row_count),
            unique_value_count=int(non_null_series.n_unique()),
            top_values=self._polars_top_values(non_null_series),
            sample_values=self._polars_sample_values(non_null_series),
        )

    def _pandas_top_values(self, series: pd.Series) -> list[TopValue]:
        value_counts = series.dropna().value_counts().head(self.top_n)
        return [
            TopValue(value=self._json_safe_value(value), count=int(count))
            for value, count in value_counts.items()
        ]

    def _pandas_sample_values(self, series: pd.Series) -> list[object]:
        return [
            self._json_safe_value(value)
            for value in series.dropna().head(self.top_n).tolist()
        ]

    def _polars_top_values(self, series: object) -> list[TopValue]:
        value_counts = series.value_counts(sort=True).head(self.top_n)  # type: ignore[attr-defined]
        rows: list[dict[str, Any]] = value_counts.to_dicts()
        if not rows:
            return []

        count_key = "count" if "count" in rows[0] else "counts"
        value_key = next(key for key in rows[0] if key != count_key)

        return [
            TopValue(
                value=self._json_safe_value(row[value_key]),
                count=int(row[count_key]),
            )
            for row in rows
        ]

    def _polars_sample_values(self, series: object) -> list[object]:
        return [
            self._json_safe_value(value)
            for value in series.head(self.top_n).to_list()  # type: ignore[attr-defined]
        ]

    @staticmethod
    def _null_percentage(null_count: int, row_count: int) -> float:
        if row_count == 0:
            return 0.0
        return round((null_count / row_count) * 100, 4)

    @staticmethod
    def _json_safe_value(value: object) -> object:
        if hasattr(value, "item"):
            try:
                value = value.item()  # type: ignore[assignment,union-attr]
            except (AttributeError, ValueError, TypeError):
                logger.debug("Could not convert scalar value with item()", exc_info=True)

        if isinstance(value, (date, datetime)):
            return value.isoformat()

        if isinstance(value, float) and math.isnan(value):
            return None

        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)
