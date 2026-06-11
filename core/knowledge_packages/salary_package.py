from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from core.knowledge_packages.base_package import BasePackage


class SalaryPackage(BasePackage):
    """Generate semantic business attributes from standardized salary columns."""

    package_id = "salary"
    name = "Salary Package"
    description = "Generate semantic business attributes from standardized salary columns."
    version = "1.0"
    enabled = True
    priority = 300
    required_columns: tuple[str, ...] = ()
    produced_columns = ("salary_avg", "currency", "salary_unit")

    salary_min_semantic_type = "SALARY_MIN"
    salary_max_semantic_type = "SALARY_MAX"
    salary_type_semantic_type = "SALARY_TYPE"
    currency = "VND"
    salary_unit = "million"

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Append salary semantic columns using already-standardized salary values."""

        output = dataframe.copy(deep=True)
        salary_min_column = self._first_semantic_column(
            dataframe=output,
            runtime_context=runtime_context,
            semantic_type=self.salary_min_semantic_type,
        )
        salary_max_column = self._first_semantic_column(
            dataframe=output,
            runtime_context=runtime_context,
            semantic_type=self.salary_max_semantic_type,
        )
        salary_type_column = self._first_semantic_column(
            dataframe=output,
            runtime_context=runtime_context,
            semantic_type=self.salary_type_semantic_type,
        )
        if (
            salary_min_column is None
            or salary_max_column is None
            or salary_type_column is None
        ):
            self.request_skip("Salary package skipped; no usable semantic columns.")
            return output

        writable_columns: set[str] = set()
        for column in self.produced_columns:
            if column not in output.columns:
                output[column] = pd.NA
                writable_columns.add(column)

        for row_index, row in output.iterrows():
            if "salary_avg" in writable_columns:
                output.at[row_index, "salary_avg"] = self._salary_avg(
                    salary_min=row.get(salary_min_column),
                    salary_max=row.get(salary_max_column),
                    salary_type=row.get(salary_type_column),
                )
            if "currency" in writable_columns:
                output.at[row_index, "currency"] = self.currency
            if "salary_unit" in writable_columns:
                output.at[row_index, "salary_unit"] = self.salary_unit

        return output

    def _first_semantic_column(
        self,
        dataframe: pd.DataFrame,
        runtime_context: Mapping[str, Any] | None,
        semantic_type: str,
    ) -> str | None:
        semantic_columns = self._semantic_columns(runtime_context)
        for column in semantic_columns.get(semantic_type, []):
            if column in dataframe.columns:
                return column
        return None

    def _semantic_columns(
        self,
        runtime_context: Mapping[str, Any] | None,
    ) -> Mapping[str, list[str]]:
        if runtime_context is None:
            return {}
        semantic_columns = runtime_context.get("semantic_columns", {})
        if not isinstance(semantic_columns, Mapping):
            return {}
        return {
            str(semantic_type): [
                str(column)
                for column in columns
                if isinstance(column, str) and column
            ]
            for semantic_type, columns in semantic_columns.items()
            if isinstance(columns, list)
        }

    def _salary_avg(
        self,
        salary_min: object,
        salary_max: object,
        salary_type: object,
    ) -> float | int | None:
        min_value = self._numeric_value(salary_min)
        max_value = self._numeric_value(salary_max)
        if min_value is None or max_value is None:
            return None

        salary_type_text = "" if salary_type is None or pd.isna(salary_type) else str(salary_type)
        if salary_type_text.casefold() == "fixed" or min_value == max_value:
            return min_value
        return (min_value + max_value) / 2

    def _numeric_value(self, value: object) -> float | int | None:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, int | float):
            return value
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
