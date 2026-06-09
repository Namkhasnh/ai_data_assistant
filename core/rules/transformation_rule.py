from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from core.rules.base_rule import BaseRule, RuleApplicationError
from models.rule import Rule


TransformationOperation = Callable[[pd.Series, dict[str, object]], pd.Series]


class TransformationRule(BaseRule):
    """Apply common deterministic text transformations."""

    def __init__(self) -> None:
        self.operations: dict[str, TransformationOperation] = {
            "trim": self._trim,
            "lowercase": self._lowercase,
            "uppercase": self._uppercase,
            "titlecase": self._titlecase,
            "replace": self._replace,
            "remove_extra_spaces": self._remove_extra_spaces,
        }

    def apply(self, df: pd.DataFrame, rule: Rule) -> pd.DataFrame:
        self.require_column(df, rule.column)
        operation_name = rule.parameters.get("operation")
        if not isinstance(operation_name, str):
            raise RuleApplicationError("TransformationRule requires parameters.operation")

        operation = self.operations.get(operation_name)
        if operation is None:
            raise RuleApplicationError(f"Unsupported transformation operation: {operation_name}")

        output_column = rule.parameters.get("output_column", rule.column)
        if not isinstance(output_column, str) or not output_column:
            raise RuleApplicationError("TransformationRule output_column must be a non-empty string")

        result = self.copy_dataframe(df)
        source = result[rule.column].astype("string")
        result[output_column] = operation(source, rule.parameters)
        return result

    @staticmethod
    def _trim(series: pd.Series, parameters: dict[str, object]) -> pd.Series:
        return series.str.strip()

    @staticmethod
    def _lowercase(series: pd.Series, parameters: dict[str, object]) -> pd.Series:
        return series.str.lower()

    @staticmethod
    def _uppercase(series: pd.Series, parameters: dict[str, object]) -> pd.Series:
        return series.str.upper()

    @staticmethod
    def _titlecase(series: pd.Series, parameters: dict[str, object]) -> pd.Series:
        return series.str.title()

    @staticmethod
    def _replace(series: pd.Series, parameters: dict[str, object]) -> pd.Series:
        old_value = parameters.get("old")
        new_value = parameters.get("new")
        if not isinstance(old_value, str) or not isinstance(new_value, str):
            raise RuleApplicationError("replace requires string parameters.old and parameters.new")
        regex = bool(parameters.get("regex", False))
        return series.str.replace(old_value, new_value, regex=regex)

    @staticmethod
    def _remove_extra_spaces(series: pd.Series, parameters: dict[str, object]) -> pd.Series:
        return series.str.replace(r"\s+", " ", regex=True).str.strip()
