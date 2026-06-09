from __future__ import annotations

import re
from typing import Any

import pandas as pd

from core.rules.base_rule import BaseRule, RuleApplicationError
from models.rule import Rule


class RegexRule(BaseRule):
    """Extract regex capture groups into one or more output columns."""

    def apply(self, df: pd.DataFrame, rule: Rule) -> pd.DataFrame:
        self.require_column(df, rule.column)
        pattern = rule.parameters.get("pattern")
        output_columns = rule.parameters.get("output_columns")
        output_types = rule.parameters.get("output_types", {})

        if not isinstance(pattern, str) or not pattern:
            raise RuleApplicationError("RegexRule requires parameters.pattern")
        if not isinstance(output_columns, list) or not output_columns:
            raise RuleApplicationError("RegexRule requires parameters.output_columns")
        if not all(isinstance(column, str) and column for column in output_columns):
            raise RuleApplicationError("RegexRule output columns must be non-empty strings")
        if not isinstance(output_types, dict):
            raise RuleApplicationError("RegexRule parameters.output_types must be a dictionary")

        try:
            compiled_pattern = re.compile(pattern)
        except re.error as exc:
            raise RuleApplicationError(f"Invalid regex pattern: {pattern}") from exc

        result = self.copy_dataframe(df)
        extracted = result[rule.column].astype("string").str.extract(compiled_pattern)
        if extracted.shape[1] != len(output_columns):
            raise RuleApplicationError(
                "RegexRule output_columns count must match regex capture group count"
            )

        for index, output_column in enumerate(output_columns):
            output_type = output_types.get(output_column, "str")
            result[output_column] = self._cast_series(extracted[index], output_type)

        return result

    @staticmethod
    def _cast_series(series: pd.Series, output_type: Any) -> pd.Series:
        if output_type == "int":
            return pd.to_numeric(series, errors="coerce").astype("Int64")
        if output_type == "float":
            return pd.to_numeric(series, errors="coerce")
        if output_type == "str":
            return series.astype("string")
        raise RuleApplicationError(f"Unsupported regex output type: {output_type}")
