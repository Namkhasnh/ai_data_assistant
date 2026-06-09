from __future__ import annotations

import re
from typing import Any

import pandas as pd

from core.rules.base_rule import BaseRule, RuleApplicationError
from models.rule import Rule


class ValidationRule(BaseRule):
    """Validate data quality without removing rows."""

    EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
    PHONE_PATTERN = r"^\+?[\d\s().-]{7,}$"

    def apply(self, df: pd.DataFrame, rule: Rule) -> pd.DataFrame:
        self.require_column(df, rule.column)
        result = self.copy_dataframe(df)
        source_series = result[rule.column]
        valid_mask = self._build_valid_mask(source_series, rule).fillna(False)
        if bool(rule.parameters.get("allow_null", False)):
            valid_mask = valid_mask | source_series.isna()

        is_valid_column, error_column = self._validation_columns(result, rule)
        message = str(
            rule.parameters.get(
                "message",
                f"{rule.column} failed validation",
            )
        )
        result[is_valid_column] = valid_mask.astype(bool)
        result[error_column] = result[is_valid_column].map(
            lambda is_valid: "" if is_valid else message
        )
        return result

    def _build_valid_mask(self, series: pd.Series, rule: Rule) -> pd.Series:
        if "operator" in rule.parameters:
            return self._comparison_mask(series, rule.parameters)
        if "format" in rule.parameters:
            return self._format_mask(series, rule.parameters)
        if "pattern" in rule.parameters:
            return self._pattern_mask(series, str(rule.parameters["pattern"]))
        raise RuleApplicationError(
            "ValidationRule requires one of parameters.operator, parameters.format, or parameters.pattern"
        )

    @staticmethod
    def _comparison_mask(series: pd.Series, parameters: dict[str, Any]) -> pd.Series:
        operator = parameters.get("operator")
        expected_value = parameters.get("value")
        if operator not in {">", ">=", "<", "<=", "==", "!="}:
            raise RuleApplicationError(f"Unsupported validation operator: {operator}")

        if isinstance(expected_value, (int, float)):
            comparable = pd.to_numeric(series, errors="coerce")
        else:
            comparable = series

        if operator == ">":
            return comparable > expected_value
        if operator == ">=":
            return comparable >= expected_value
        if operator == "<":
            return comparable < expected_value
        if operator == "<=":
            return comparable <= expected_value
        if operator == "==":
            return comparable == expected_value
        return comparable != expected_value

    def _format_mask(self, series: pd.Series, parameters: dict[str, Any]) -> pd.Series:
        value_format = parameters.get("format")
        if value_format == "email":
            return self._pattern_mask(series, self.EMAIL_PATTERN)
        if value_format == "phone":
            return self._pattern_mask(series, self.PHONE_PATTERN)
        if value_format == "date":
            date_format = parameters.get("date_format")
            parsed = pd.to_datetime(
                series,
                format=date_format if isinstance(date_format, str) else None,
                errors="coerce",
            )
            return parsed.notna()
        raise RuleApplicationError(f"Unsupported validation format: {value_format}")

    @staticmethod
    def _pattern_mask(series: pd.Series, pattern: str) -> pd.Series:
        try:
            compiled_pattern = re.compile(pattern)
        except re.error as exc:
            raise RuleApplicationError(f"Invalid validation pattern: {pattern}") from exc
        return series.astype("string").map(
            lambda value: bool(compiled_pattern.search(str(value))) if pd.notna(value) else False
        )

    @staticmethod
    def _validation_columns(df: pd.DataFrame, rule: Rule) -> tuple[str, str]:
        base_valid = f"{rule.column}__is_valid"
        base_error = f"{rule.column}__validation_error"
        if base_valid not in df.columns and base_error not in df.columns:
            return base_valid, base_error

        safe_rule_id = re.sub(r"[^a-zA-Z0-9_]+", "_", rule.id).strip("_")
        return (
            f"{rule.column}__{safe_rule_id}__is_valid",
            f"{rule.column}__{safe_rule_id}__validation_error",
        )
