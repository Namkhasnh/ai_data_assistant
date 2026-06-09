from __future__ import annotations

from typing import Any

import pandas as pd

from core.rules.base_rule import BaseRule, RuleApplicationError
from models.rule import Rule


class MappingRule(BaseRule):
    """Normalize categorical values using a dictionary mapping."""

    def apply(self, df: pd.DataFrame, rule: Rule) -> pd.DataFrame:
        self.require_column(df, rule.column)
        mapping = rule.parameters.get("mapping")
        if not isinstance(mapping, dict):
            raise RuleApplicationError("MappingRule requires parameters.mapping")

        case_sensitive = bool(rule.parameters.get("case_sensitive", True))
        result = self.copy_dataframe(df)
        result[rule.column] = result[rule.column].map(
            lambda value: self._map_value(
                value=value,
                mapping=mapping,
                case_sensitive=case_sensitive,
            )
        )
        return result

    @staticmethod
    def _map_value(
        value: Any,
        mapping: dict[Any, Any],
        case_sensitive: bool,
    ) -> Any:
        if pd.isna(value):
            return value

        if case_sensitive:
            return mapping.get(value, value)

        normalized_mapping = {
            str(source).casefold(): target
            for source, target in mapping.items()
        }
        return normalized_mapping.get(str(value).casefold(), value)
