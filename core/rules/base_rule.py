from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from models.rule import Rule


class RuleApplicationError(ValueError):
    """Raised when a rule cannot be applied safely."""


class BaseRule(ABC):
    """Abstract interface for deterministic rule executors."""

    @abstractmethod
    def apply(self, df: pd.DataFrame, rule: Rule) -> pd.DataFrame:
        """Apply a rule and return a new dataframe."""

    @staticmethod
    def copy_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        return df.copy(deep=True)

    @staticmethod
    def require_column(df: pd.DataFrame, column: str) -> None:
        if column not in df.columns:
            raise RuleApplicationError(f"Column not found: {column}")
