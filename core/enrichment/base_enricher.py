from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from models.enrichment import EnrichmentConfigItem
from models.semantic_tag import SemanticDetectionReport


class EnrichmentError(ValueError):
    """Raised when an enrichment config cannot be applied safely."""


class BaseEnricher(ABC):
    """Base interface for deterministic enrichment plugins."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name
        self.warnings: list[str] = []
        self.enriched_columns: set[str] = set()
        self.processed_columns: set[str] = set()

    @abstractmethod
    def enrich(
        self,
        dataframe: pd.DataFrame,
        semantic_report: SemanticDetectionReport,
        enrichment_config: EnrichmentConfigItem,
    ) -> pd.DataFrame:
        """Return an enriched dataframe without mutating the input."""

    @staticmethod
    def copy_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
        return dataframe.copy(deep=True)
