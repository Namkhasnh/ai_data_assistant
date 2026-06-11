from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from models.business_dataset import BusinessDatasetReport
from storage.artifact_store import ArtifactStore


DEFAULT_INTERNAL_SUFFIXES: tuple[str, ...] = (
    "__is_valid",
    "__validation_error",
    "__execution_metadata",
    "__execution_warning",
    "__execution_warnings",
    "__execution_status",
    "__execution_error",
    "__technical_warning",
    "__technical_warnings",
)


class BusinessDatasetService:
    """Generate user-facing dataset artifacts from technical dataset artifacts."""

    def __init__(
        self,
        artifact_store: ArtifactStore | None = None,
        excluded_suffixes: Sequence[str] = DEFAULT_INTERNAL_SUFFIXES,
    ) -> None:
        self.artifact_store = artifact_store or ArtifactStore()
        self.excluded_suffixes = tuple(excluded_suffixes)

    def generate_from_csv(
        self,
        input_csv_path: str | Path = "storage/artifacts/enriched_dataset.csv",
        business_filename: str = "business_dataset.csv",
        report_filename: str = "business_dataset_report.json",
    ) -> tuple[pd.DataFrame, BusinessDatasetReport]:
        """Load a technical dataset CSV and persist business dataset artifacts."""

        input_path = Path(input_csv_path)
        try:
            dataframe = pd.read_csv(input_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Business dataset input not found: {input_path}") from exc
        return self.generate_from_dataframe(
            dataframe=dataframe,
            business_filename=business_filename,
            report_filename=report_filename,
        )

    def generate_from_dataframe(
        self,
        dataframe: pd.DataFrame,
        business_filename: str = "business_dataset.csv",
        report_filename: str = "business_dataset_report.json",
    ) -> tuple[pd.DataFrame, BusinessDatasetReport]:
        """Create a read-only business dataset copy and persist its report."""

        source_dataframe = dataframe.copy(deep=True)
        included_columns = self._select_business_columns(source_dataframe)
        excluded_columns = [
            column for column in source_dataframe.columns if column not in included_columns
        ]
        business_dataframe = source_dataframe.loc[:, included_columns].copy(deep=True)
        warnings: list[str] = []
        if business_dataframe.empty and len(business_dataframe.columns) == 0:
            warnings.append("Business dataset contains no visible columns")

        report = BusinessDatasetReport(
            total_columns=len(included_columns),
            excluded_columns=[str(column) for column in excluded_columns],
            included_columns=[str(column) for column in included_columns],
            warnings=warnings,
        )
        self.artifact_store.write_dataframe_csv(business_filename, business_dataframe)
        self.artifact_store.write_json(report_filename, report)
        return business_dataframe, report

    def _select_business_columns(self, dataframe: pd.DataFrame) -> list[object]:
        return [
            column
            for column in dataframe.columns
            if not self._is_internal_column(str(column))
        ]

    def _is_internal_column(self, column_name: str) -> bool:
        return any(column_name.endswith(suffix) for suffix in self.excluded_suffixes)
