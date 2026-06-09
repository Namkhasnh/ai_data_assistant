from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.profiling.column_profiler import ColumnProfiler, DataFrameBackend
from models.dataset import DatasetMetadata


class MetadataBuilder:
    """Build dataset profiling metadata."""

    def __init__(self, column_profiler: ColumnProfiler | None = None) -> None:
        self.column_profiler = column_profiler or ColumnProfiler()

    def build(
        self,
        source_path: Path,
        dataframe: object,
        backend: DataFrameBackend,
    ) -> DatasetMetadata:
        """Build dataset and column metadata for a loaded dataframe."""

        if backend == "polars":
            return self._build_for_polars(source_path=source_path, dataframe=dataframe)
        if backend == "pandas":
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError("Pandas backend requires a pandas.DataFrame")
            return self._build_for_pandas(source_path=source_path, dataframe=dataframe)
        raise ValueError(f"Unsupported dataframe backend: {backend}")

    def _build_for_pandas(
        self,
        source_path: Path,
        dataframe: pd.DataFrame,
    ) -> DatasetMetadata:
        row_count = int(len(dataframe))
        columns = [
            self.column_profiler.profile(
                name=str(column_name),
                series=dataframe[column_name],
                row_count=row_count,
                backend="pandas",
            )
            for column_name in dataframe.columns
        ]

        return DatasetMetadata(
            source_file=source_path.name,
            file_format=self._file_format(source_path),
            row_count=row_count,
            column_count=int(len(dataframe.columns)),
            duplicate_count=int(dataframe.duplicated(keep="first").sum()),
            columns=columns,
        )

    def _build_for_polars(self, source_path: Path, dataframe: object) -> DatasetMetadata:
        row_count = int(dataframe.height)  # type: ignore[attr-defined]
        column_names: list[str] = list(dataframe.columns)  # type: ignore[attr-defined]
        columns = [
            self.column_profiler.profile(
                name=column_name,
                series=dataframe[column_name],  # type: ignore[index]
                row_count=row_count,
                backend="polars",
            )
            for column_name in column_names
        ]

        return DatasetMetadata(
            source_file=source_path.name,
            file_format=self._file_format(source_path),
            row_count=row_count,
            column_count=int(len(column_names)),
            duplicate_count=self._polars_duplicate_count(dataframe),
            columns=columns,
        )

    @staticmethod
    def _polars_duplicate_count(dataframe: object) -> int:
        unique_rows = dataframe.unique(maintain_order=False).height  # type: ignore[attr-defined]
        return int(dataframe.height - unique_rows)  # type: ignore[attr-defined]

    @staticmethod
    def _file_format(source_path: Path) -> str:
        return source_path.suffix.lower().lstrip(".")
