from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pandas.errors import EmptyDataError, ParserError

from core.profiling.column_profiler import DataFrameBackend
from core.profiling.metadata_builder import MetadataBuilder
from models.dataset import DatasetMetadata


logger = logging.getLogger(__name__)


class ProfilingError(Exception):
    """Base exception for user-facing profiling errors."""


class EmptyDatasetError(ProfilingError):
    """Raised when a dataset has no rows or no columns."""


class MalformedDatasetError(ProfilingError):
    """Raised when a dataset file cannot be parsed."""


@dataclass(frozen=True)
class LoadedDataset:
    dataframe: object
    backend: DataFrameBackend


class DatasetProfiler:
    """Load structured datasets and produce profiling metadata."""

    SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}

    def __init__(
        self,
        metadata_builder: MetadataBuilder | None = None,
    ) -> None:
        self.metadata_builder = metadata_builder or MetadataBuilder()

    def profile(
        self,
        file_path: str | Path,
    ) -> DatasetMetadata:
        """Profile a CSV or XLSX file and return metadata."""

        source_path = Path(file_path)
        self._validate_source_path(source_path)

        loaded_dataset = self._load_dataset(source_path)
        metadata = self.metadata_builder.build(
            source_path=source_path,
            dataframe=loaded_dataset.dataframe,
            backend=loaded_dataset.backend,
        )

        return metadata

    def _validate_source_path(self, source_path: Path) -> None:
        if not source_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {source_path}")
        if not source_path.is_file():
            raise ValueError(f"Dataset path must be a file: {source_path}")
        if source_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            supported = ", ".join(sorted(self.SUPPORTED_EXTENSIONS))
            raise ValueError(
                f"Unsupported dataset format '{source_path.suffix}'. "
                f"Supported formats: {supported}"
            )
        if source_path.stat().st_size == 0:
            raise EmptyDatasetError(f"Dataset file is empty: {source_path}")

    def _load_dataset(self, source_path: Path) -> LoadedDataset:
        suffix = source_path.suffix.lower()
        if suffix == ".csv":
            return self._load_csv(source_path)
        if suffix == ".xlsx":
            return self._load_xlsx(source_path)
        raise ValueError(f"Unsupported dataset format: {suffix}")

    def _load_csv(self, source_path: Path) -> LoadedDataset:
        polars_dataset = self._try_load_csv_with_polars(source_path)
        if polars_dataset is not None:
            self._validate_loaded_dataset(polars_dataset, source_path)
            return polars_dataset

        logger.info("Loading CSV with Pandas fallback: %s", source_path)
        try:
            loaded_dataset = LoadedDataset(dataframe=pd.read_csv(source_path), backend="pandas")
        except EmptyDataError as exc:
            raise EmptyDatasetError(f"CSV file is empty: {source_path}") from exc
        except (ParserError, UnicodeDecodeError, ValueError) as exc:
            raise MalformedDatasetError(f"CSV file could not be parsed: {source_path}") from exc
        self._validate_loaded_dataset(loaded_dataset, source_path)
        return loaded_dataset

    def _load_xlsx(self, source_path: Path) -> LoadedDataset:
        polars_dataset = self._try_load_xlsx_with_polars(source_path)
        if polars_dataset is not None:
            self._validate_loaded_dataset(polars_dataset, source_path)
            return polars_dataset

        logger.info("Loading XLSX with Pandas fallback: %s", source_path)
        try:
            loaded_dataset = LoadedDataset(dataframe=pd.read_excel(source_path), backend="pandas")
        except Exception as exc:
            raise MalformedDatasetError(f"XLSX file could not be parsed: {source_path}") from exc
        self._validate_loaded_dataset(loaded_dataset, source_path)
        return loaded_dataset

    def _try_load_csv_with_polars(self, source_path: Path) -> LoadedDataset | None:
        polars = self._import_polars()
        if polars is None:
            return None

        try:
            logger.info("Loading CSV with Polars: %s", source_path)
            return LoadedDataset(dataframe=polars.read_csv(source_path), backend="polars")
        except Exception:
            logger.warning(
                "Polars failed to load CSV; falling back to Pandas: %s",
                source_path,
                exc_info=True,
            )
            return None

    def _try_load_xlsx_with_polars(self, source_path: Path) -> LoadedDataset | None:
        polars = self._import_polars()
        if polars is None or not hasattr(polars, "read_excel"):
            return None

        try:
            logger.info("Loading XLSX with Polars: %s", source_path)
            return LoadedDataset(dataframe=polars.read_excel(source_path), backend="polars")
        except Exception:
            logger.warning(
                "Polars failed to load XLSX; falling back to Pandas: %s",
                source_path,
                exc_info=True,
            )
            return None

    def _validate_loaded_dataset(
        self,
        loaded_dataset: LoadedDataset,
        source_path: Path,
    ) -> None:
        if loaded_dataset.backend == "pandas":
            dataframe = loaded_dataset.dataframe
            if not isinstance(dataframe, pd.DataFrame):
                raise TypeError("Pandas backend requires a pandas.DataFrame")
            row_count = int(len(dataframe))
            column_count = int(len(dataframe.columns))
        else:
            row_count = int(loaded_dataset.dataframe.height)  # type: ignore[attr-defined]
            column_count = int(len(loaded_dataset.dataframe.columns))  # type: ignore[attr-defined]

        if row_count == 0 or column_count == 0:
            raise EmptyDatasetError(f"Dataset has no rows or columns: {source_path}")

    @staticmethod
    def _import_polars() -> Any | None:
        try:
            import polars as pl
        except ImportError:
            logger.info("Polars is not installed; using Pandas fallback")
            return None
        return pl
