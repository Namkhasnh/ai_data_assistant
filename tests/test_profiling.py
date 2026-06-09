from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from core.profiling.dataset_profiler import (
    DatasetProfiler,
    EmptyDatasetError,
    MalformedDatasetError,
)
from services.profiling_service import ProfilingService
from storage.artifact_store import ArtifactStore


def test_profile_csv_writes_metadata_json(tmp_path: Path) -> None:
    csv_path = tmp_path / "jobs.csv"
    output_path = tmp_path / "metadata.json"
    dataframe = pd.DataFrame(
        {
            "title": ["AI Engineer", "AI Engineer", None, "Data Analyst"],
            "location": ["HN", "HN", "Ha Noi", "HCM"],
            "salary": ["20-30tr", "20-30tr", None, "15M-20M"],
        }
    )
    dataframe.to_csv(csv_path, index=False)

    metadata = ProfilingService().profile_dataset(csv_path, output_path=output_path)

    assert output_path.exists()
    assert metadata.row_count == 4
    assert metadata.column_count == 3
    assert metadata.duplicate_count == 1

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["source_file"] == "jobs.csv"
    assert payload["file_format"] == "csv"
    assert payload["row_count"] == 4
    assert payload["column_count"] == 3
    assert payload["duplicate_count"] == 1

    columns = {column["name"]: column for column in payload["columns"]}
    assert columns["title"]["null_percentage"] == 25.0
    assert columns["title"]["unique_value_count"] == 2
    assert columns["title"]["top_values"][0] == {
        "value": "AI Engineer",
        "count": 2,
    }
    assert columns["salary"]["null_percentage"] == 25.0
    assert columns["salary"]["unique_value_count"] == 2


def test_profile_xlsx_uses_supported_metadata_shape(tmp_path: Path) -> None:
    xlsx_path = tmp_path / "jobs.xlsx"
    dataframe = pd.DataFrame(
        {
            "title": ["AI Engineer", "Data Analyst"],
            "salary": ["20-30tr", None],
        }
    )
    dataframe.to_excel(xlsx_path, index=False)

    metadata = DatasetProfiler().profile(xlsx_path)

    assert metadata.source_file == "jobs.xlsx"
    assert metadata.file_format == "xlsx"
    assert metadata.row_count == 2
    assert metadata.column_count == 2
    assert metadata.duplicate_count == 0

    salary_profile = next(column for column in metadata.columns if column.name == "salary")
    assert salary_profile.null_percentage == 50.0
    assert salary_profile.unique_value_count == 1
    assert salary_profile.top_values[0].value == "20-30tr"


def test_profile_dataset_defaults_to_metadata_json_next_to_source(tmp_path: Path) -> None:
    csv_path = tmp_path / "jobs.csv"
    artifact_dir = tmp_path / "artifacts"
    pd.DataFrame({"title": ["AI Engineer"]}).to_csv(csv_path, index=False)

    artifact_store = ArtifactStore(artifact_dir=artifact_dir)
    ProfilingService(artifact_store=artifact_store).profile_dataset(csv_path)

    assert (artifact_dir / "metadata.json").exists()
    assert not (tmp_path / "metadata.json").exists()


def test_profile_rejects_unsupported_format(tmp_path: Path) -> None:
    text_path = tmp_path / "jobs.txt"
    text_path.write_text("title\nAI Engineer\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported dataset format"):
        DatasetProfiler().profile(text_path)


def test_profile_rejects_empty_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("", encoding="utf-8")

    with pytest.raises(EmptyDatasetError, match="empty"):
        DatasetProfiler().profile(csv_path)


def test_profile_rejects_header_only_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "header_only.csv"
    csv_path.write_text("title,salary\n", encoding="utf-8")

    with pytest.raises(EmptyDatasetError, match="no rows or columns"):
        DatasetProfiler().profile(csv_path)


def test_profile_rejects_empty_xlsx(tmp_path: Path) -> None:
    xlsx_path = tmp_path / "empty.xlsx"
    xlsx_path.write_bytes(b"")

    with pytest.raises(EmptyDatasetError, match="empty"):
        DatasetProfiler().profile(xlsx_path)


def test_profile_rejects_malformed_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "malformed.csv"
    csv_path.write_bytes(b"\xff\xfe\x00")

    with pytest.raises(MalformedDatasetError, match="CSV file could not be parsed"):
        DatasetProfiler().profile(csv_path)


def test_profile_rejects_malformed_xlsx(tmp_path: Path) -> None:
    xlsx_path = tmp_path / "malformed.xlsx"
    xlsx_path.write_text("not an xlsx workbook", encoding="utf-8")

    with pytest.raises(MalformedDatasetError, match="XLSX file could not be parsed"):
        DatasetProfiler().profile(xlsx_path)
