from __future__ import annotations

import json
from pathlib import Path

from models.dataset import DatasetMetadata
from services.profiling_service import ProfilingService


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "topcv_jobs.csv"
METADATA_PATH = PROJECT_ROOT / "storage" / "artifacts" / "metadata.json"


def test_phase1_profiles_topcv_jobs_and_generates_metadata_json() -> None:
    metadata = ProfilingService().profile_dataset(
        file_path=DATASET_PATH,
        output_path=METADATA_PATH,
    )

    assert isinstance(metadata, DatasetMetadata)
    assert metadata.source_file == "topcv_jobs.csv"
    assert metadata.file_format == "csv"
    assert metadata.row_count == 210
    assert metadata.column_count == 13
    assert metadata.duplicate_count == 0
    assert len(metadata.columns) == metadata.column_count

    column_names = {column.name for column in metadata.columns}
    assert {
        "job_id",
        "title",
        "salary",
        "location",
        "experience",
        "company_name",
    }.issubset(column_names)

    title_profile = next(column for column in metadata.columns if column.name == "title")
    assert title_profile.data_type
    assert 0.0 <= title_profile.null_percentage <= 100.0
    assert title_profile.unique_value_count > 0
    assert title_profile.top_values

    assert METADATA_PATH.exists()
    payload = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    assert payload["source_file"] == metadata.source_file
    assert payload["row_count"] == metadata.row_count
    assert payload["column_count"] == metadata.column_count
    assert len(payload["columns"]) == metadata.column_count
    assert not (PROJECT_ROOT / "metadata.json").exists()
