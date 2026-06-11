from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from core.exporting.csv_exporter import CSVExporter
from core.exporting.export_registry import ExportRegistry
from services.business_dataset_service import BusinessDatasetService
from services.export_service import ExportService
from storage.artifact_store import ArtifactStore


def test_business_dataset_filters_internal_columns_and_preserves_visible_columns(
    tmp_path: Path,
) -> None:
    dataframe = pd.DataFrame(
        {
            "title": ["Data Analyst"],
            "salary": ["20-30 triệu"],
            "unknown_future_column": ["visible"],
            "salary_min": [20],
            "salary_max": [30],
            "salary_type": ["20-30 triệu"],
            "experience_level": ["Mid"],
            "standardized_title": ["Data Analyst"],
            "standardized_location": ["Hà Nội"],
            "job_family": ["Analytics"],
            "job_domain": ["Data"],
            "region": ["North"],
            "country": ["Vietnam"],
            "salary_min__is_valid": [True],
            "salary_min__validation_error": [None],
            "rule__execution_metadata": ["internal"],
        }
    )
    original = dataframe.copy(deep=True)

    business, report = BusinessDatasetService(
        artifact_store=ArtifactStore(tmp_path / "artifacts")
    ).generate_from_dataframe(dataframe)

    expected_columns = [
        "title",
        "salary",
        "unknown_future_column",
        "salary_min",
        "salary_max",
        "salary_type",
        "experience_level",
        "standardized_title",
        "standardized_location",
        "job_family",
        "job_domain",
        "region",
        "country",
    ]
    assert list(business.columns) == expected_columns
    assert report.included_columns == expected_columns
    assert report.excluded_columns == [
        "salary_min__is_valid",
        "salary_min__validation_error",
        "rule__execution_metadata",
    ]
    assert report.total_columns == len(expected_columns)
    assert report.warnings == []
    assert "unknown_future_column" in business.columns
    assert "standardized_title" in business.columns
    assert "standardized_location" in business.columns
    assert "job_family" in business.columns
    assert "job_domain" in business.columns
    assert all(not column.endswith("__is_valid") for column in business.columns)
    assert all(not column.endswith("__validation_error") for column in business.columns)
    pd.testing.assert_frame_equal(dataframe, original)
    assert (tmp_path / "artifacts" / "business_dataset.csv").exists()
    assert (tmp_path / "artifacts" / "business_dataset_report.json").exists()


def test_business_dataset_generation_does_not_modify_technical_dataset_file(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    technical_path = artifact_dir / "enriched_dataset.csv"
    pd.DataFrame(
        {
            "title": ["Data Engineer"],
            "standardized_title": ["Data Engineer"],
            "job_family": ["Engineering"],
            "salary_min__is_valid": [True],
        }
    ).to_csv(technical_path, index=False)
    before_hash = _sha256(technical_path)

    business, report = BusinessDatasetService(
        artifact_store=ArtifactStore(artifact_dir)
    ).generate_from_csv(technical_path)

    assert _sha256(technical_path) == before_hash
    assert list(business.columns) == ["title", "standardized_title", "job_family"]
    assert report.excluded_columns == ["salary_min__is_valid"]


def test_export_service_prefers_business_dataset_when_present(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    export_dir = tmp_path / "exports"
    artifact_dir.mkdir()
    pd.DataFrame(
        {
            "title": ["Business Data Analyst"],
            "standardized_title": ["Data Analyst"],
            "job_family": ["Analytics"],
        }
    ).to_csv(artifact_dir / "business_dataset.csv", index=False)
    pd.DataFrame({"title": ["technical"], "salary_min__is_valid": [True]}).to_csv(
        artifact_dir / "enriched_dataset.csv",
        index=False,
    )
    pd.DataFrame({"title": ["standardized"]}).to_csv(
        artifact_dir / "standardized_dataset.csv",
        index=False,
    )

    report = ExportService(
        artifact_dir=artifact_dir,
        audit_dir=tmp_path / "audit",
        report_dir=tmp_path / "reports",
        export_dir=export_dir,
        registry=_csv_only_registry(),
    ).export_all()

    exported = pd.read_csv(export_dir / "csv" / "export_dataset.csv")
    assert list(exported.columns) == ["title", "standardized_title", "job_family"]
    assert exported.loc[0, "title"] == "Business Data Analyst"
    assert "salary_min__is_valid" not in exported.columns
    assert report.artifacts[0].format == "csv"


def _csv_only_registry() -> ExportRegistry:
    registry = ExportRegistry()
    registry.register("csv", CSVExporter)
    return registry


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
