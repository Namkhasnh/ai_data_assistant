from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.exporting.csv_exporter import CSVExporter
from core.exporting.export_registry import ExportRegistry
from models.knowledge_package import KnowledgePackageReport, KnowledgePackageResult
from models.semantic_dataset import SemanticDataset
from services.export_service import ExportService
from services.semantic_dataset_service import SemanticDatasetService
from storage.artifact_store import ArtifactStore


def test_semantic_dataset_generation_is_append_only_read_only_and_deterministic(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    fixed_time = datetime(2026, 6, 10, 12, 0, 0)
    knowledge_result = _knowledge_result()
    original_dataframe = knowledge_result.dataframe.copy(deep=True)

    semantic_dataset = SemanticDatasetService(
        artifact_store=ArtifactStore(artifact_dir)
    ).generate(
        knowledge_package_result=knowledge_result,
        generated_at=fixed_time,
    )

    assert isinstance(semantic_dataset, SemanticDataset)
    assert list(semantic_dataset.dataframe.columns) == [
        "title",
        "location",
        "salary_min",
        "job_group",
        "city",
        "salary_avg",
        "department",
        "test_name",
    ]
    assert semantic_dataset.report.source_columns == ["title", "location", "salary_min"]
    assert semantic_dataset.report.semantic_columns == [
        "job_group",
        "city",
        "salary_avg",
        "department",
        "test_name",
    ]
    assert semantic_dataset.report.source_column_count == 3
    assert semantic_dataset.report.semantic_column_count == 5
    assert semantic_dataset.report.total_columns == 8
    assert semantic_dataset.report.warnings == ["package warning"]
    pd.testing.assert_frame_equal(knowledge_result.dataframe, original_dataframe)
    assert (artifact_dir / "semantic_dataset.csv").exists()
    assert (artifact_dir / "semantic_dataset_report.json").exists()


def test_semantic_dataset_artifacts_are_byte_for_byte_deterministic(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    service = SemanticDatasetService(artifact_store=ArtifactStore(artifact_dir))
    fixed_time = datetime(2026, 6, 10, 12, 0, 0)
    knowledge_result = _knowledge_result()

    service.generate(knowledge_result, generated_at=fixed_time)
    first_csv_hash = _sha256(artifact_dir / "semantic_dataset.csv")
    first_report_hash = _sha256(artifact_dir / "semantic_dataset_report.json")

    service.generate(knowledge_result, generated_at=fixed_time)
    assert _sha256(artifact_dir / "semantic_dataset.csv") == first_csv_hash
    assert _sha256(artifact_dir / "semantic_dataset_report.json") == first_report_hash


def test_semantic_dataset_generation_preserves_upstream_artifact_hashes(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    business_path = artifact_dir / "business_dataset.csv"
    enriched_path = artifact_dir / "enriched_dataset.csv"
    standardized_path = artifact_dir / "standardized_dataset.csv"
    pd.DataFrame({"title": ["business"]}).to_csv(business_path, index=False)
    pd.DataFrame({"title": ["technical"]}).to_csv(enriched_path, index=False)
    pd.DataFrame({"title": ["standardized"]}).to_csv(standardized_path, index=False)
    before_hashes = {
        path: _sha256(path) for path in [business_path, enriched_path, standardized_path]
    }

    SemanticDatasetService(artifact_store=ArtifactStore(artifact_dir)).generate(
        _knowledge_result(),
        generated_at=datetime(2026, 6, 10, 12, 0, 0),
    )

    assert before_hashes == {
        path: _sha256(path) for path in [business_path, enriched_path, standardized_path]
    }


def test_semantic_dataset_service_does_not_execute_packages() -> None:
    service_source = Path("services/semantic_dataset_service.py").read_text(encoding="utf-8")

    assert "core.knowledge_packages" not in service_source
    assert "KnowledgePackageEngine" not in service_source
    assert ".apply_packages(" not in service_source


def test_export_service_prefers_semantic_dataset_when_present(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    export_dir = tmp_path / "exports"
    artifact_dir.mkdir()
    pd.DataFrame({"title": ["semantic"], "job_group": ["AI Engineer"]}).to_csv(
        artifact_dir / "semantic_dataset.csv",
        index=False,
    )
    pd.DataFrame({"title": ["business"]}).to_csv(
        artifact_dir / "business_dataset.csv",
        index=False,
    )
    pd.DataFrame({"title": ["enriched"]}).to_csv(
        artifact_dir / "enriched_dataset.csv",
        index=False,
    )
    pd.DataFrame({"title": ["standardized"]}).to_csv(
        artifact_dir / "standardized_dataset.csv",
        index=False,
    )

    ExportService(
        artifact_dir=artifact_dir,
        audit_dir=tmp_path / "audit",
        report_dir=tmp_path / "reports",
        export_dir=export_dir,
        registry=_csv_only_registry(),
    ).export_all()

    exported = pd.read_csv(export_dir / "csv" / "export_dataset.csv")
    assert list(exported.columns) == ["title", "job_group"]
    assert exported.loc[0, "title"] == "semantic"


def _knowledge_result() -> KnowledgePackageResult:
    return KnowledgePackageResult(
        dataframe=pd.DataFrame(
            {
                "title": ["Data Analyst"],
                "job_group": ["Analytics"],
                "location": ["Hà Nội"],
                "city": ["Hà Nội"],
                "salary_avg": [25],
                "salary_min": [20],
                "department": ["Laboratory"],
                "test_name": ["Complete Blood Count"],
            }
        ),
        report=KnowledgePackageReport(
            warnings=["package warning"],
            produced_columns=[
                "job_group",
                "city",
                "salary_avg",
                "department",
                "test_name",
            ],
            produced_columns_by_package={
                "job_title": ["job_group"],
                "location": ["city"],
                "salary": ["salary_avg"],
                "healthcare_test": ["department", "test_name"],
            },
        ),
    )


def _csv_only_registry() -> ExportRegistry:
    registry = ExportRegistry()
    registry.register("csv", CSVExporter)
    return registry


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
