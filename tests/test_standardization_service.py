from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from models.semantic_tag import SemanticDetectionReport, SemanticTag
from services.standardization_service import StandardizationService
from storage.artifact_store import ArtifactStore


def test_standardization_service_loads_inputs_and_writes_artifacts(tmp_path: Path) -> None:
    input_csv = tmp_path / "cleaned.csv"
    semantic_path = tmp_path / "semantic_columns.json"
    config_path = tmp_path / "standardization_rules.json"
    dataframe = pd.DataFrame(
        {
            "salary": ["Thoả thuận", "10 - 20 triệu"],
            "experience_years": [1, 5],
        }
    )
    dataframe.to_csv(input_csv, index=False)
    semantic_report = SemanticDetectionReport(
        source_file="sample.csv",
        column_count=2,
        columns=[
            SemanticTag(
                column_name="salary",
                semantic_type="SALARY",
                confidence=0.9,
                detector_name="test",
            ),
            SemanticTag(
                column_name="experience",
                semantic_type="EXPERIENCE",
                confidence=0.9,
                detector_name="test",
            ),
        ],
    )
    semantic_path.write_text(semantic_report.model_dump_json(), encoding="utf-8")
    config_path.write_text(
        json.dumps(
            {
                "SALARY": {
                    "standardizer": "categorical",
                    "output_column": "salary_type",
                    "categories": {
                        "Thoả thuận": "negotiable",
                    },
                },
                "EXPERIENCE": {
                    "standardizer": "numeric_bucket",
                    "source_column": "experience_years",
                    "output_column": "experience_level",
                    "buckets": [
                        {"min": 0, "max": 1, "label": "Junior"},
                        {"min": 5, "label": "Senior"},
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    standardized, report = StandardizationService(
        artifact_store=ArtifactStore(artifact_dir=tmp_path / "artifacts")
    ).standardize_csv(
        input_csv_path=input_csv,
        semantic_report_path=semantic_path,
        config_path=config_path,
    )

    assert standardized["salary_type"].tolist() == ["negotiable", "10 - 20 triệu"]
    assert standardized["experience_level"].tolist() == ["Junior", "Senior"]
    assert report.total_standardized_columns == 2
    assert (tmp_path / "artifacts" / "standardized_dataset.csv").exists()
    assert (tmp_path / "artifacts" / "standardization_report.json").exists()
