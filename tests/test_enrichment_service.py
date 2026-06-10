from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from models.semantic_tag import SemanticDetectionReport, SemanticTag
from services.enrichment_service import EnrichmentService
from storage.artifact_store import ArtifactStore


def test_enrichment_service_loads_inputs_and_writes_artifacts(tmp_path: Path) -> None:
    input_csv = tmp_path / "standardized.csv"
    semantic_path = tmp_path / "semantic_columns.json"
    config_path = tmp_path / "enrichment_rules.json"
    knowledge_path = tmp_path / "job_families.json"
    dataframe = pd.DataFrame(
        {
            "title": ["Data Engineer", "Unknown Title"],
            "standardized_title": ["Data Engineer", None],
            "standardized_location": ["Hà Nội", None],
        }
    )
    dataframe.to_csv(input_csv, index=False)
    semantic_report = SemanticDetectionReport(
        source_file="sample.csv",
        column_count=1,
        columns=[
            SemanticTag(
                column_name="title",
                semantic_type="JOB_TITLE",
                confidence=0.9,
                detector_name="test",
            )
        ],
    )
    semantic_path.write_text(semantic_report.model_dump_json(), encoding="utf-8")
    knowledge_path.write_text(
        json.dumps(
            {
                "Data Engineer": {
                    "job_family": "Engineering",
                    "job_domain": "Data",
                }
            }
        ),
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(
            {
                "enrichments": [
                    {
                        "enricher_id": "test_title_family_001",
                        "enricher": "knowledge",
                        "source_column": "title",
                        "knowledge_file": str(knowledge_path),
                        "output_columns": ["job_family", "job_domain"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    enriched, report = EnrichmentService(
        artifact_store=ArtifactStore(artifact_dir=tmp_path / "artifacts")
    ).enrich_csv(
        input_csv_path=input_csv,
        semantic_report_path=semantic_path,
        config_path=config_path,
    )

    assert enriched.loc[0, "job_family"] == "Engineering"
    assert pd.isna(enriched.loc[1, "job_family"])
    assert "standardized_title" in enriched.columns
    assert "standardized_location" in enriched.columns
    assert enriched.loc[0, "standardized_title"] == "Data Engineer"
    assert enriched.loc[0, "standardized_location"] == "Hà Nội"
    assert list(enriched.columns) == [
        "title",
        "standardized_title",
        "standardized_location",
        "job_family",
        "job_domain",
    ]
    assert report.total_enriched_columns == 2
    assert (tmp_path / "artifacts" / "enriched_dataset.csv").exists()
    assert (tmp_path / "artifacts" / "enrichment_report.json").exists()
