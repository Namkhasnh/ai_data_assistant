from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.enrichment.enrichment_engine import EnrichmentEngine
from models.enrichment import EnrichmentConfig, EnrichmentConfigItem
from models.semantic_tag import SemanticDetectionReport, SemanticTag


def test_enrichment_engine_is_deterministic_and_does_not_mutate_input(
    tmp_path: Path,
) -> None:
    job_knowledge = tmp_path / "job_families.json"
    region_knowledge = tmp_path / "regions.json"
    job_knowledge.write_text(
        json.dumps(
            {
                "Data Analyst": {
                    "job_family": "Analytics",
                    "job_domain": "Data",
                }
            }
        ),
        encoding="utf-8",
    )
    region_knowledge.write_text(
        json.dumps(
            {
                "Hà Nội": {
                    "region": "North",
                    "country": "Vietnam",
                }
            }
        ),
        encoding="utf-8",
    )
    dataframe = pd.DataFrame(
        {
            "title": ["Data Analyst", "Unknown Title"],
            "location": ["Hà Nội", "Unknown Location"],
            "salary": ["Negotiable", "10 - 20 triệu"],
        }
    )
    original = dataframe.copy(deep=True)
    semantic_report = SemanticDetectionReport(
        source_file="sample.csv",
        column_count=3,
        columns=[
            SemanticTag(
                column_name="title",
                semantic_type="JOB_TITLE",
                confidence=0.9,
                detector_name="test",
            ),
            SemanticTag(
                column_name="location",
                semantic_type="LOCATION",
                confidence=0.9,
                detector_name="test",
            ),
            SemanticTag(
                column_name="salary",
                semantic_type="SALARY",
                confidence=0.9,
                detector_name="test",
            ),
        ],
    )
    config = EnrichmentConfig(
        enrichments=[
            EnrichmentConfigItem(
                enricher_id="test_title_family_001",
                enricher="knowledge",
                source_column="title",
                knowledge_file=str(job_knowledge),
                output_columns=["job_family", "job_domain"],
                priority=20,
            ),
            EnrichmentConfigItem(
                enricher_id="test_location_region_001",
                enricher="knowledge",
                source_column="location",
                knowledge_file=str(region_knowledge),
                output_columns=["region", "country"],
                priority=10,
            ),
        ]
    )
    engine = EnrichmentEngine()

    first, first_report = engine.enrich(dataframe, semantic_report, config)
    second, second_report = engine.enrich(dataframe, semantic_report, config)

    pd.testing.assert_frame_equal(first, second)
    assert first_report == second_report
    pd.testing.assert_frame_equal(dataframe, original)
    assert first.loc[0, "job_family"] == "Analytics"
    assert first.loc[0, "region"] == "North"
    assert pd.isna(first.loc[1, "job_family"])
    assert pd.isna(first.loc[1, "region"])
    assert first_report.total_enriched_columns == 4
    assert first_report.enriched_by_enricher == {"knowledge": 4}
    assert first_report.skipped_columns == ["salary"]
