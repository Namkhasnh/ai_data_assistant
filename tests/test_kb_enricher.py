from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.enrichment.kb_enricher import KBEnricher
from models.enrichment import EnrichmentConfigItem
from models.semantic_tag import SemanticDetectionReport


def test_kb_enricher_adds_configured_columns_and_preserves_unknowns(
    tmp_path: Path,
) -> None:
    knowledge_path = tmp_path / "job_families.json"
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
    dataframe = pd.DataFrame({"title": ["Data Engineer", "Unknown Title"]})
    original = dataframe.copy(deep=True)
    config_item = EnrichmentConfigItem(
        enricher_id="test_title_family_001",
        enricher="knowledge",
        source_column="title",
        knowledge_file=str(knowledge_path),
        output_columns=["job_family", "job_domain"],
    )

    enriched = KBEnricher().enrich(
        dataframe,
        SemanticDetectionReport(source_file="sample.csv", column_count=0),
        config_item,
    )

    assert enriched["title"].tolist() == ["Data Engineer", "Unknown Title"]
    assert enriched.loc[0, "job_family"] == "Engineering"
    assert enriched.loc[0, "job_domain"] == "Data"
    assert pd.isna(enriched.loc[1, "job_family"])
    assert pd.isna(enriched.loc[1, "job_domain"])
    pd.testing.assert_frame_equal(dataframe, original)


def test_kb_enricher_supports_contains_alias_regex_and_composite_values(
    tmp_path: Path,
) -> None:
    knowledge_path = tmp_path / "knowledge.json"
    knowledge_path.write_text(
        json.dumps(
            {
                "Data Analyst": {
                    "match_type": "contains",
                    "aliases": ["Chuyên viên phân tích dữ liệu"],
                    "outputs": {
                        "job_family": "Analytics",
                        "job_domain": "Data",
                    },
                    "priority": 10,
                },
                "AI Engineer": {
                    "match_type": "alias",
                    "aliases": ["ML Engineer"],
                    "outputs": {
                        "job_family": "AI",
                        "job_domain": "Data",
                    },
                    "priority": 20,
                },
                "AI Leadership": {
                    "match_type": "regex",
                    "patterns": [r"\bAI\b.*\bLeader\b"],
                    "outputs": {
                        "job_family": "AI",
                        "job_domain": "Data",
                    },
                    "priority": 30,
                },
                "Hà Nội": {
                    "match_type": "contains",
                    "aliases": ["Ha Noi", "Hanoi"],
                    "outputs": {
                        "job_family": "North",
                        "job_domain": "Vietnam",
                    },
                    "priority": 40,
                },
            }
        ),
        encoding="utf-8",
    )
    dataframe = pd.DataFrame(
        {
            "value": [
                "Senior Data Analyst",
                "ML Engineer",
                "AI Team Leader",
                "Hà Nội, Phường Vĩnh Tuy",
                "Unknown",
            ]
        }
    )
    config_item = EnrichmentConfigItem(
        enricher_id="test_flexible_001",
        enricher="knowledge",
        source_column="value",
        knowledge_file=str(knowledge_path),
        output_columns=["job_family", "job_domain"],
    )

    enriched = KBEnricher().enrich(
        dataframe,
        SemanticDetectionReport(source_file="sample.csv", column_count=0),
        config_item,
    )

    assert enriched["job_family"].tolist()[:4] == [
        "Analytics",
        "AI",
        "AI",
        "North",
    ]
    assert enriched["job_domain"].tolist()[:4] == [
        "Data",
        "Data",
        "Data",
        "Vietnam",
    ]
    assert pd.isna(enriched.loc[4, "job_family"])
    assert pd.isna(enriched.loc[4, "job_domain"])


def test_kb_enricher_keeps_backward_compatibility_with_flat_knowledge(
    tmp_path: Path,
) -> None:
    knowledge_path = tmp_path / "old_regions.json"
    knowledge_path.write_text(
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
    dataframe = pd.DataFrame({"location": ["Hà Nội", "Hà Nội, Cầu Giấy"]})
    config_item = EnrichmentConfigItem(
        enricher_id="test_old_format_001",
        enricher="knowledge",
        source_column="location",
        knowledge_file=str(knowledge_path),
        output_columns=["region", "country"],
    )

    enriched = KBEnricher().enrich(
        dataframe,
        SemanticDetectionReport(source_file="sample.csv", column_count=0),
        config_item,
    )

    assert enriched.loc[0, "region"] == "North"
    assert pd.isna(enriched.loc[1, "region"])


def test_kb_enricher_does_not_overwrite_existing_columns(tmp_path: Path) -> None:
    knowledge_path = tmp_path / "job_families.json"
    knowledge_path.write_text(
        json.dumps(
            {
                "Data Engineer": {
                    "job_family": "Engineering",
                }
            }
        ),
        encoding="utf-8",
    )
    dataframe = pd.DataFrame(
        {
            "title": ["Data Engineer"],
            "job_family": ["Existing"],
        }
    )
    config_item = EnrichmentConfigItem(
        enricher_id="test_title_family_001",
        enricher="knowledge",
        source_column="title",
        knowledge_file=str(knowledge_path),
        output_columns=["job_family"],
    )
    enricher = KBEnricher()

    enriched = enricher.enrich(
        dataframe,
        SemanticDetectionReport(source_file="sample.csv", column_count=0),
        config_item,
    )

    assert enriched["job_family"].tolist() == ["Existing"]
    assert enricher.enriched_columns == set()
    assert enricher.warnings
