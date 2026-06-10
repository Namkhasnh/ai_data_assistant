from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.standardization.mapping_standardizer import MappingStandardizer
from models.semantic_tag import SemanticDetectionReport, SemanticTag


def test_mapping_standardizer_creates_canonical_column_and_preserves_source(
    tmp_path: Path,
) -> None:
    knowledge_path = tmp_path / "provinces.json"
    knowledge_path.write_text(
        json.dumps(
            {
                "Hà Nội": {
                    "match_type": "contains",
                    "aliases": ["HN", "Ha Noi"],
                    "canonical_value": "Hà Nội",
                }
            }
        ),
        encoding="utf-8",
    )
    dataframe = pd.DataFrame({"location": ["Hà Nội, Cầu Giấy", "Ha Noi", "Unknown"]})
    original = dataframe.copy(deep=True)
    semantic_report = SemanticDetectionReport(
        source_file="sample.csv",
        column_count=1,
        columns=[
            SemanticTag(
                column_name="location",
                semantic_type="LOCATION",
                confidence=0.9,
                detector_name="test",
            )
        ],
    )
    config = {
        "LOCATION": {
            "standardizer": "mapping",
            "source_column": "location",
            "output_column": "standardized_location",
            "knowledge_file": str(knowledge_path),
        }
    }

    standardized = MappingStandardizer().standardize(dataframe, semantic_report, config)

    assert standardized["location"].tolist() == ["Hà Nội, Cầu Giấy", "Ha Noi", "Unknown"]
    assert standardized["standardized_location"].tolist()[:2] == ["Hà Nội", "Hà Nội"]
    assert pd.isna(standardized.loc[2, "standardized_location"])
    pd.testing.assert_frame_equal(dataframe, original)


def test_mapping_standardizer_generates_canonical_titles_from_matchers(
    tmp_path: Path,
) -> None:
    knowledge_path = tmp_path / "job_titles.json"
    knowledge_path.write_text(
        json.dumps(
            {
                "Data Analyst": {
                    "match_type": "contains",
                    "aliases": ["Business Data Analyst", "Data Analyst (Banking)"],
                    "canonical_value": "Data Analyst",
                    "priority": 10,
                },
                "AI Leader": {
                    "match_type": "regex",
                    "patterns": [r"\bAI\b.*\bLeader\b"],
                    "canonical_value": "AI Leader",
                    "priority": 20,
                },
            }
        ),
        encoding="utf-8",
    )
    dataframe = pd.DataFrame(
        {
            "title": [
                "Data Analyst (Banking)",
                "Business Data Analyst",
                "AI Team Leader",
                "Unknown Title",
            ]
        }
    )
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
    config = {
        "JOB_TITLE": {
            "standardizer": "mapping",
            "source_column": "title",
            "output_column": "standardized_title",
            "knowledge_file": str(knowledge_path),
        }
    }

    standardized = MappingStandardizer().standardize(dataframe, semantic_report, config)

    assert standardized["standardized_title"].tolist()[:3] == [
        "Data Analyst",
        "Data Analyst",
        "AI Leader",
    ]
    assert pd.isna(standardized.loc[3, "standardized_title"])


def test_mapping_standardizer_keeps_backward_compatibility_with_old_knowledge(
    tmp_path: Path,
) -> None:
    knowledge_path = tmp_path / "old_titles.json"
    knowledge_path.write_text(
        json.dumps(
            {
                "AI Engineer": {
                    "aliases": ["ML Engineer"],
                }
            }
        ),
        encoding="utf-8",
    )
    dataframe = pd.DataFrame({"title": ["ML Engineer", "Senior ML Engineer"]})
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
    config = {
        "JOB_TITLE": {
            "standardizer": "mapping",
            "source_column": "title",
            "output_column": "standardized_title",
            "knowledge_file": str(knowledge_path),
        }
    }

    standardized = MappingStandardizer().standardize(dataframe, semantic_report, config)

    assert standardized.loc[0, "standardized_title"] == "AI Engineer"
    assert pd.isna(standardized.loc[1, "standardized_title"])
