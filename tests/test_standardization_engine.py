from __future__ import annotations

import pandas as pd

from core.standardization.standardization_engine import StandardizationEngine
from models.semantic_tag import SemanticDetectionReport, SemanticTag


def test_standardization_engine_applies_configured_standardizers_sequentially_without_mutation() -> None:
    dataframe = pd.DataFrame(
        {
            "salary": [" Thoả thuận ", "10 - 20 triệu"],
            "experience_years": [1, 5],
        }
    )
    original = dataframe.copy(deep=True)
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
                column_name="experience_years",
                semantic_type="EXPERIENCE",
                confidence=0.9,
                detector_name="test",
            ),
        ],
    )
    config = {
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

    standardized, report = StandardizationEngine().standardize(
        dataframe,
        semantic_report,
        config,
    )

    assert standardized["salary_type"].tolist() == [" Thoả thuận ", "10 - 20 triệu"]
    assert standardized["experience_level"].tolist() == ["Junior", "Senior"]
    assert report.total_standardized_columns == 2
    assert report.standardized_by_standardizer == {
        "categorical": 1,
        "numeric_bucket": 1,
    }
    pd.testing.assert_frame_equal(dataframe, original)


def test_standardization_engine_is_deterministic() -> None:
    dataframe = pd.DataFrame({"experience_years": [0, 3, 7]})
    semantic_report = SemanticDetectionReport(source_file="sample.csv", column_count=0)
    config = {
        "EXPERIENCE": {
            "standardizer": "numeric_bucket",
            "source_column": "experience_years",
            "output_column": "experience_level",
            "buckets": [
                {"min": 0, "max": 1, "label": "Junior"},
                {"min": 2, "max": 4, "label": "Mid"},
                {"min": 5, "label": "Senior"},
            ],
        }
    }

    first, first_report = StandardizationEngine().standardize(dataframe, semantic_report, config)
    second, second_report = StandardizationEngine().standardize(dataframe, semantic_report, config)

    pd.testing.assert_frame_equal(first, second)
    assert first_report == second_report
