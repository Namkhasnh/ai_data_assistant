from __future__ import annotations

import pandas as pd

from core.standardization.categorical_standardizer import CategoricalStandardizer
from models.semantic_tag import SemanticDetectionReport, SemanticTag


def test_categorical_standardizer_normalizes_categories_and_preserves_unknowns() -> None:
    dataframe = pd.DataFrame({"salary": ["Thoả thuận", "Negotiable", "10 - 20 triệu"]})
    semantic_report = SemanticDetectionReport(
        source_file="sample.csv",
        column_count=1,
        columns=[
            SemanticTag(
                column_name="salary",
                semantic_type="SALARY",
                confidence=0.9,
                detector_name="test",
            )
        ],
    )
    config = {
        "SALARY": {
            "standardizer": "categorical",
            "output_column": "salary_type",
            "categories": {
                "Thoả thuận": "negotiable",
                "Negotiable": "negotiable",
            },
        }
    }

    standardized = CategoricalStandardizer().standardize(dataframe, semantic_report, config)

    assert standardized["salary_type"].tolist() == [
        "negotiable",
        "negotiable",
        "10 - 20 triệu",
    ]
    assert "salary_type" not in dataframe.columns
