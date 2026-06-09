from __future__ import annotations

import pandas as pd

from core.standardization.numeric_standardizer import NumericStandardizer
from models.semantic_tag import SemanticDetectionReport


def test_numeric_standardizer_buckets_values_from_config_without_mutation() -> None:
    dataframe = pd.DataFrame({"experience_years": [0, 1, 2, 4, 5, None]})
    original = dataframe.copy(deep=True)
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

    standardized = NumericStandardizer().standardize(
        dataframe,
        SemanticDetectionReport(source_file="sample.csv", column_count=0),
        config,
    )

    assert standardized["experience_level"].tolist()[:5] == [
        "Junior",
        "Junior",
        "Mid",
        "Mid",
        "Senior",
    ]
    assert pd.isna(standardized.loc[5, "experience_level"])
    pd.testing.assert_frame_equal(dataframe, original)
