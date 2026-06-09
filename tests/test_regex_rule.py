from __future__ import annotations

import pandas as pd

from core.rules.regex_rule import RegexRule
from models.rule import Rule


def test_regex_rule_extracts_capture_groups_with_output_types() -> None:
    dataframe = pd.DataFrame(
        {
            "salary": ["15-25 triệu", "3-5", "Thoả thuận"],
        }
    )
    rule = Rule(
        id="extract_salary",
        type="regex",
        column="salary",
        parameters={
            "pattern": r"(\d+)\s*-\s*(\d+)",
            "output_columns": ["salary_min", "salary_max"],
            "output_types": {
                "salary_min": "int",
                "salary_max": "int",
            },
        },
    )

    cleaned = RegexRule().apply(dataframe, rule)

    assert cleaned["salary_min"].tolist()[:2] == [15, 3]
    assert cleaned["salary_max"].tolist()[:2] == [25, 5]
    assert pd.isna(cleaned.loc[2, "salary_min"])
    assert str(cleaned["salary_min"].dtype) == "Int64"
    assert "salary_min" not in dataframe.columns
