from __future__ import annotations

import pandas as pd

from core.rules.validation_rule import ValidationRule
from models.rule import Rule


def test_validation_rule_adds_namespaced_columns_without_removing_rows() -> None:
    dataframe = pd.DataFrame({"salary": [10, 0, -1]})
    rule = Rule(
        id="salary_positive",
        type="validation",
        column="salary",
        parameters={
            "operator": ">",
            "value": 0,
            "message": "salary must be positive",
        },
    )

    cleaned = ValidationRule().apply(dataframe, rule)

    assert len(cleaned) == 3
    assert cleaned["salary__is_valid"].tolist() == [True, False, False]
    assert cleaned["salary__validation_error"].tolist() == [
        "",
        "salary must be positive",
        "salary must be positive",
    ]
    assert "salary__is_valid" not in dataframe.columns


def test_validation_rule_avoids_column_name_collisions() -> None:
    dataframe = pd.DataFrame(
        {
            "email": ["a@example.com", "bad"],
            "email__is_valid": [True, True],
            "email__validation_error": ["", ""],
        }
    )
    rule = Rule(
        id="email_format",
        type="validation",
        column="email",
        parameters={
            "format": "email",
            "message": "email is invalid",
        },
    )

    cleaned = ValidationRule().apply(dataframe, rule)

    assert "email__email_format__is_valid" in cleaned.columns
    assert "email__email_format__validation_error" in cleaned.columns
    assert cleaned["email__email_format__is_valid"].tolist() == [True, False]


def test_validation_rule_treats_missing_comparison_values_as_invalid() -> None:
    dataframe = pd.DataFrame({"salary_min": pd.Series([10, pd.NA], dtype="Int64")})
    rule = Rule(
        id="salary_min_positive",
        type="validation",
        column="salary_min",
        parameters={
            "operator": ">",
            "value": 0,
        },
    )

    cleaned = ValidationRule().apply(dataframe, rule)

    assert cleaned["salary_min__is_valid"].tolist() == [True, False]


def test_validation_rule_allows_null_when_configured() -> None:
    dataframe = pd.DataFrame({"salary_min": pd.Series([10, pd.NA, 0], dtype="Int64")})
    rule = Rule(
        id="salary_min_positive",
        type="validation",
        column="salary_min",
        parameters={
            "operator": ">",
            "value": 0,
            "allow_null": True,
        },
    )

    cleaned = ValidationRule().apply(dataframe, rule)

    assert cleaned["salary_min__is_valid"].tolist() == [True, True, False]
    assert cleaned["salary_min__validation_error"].tolist() == [
        "",
        "",
        "salary_min failed validation",
    ]
