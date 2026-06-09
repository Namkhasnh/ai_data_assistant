from __future__ import annotations

import pandas as pd

from core.rules.transformation_rule import TransformationRule
from models.rule import Rule


def test_transformation_rule_remove_extra_spaces() -> None:
    dataframe = pd.DataFrame({"name": ["  Alice   Nguyen  ", "Bob"]})
    rule = Rule(
        id="clean_name",
        type="transformation",
        column="name",
        parameters={"operation": "remove_extra_spaces"},
    )

    cleaned = TransformationRule().apply(dataframe, rule)

    assert cleaned["name"].tolist() == ["Alice Nguyen", "Bob"]
    assert dataframe["name"].tolist() == ["  Alice   Nguyen  ", "Bob"]


def test_transformation_rule_replace_to_output_column() -> None:
    dataframe = pd.DataFrame({"code": ["A-001", "A-002"]})
    rule = Rule(
        id="replace_prefix",
        type="transformation",
        column="code",
        parameters={
            "operation": "replace",
            "old": "A-",
            "new": "B-",
            "output_column": "normalized_code",
        },
    )

    cleaned = TransformationRule().apply(dataframe, rule)

    assert cleaned["code"].tolist() == ["A-001", "A-002"]
    assert cleaned["normalized_code"].tolist() == ["B-001", "B-002"]
