from __future__ import annotations

import pandas as pd

from core.rules.mapping_rule import MappingRule
from models.rule import Rule


def test_mapping_rule_normalizes_values_case_insensitively_without_mutation() -> None:
    dataframe = pd.DataFrame(
        {
            "location": ["HN", "ha noi", "Da Nang", None],
        }
    )
    original = dataframe.copy(deep=True)
    rule = Rule(
        id="normalize_location",
        type="mapping",
        column="location",
        parameters={
            "mapping": {
                "HN": "Hanoi",
                "Ha Noi": "Hanoi",
            },
            "case_sensitive": False,
        },
    )

    cleaned = MappingRule().apply(dataframe, rule)

    assert cleaned["location"].tolist()[:3] == ["Hanoi", "Hanoi", "Da Nang"]
    assert pd.isna(cleaned.loc[3, "location"])
    pd.testing.assert_frame_equal(dataframe, original)
