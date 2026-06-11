from __future__ import annotations

import pandas as pd

from app.components.data_preview import build_data_preview
from app.components.comparison_table import build_comparison_table


def test_data_preview_summary() -> None:
    dataframe = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})

    summary = build_data_preview(dataframe)

    assert summary.row_count == 2
    assert summary.column_count == 2
    assert summary.columns == ["a", "b"]


def test_comparison_table_only_includes_changed_values() -> None:
    before = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    after = pd.DataFrame({"a": [1, 3], "b": ["x", "z"]})

    comparison = build_comparison_table(before, after)

    assert comparison.to_dict(orient="records") == [
        {"row": 1, "column": "a", "before": 2, "after": 3},
        {"row": 1, "column": "b", "before": "y", "after": "z"},
    ]
