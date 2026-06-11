from __future__ import annotations

import pandas as pd


def build_comparison_table(
    before: pd.DataFrame,
    after: pd.DataFrame,
    max_rows: int = 20,
) -> pd.DataFrame:
    """Build a compact before/after table for columns whose values changed."""

    rows: list[dict[str, object]] = []
    shared_columns = [column for column in before.columns if column in after.columns]
    for index in before.index.intersection(after.index):
        for column in shared_columns:
            before_value = before.at[index, column]
            after_value = after.at[index, column]
            if pd.isna(before_value) and pd.isna(after_value):
                continue
            if before_value == after_value:
                continue
            rows.append(
                {
                    "row": index,
                    "column": column,
                    "before": before_value,
                    "after": after_value,
                }
            )
            if len(rows) >= max_rows:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows, columns=["row", "column", "before", "after"])


def render_comparison_table(
    before: pd.DataFrame,
    after: pd.DataFrame,
    max_rows: int = 20,
) -> pd.DataFrame:
    """Render a comparison table and return the displayed data."""

    import streamlit as st

    comparison = build_comparison_table(before, after, max_rows=max_rows)
    st.dataframe(comparison, use_container_width=True)
    return comparison
