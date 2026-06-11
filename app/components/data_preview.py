from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DataPreviewSummary:
    """Small dataframe summary used by upload and result pages."""

    row_count: int
    column_count: int
    columns: list[str]


def build_data_preview(dataframe: pd.DataFrame) -> DataPreviewSummary:
    return DataPreviewSummary(
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        columns=list(dataframe.columns),
    )


def render_data_preview(dataframe: pd.DataFrame, rows: int = 20) -> DataPreviewSummary:
    """Render a dataframe preview and return its summary."""

    import streamlit as st

    summary = build_data_preview(dataframe)
    col_rows, col_columns = st.columns(2)
    col_rows.metric("Rows", summary.row_count)
    col_columns.metric("Columns", summary.column_count)
    st.dataframe(dataframe.head(rows), use_container_width=True)
    return summary
