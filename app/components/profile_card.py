from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ProfileMetrics:
    """Dataset-level metrics displayed by the profile page."""

    row_count: int
    column_count: int
    duplicate_count: int
    missing_values: int


def build_profile_metrics(metadata: Any) -> ProfileMetrics:
    missing_values = sum(column.null_count for column in getattr(metadata, "columns", []))
    return ProfileMetrics(
        row_count=int(getattr(metadata, "row_count", 0)),
        column_count=int(getattr(metadata, "column_count", 0)),
        duplicate_count=int(getattr(metadata, "duplicate_count", 0)),
        missing_values=missing_values,
    )


def render_profile_cards(metadata: Any) -> ProfileMetrics:
    """Render profile metric cards and return the computed metrics."""

    import streamlit as st

    metrics = build_profile_metrics(metadata)
    row_count, column_count, duplicate_count, missing_values = st.columns(4)
    row_count.metric("Rows", metrics.row_count)
    column_count.metric("Columns", metrics.column_count)
    duplicate_count.metric("Duplicates", metrics.duplicate_count)
    missing_values.metric("Missing Values", metrics.missing_values)
    return metrics
