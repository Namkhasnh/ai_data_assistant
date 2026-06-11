from __future__ import annotations


def normalize_warnings(warnings: list[str] | None) -> list[str]:
    return list(dict.fromkeys(warnings or []))


def render_warning_panel(warnings: list[str] | None) -> list[str]:
    """Render warnings and return the normalized list."""

    import streamlit as st

    normalized = normalize_warnings(warnings)
    for warning in normalized:
        st.warning(warning)
    return normalized
