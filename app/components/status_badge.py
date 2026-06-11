from __future__ import annotations

from typing import Literal


Status = Literal["SUCCESS", "WARNING", "FAILED", "RUNNING", "NOT_GENERATED"]


STATUS_LABELS: dict[Status, str] = {
    "SUCCESS": "Success",
    "WARNING": "Warning",
    "FAILED": "Failed",
    "RUNNING": "Running",
    "NOT_GENERATED": "Not generated",
}

STATUS_COLORS: dict[Status, str] = {
    "SUCCESS": "#2f6f73",
    "WARNING": "#8a6f2f",
    "FAILED": "#9f3a38",
    "RUNNING": "#5b6f9f",
    "NOT_GENERATED": "#6b7280",
}


def badge_html(status: Status) -> str:
    color = STATUS_COLORS[status]
    label = STATUS_LABELS[status]
    return (
        f'<span style="display:inline-block;padding:2px 8px;border-radius:6px;'
        f'background:{color};color:white;font-size:12px;">{label}</span>'
    )


def render_status_badge(status: Status) -> str:
    """Render and return a small status badge."""

    import streamlit as st

    html = badge_html(status)
    st.markdown(html, unsafe_allow_html=True)
    return html
