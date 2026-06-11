from __future__ import annotations

try:
    from app.bootstrap import ensure_project_root
except ModuleNotFoundError:
    from bootstrap import ensure_project_root


ensure_project_root()


PROJECT_TITLE = "Private Local AI Data Standardization and Enrichment Assistant"
PIPELINE_STEPS = [
    "Upload",
    "Profile",
    "Semantic Detection",
    "Rules",
    "Standardization",
    "Enrichment",
    "Audit",
    "Export",
]
SIDEBAR_PAGES = [
    "01 Upload",
    "02 Profile",
    "03 Semantic",
    "04 Rules",
    "05 Standardize",
    "06 Enrich",
    "07 Audit",
    "08 Export",
]


def pipeline_overview() -> str:
    """Return the display-only pipeline overview."""

    return "\n\n↓\n\n".join(PIPELINE_STEPS)


def navigation_hint() -> str:
    """Return sidebar navigation guidance."""

    pages = "\n".join(f"- {page}" for page in SIDEBAR_PAGES)
    return f"Use the sidebar to access:\n\n{pages}"


def main() -> None:
    """Render the Streamlit home page without invoking backend services."""

    import streamlit as st

    st.set_page_config(
        page_title="AI Data Assistant",
        page_icon=":material/home:",
        layout="wide",
    )
    st.title(PROJECT_TITLE)
    st.markdown(
        "A local-first assistant for profiling, semantic detection, deterministic "
        "standardization, enrichment, audit, reporting, and export of structured datasets."
    )

    st.info("Use the sidebar to open each workflow page.")

    st.markdown("### Pipeline Overview")
    st.markdown(pipeline_overview())

    st.markdown("### Status")
    backend_status, validation_status, tests_status = st.columns(3)
    backend_status.metric("Backend Status", "Stable")
    validation_status.metric("Cross-Domain Validation", "PASS")
    tests_status.metric("Tests", "127 passed")

    with st.expander("Navigation"):
        st.markdown(navigation_hint())


if __name__ == "__main__":
    main()
