from __future__ import annotations

from app.bootstrap import ensure_project_root

ensure_project_root()

from app.components.comparison_table import render_comparison_table
from app.components.data_preview import render_data_preview
from app.components.warning_panel import render_warning_panel
from app.controllers.pipeline_controller import PipelineController
from app.controllers.session_controller import SessionController
from app.controllers.workspace_controller import WorkspaceController


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Standardization", layout="wide")
    st.title("Standardization")

    session = SessionController.from_streamlit()
    workspace = WorkspaceController(session)
    pipeline = PipelineController(session=session, workspace=workspace)

    if st.button("Run Standardization", type="primary"):
        try:
            pipeline.run_standardization()
            st.success("Standardization completed")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Standardization failed: {exc}")

    standardized = session.get("standardized_df")
    uploaded = session.get("uploaded_df")
    if standardized is not None:
        render_data_preview(standardized)
        if uploaded is not None:
            with st.expander("Before vs After", expanded=True):
                render_comparison_table(uploaded, standardized)
    else:
        st.info("Run standardization after rules have executed.")
    render_warning_panel(session.get("warnings"))


if __name__ == "__main__":
    main()
