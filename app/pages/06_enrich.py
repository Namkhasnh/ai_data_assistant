from __future__ import annotations

from app.bootstrap import ensure_project_root

ensure_project_root()

from app.components.data_preview import render_data_preview
from app.components.warning_panel import render_warning_panel
from app.controllers.pipeline_controller import PipelineController
from app.controllers.session_controller import SessionController
from app.controllers.workspace_controller import WorkspaceController


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Enrichment", layout="wide")
    st.title("Enrichment")

    session = SessionController.from_streamlit()
    workspace = WorkspaceController(session)
    pipeline = PipelineController(session=session, workspace=workspace)

    if st.button("Run Enrichment", type="primary"):
        try:
            pipeline.run_enrichment()
            st.success("Enrichment completed")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Enrichment failed: {exc}")

    enriched = session.get("enriched_df")
    if enriched is not None:
        render_data_preview(enriched)
        new_columns = [
            column
            for column in enriched.columns
            if column not in (session.get("standardized_df").columns if session.get("standardized_df") is not None else [])
        ]
        st.write("New columns:", new_columns or "No new columns")
    else:
        st.info("Run enrichment after standardization.")
    render_warning_panel(session.get("warnings"))


if __name__ == "__main__":
    main()
