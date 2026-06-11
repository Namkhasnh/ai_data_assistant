from __future__ import annotations

from app.bootstrap import ensure_project_root

ensure_project_root()

from app.components.profile_card import render_profile_cards
from app.components.warning_panel import render_warning_panel
from app.controllers.pipeline_controller import PipelineController
from app.controllers.session_controller import SessionController
from app.controllers.workspace_controller import WorkspaceController


def main() -> None:
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="Profile", layout="wide")
    st.title("Profile")

    session = SessionController.from_streamlit()
    workspace = WorkspaceController(session)
    pipeline = PipelineController(session=session, workspace=workspace)

    if st.button("Run Profiling", type="primary"):
        try:
            pipeline.run_profile()
            st.success("Profiling completed")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Profiling failed: {exc}")

    metadata = session.get("metadata")
    if metadata is not None:
        render_profile_cards(metadata)
        rows = [
            {
                "column": column.name,
                "type": column.data_type,
                "null_count": column.null_count,
                "null_percentage": column.null_percentage,
                "unique_values": column.unique_value_count,
            }
            for column in metadata.columns
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.info("Run profiling after uploading a dataset.")
    render_warning_panel(session.get("warnings"))


if __name__ == "__main__":
    main()
