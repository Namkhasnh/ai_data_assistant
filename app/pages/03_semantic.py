from __future__ import annotations

from app.bootstrap import ensure_project_root

ensure_project_root()

from app.components.warning_panel import render_warning_panel
from app.controllers.pipeline_controller import PipelineController
from app.controllers.session_controller import SessionController
from app.controllers.workspace_controller import WorkspaceController


def main() -> None:
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="Semantic Detection", layout="wide")
    st.title("Semantic Detection")

    session = SessionController.from_streamlit()
    workspace = WorkspaceController(session)
    pipeline = PipelineController(session=session, workspace=workspace)

    if st.button("Run Semantic Detection", type="primary"):
        try:
            pipeline.run_semantic()
            st.success("Semantic detection completed")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Semantic detection failed: {exc}")

    report = session.get("semantic_report")
    if report is not None:
        rows = [
            {
                "column": tag.column_name,
                "semantic_type": tag.semantic_type,
                "confidence": tag.confidence,
                "detector": tag.detector_name,
                "evidence": "; ".join(tag.evidence),
            }
            for tag in report.columns
        ]
        dataframe = pd.DataFrame(rows)
        semantic_filter = st.multiselect(
            "Semantic type",
            options=sorted(dataframe["semantic_type"].unique()) if not dataframe.empty else [],
        )
        if semantic_filter:
            dataframe = dataframe[dataframe["semantic_type"].isin(semantic_filter)]
        st.dataframe(dataframe.sort_values(["semantic_type", "column"]), use_container_width=True)
    else:
        st.info("Run semantic detection after profiling.")
    render_warning_panel(session.get("warnings"))


if __name__ == "__main__":
    main()
