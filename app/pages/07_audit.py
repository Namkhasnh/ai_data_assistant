from __future__ import annotations

from app.bootstrap import ensure_project_root

ensure_project_root()

from app.components.status_badge import render_status_badge
from app.components.warning_panel import render_warning_panel
from app.controllers.artifact_controller import ArtifactController
from app.controllers.pipeline_controller import PipelineController
from app.controllers.session_controller import SessionController
from app.controllers.workspace_controller import WorkspaceController


def main() -> None:
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="Audit", layout="wide")
    st.title("Audit")

    session = SessionController.from_streamlit()
    workspace = WorkspaceController(session)
    pipeline = PipelineController(session=session, workspace=workspace)
    artifacts = ArtifactController(
        artifact_dir=workspace.artifact_dir,
        audit_dir=workspace.audit_dir,
        report_dir=workspace.report_dir,
        export_dir=workspace.export_dir,
    )

    if st.button("Generate Audit", type="primary"):
        try:
            pipeline.run_audit()
            st.success("Audit generated")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Audit failed: {exc}")

    audit_path = workspace.audit_dir / "audit_report.json"
    render_status_badge("SUCCESS" if artifacts.artifact_exists(audit_path) else "NOT_GENERATED")
    payload, warning = artifacts.read_json(audit_path)
    if warning:
        st.warning(warning)
    if payload:
        render_warning_panel(payload.get("warnings", []))
        st.subheader("Artifacts")
        st.dataframe(pd.DataFrame(payload.get("artifacts", [])), use_container_width=True)
        st.subheader("Rule History")
        st.dataframe(
            pd.DataFrame(payload.get("rule_history", {}).get("records", [])),
            use_container_width=True,
        )
    render_warning_panel(session.get("warnings"))


if __name__ == "__main__":
    main()
