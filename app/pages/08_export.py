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
    import streamlit as st

    st.set_page_config(page_title="Export", layout="wide")
    st.title("Export")

    session = SessionController.from_streamlit()
    workspace = WorkspaceController(session)
    pipeline = PipelineController(session=session, workspace=workspace)
    artifacts = ArtifactController(
        artifact_dir=workspace.artifact_dir,
        audit_dir=workspace.audit_dir,
        report_dir=workspace.report_dir,
        export_dir=workspace.export_dir,
    )

    if st.button("Regenerate Exports", type="primary"):
        try:
            pipeline.run_export()
            st.success("Exports regenerated")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Export failed: {exc}")

    exports = artifacts.list_exports()
    render_status_badge("SUCCESS" if exports else "NOT_GENERATED")
    if exports:
        for artifact in exports:
            with artifact.path.open("rb") as file:
                st.download_button(
                    label=f"Download {artifact.name}",
                    data=file.read(),
                    file_name=artifact.name,
                )
    else:
        st.info("No exports are available. Use Regenerate Exports to create them.")

    export_report, warning = artifacts.read_json(workspace.export_dir / "export_report.json")
    if warning:
        st.warning(warning)
    elif export_report:
        render_warning_panel(export_report.get("warnings", []))
    render_warning_panel(session.get("warnings"))


if __name__ == "__main__":
    main()
