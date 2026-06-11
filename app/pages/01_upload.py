from __future__ import annotations

from app.bootstrap import ensure_project_root

ensure_project_root()

from app.components.data_preview import render_data_preview
from app.components.warning_panel import render_warning_panel
from app.controllers.session_controller import SessionController
from app.controllers.workspace_controller import WorkspaceController


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Upload Dataset", layout="wide")
    st.title("Upload Dataset")

    session = SessionController.from_streamlit()
    workspace = WorkspaceController(session)

    uploaded_file = st.file_uploader("Dataset", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        try:
            file_path = workspace.save_upload(uploaded_file, uploaded_file.name)
            dataframe = workspace.load_dataframe(file_path)
            session.clear_run_outputs()
            session.set_uploaded_dataframe(dataframe, str(file_path))
            st.success("Dataset uploaded")
        except Exception as exc:  # noqa: BLE001 - UI must not crash.
            st.error(f"Upload failed: {exc}")

    dataframe = session.get("uploaded_df")
    if dataframe is not None:
        render_data_preview(dataframe)
    render_warning_panel(session.get("warnings"))


if __name__ == "__main__":
    main()
