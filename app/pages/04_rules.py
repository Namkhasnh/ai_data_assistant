from __future__ import annotations

from app.bootstrap import ensure_project_root

ensure_project_root()

from app.components.rule_editor import render_rule_editor, rules_json_preview
from app.components.warning_panel import render_warning_panel
from app.controllers.pipeline_controller import PipelineController
from app.controllers.session_controller import SessionController
from app.controllers.workspace_controller import WorkspaceController


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Rules", layout="wide")
    st.title("Rules")

    session = SessionController.from_streamlit()
    workspace = WorkspaceController(session)
    pipeline = PipelineController(session=session, workspace=workspace)

    left, right = st.columns(2)
    with left:
        if st.button("Generate Rules", type="primary"):
            try:
                pipeline.run_rule_generation()
                st.success("Rules generated")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Rule generation failed: {exc}")
    with right:
        if st.button("Generate AI Suggestions"):
            try:
                pipeline.generate_ai_suggestions()
                st.success("AI suggestions generated")
            except Exception as exc:  # noqa: BLE001
                st.error(f"AI suggestion generation failed: {exc}")

    draft_rules = session.get("draft_rules")
    if draft_rules is not None:
        edited_rules = render_rule_editor(draft_rules)
        if edited_rules != draft_rules:
            session.set_draft_rules(edited_rules)
        st.subheader("Draft JSON")
        st.code(rules_json_preview(session.get("draft_rules")), language="json")
        if session.get("rules_dirty"):
            st.warning("You have unsaved rule edits.")
        if st.button("Save Rules", type="primary"):
            try:
                pipeline.save_draft_rules()
                st.success("Rules saved")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Saving rules failed: {exc}")
    else:
        st.info("Generate rules after semantic detection.")
    render_warning_panel(session.get("warnings"))


if __name__ == "__main__":
    main()
