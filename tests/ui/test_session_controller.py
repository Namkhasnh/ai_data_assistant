from __future__ import annotations

import pandas as pd

from app.controllers.session_controller import SessionController
from app.controllers.workspace_controller import WorkspaceController
from models.rule import Rule, RuleSet


def test_session_controller_initializes_and_tracks_rule_drafts() -> None:
    state: dict = {}
    session = SessionController(state)
    rule_set = RuleSet(
        rules=[
            Rule(
                id="rule_001",
                type="transformation",
                column="name",
                parameters={"operation": "trim"},
            )
        ]
    )

    session.set_rules(rule_set)
    assert state["rules_dirty"] is False

    draft = rule_set.model_copy(deep=True)
    draft.rules[0].enabled = False
    session.set_draft_rules(draft)

    assert state["rules_dirty"] is True
    assert state["rules"].rules[0].enabled is True
    assert state["draft_rules"].rules[0].enabled is False

    session.mark_rules_saved()
    assert state["rules_dirty"] is False
    assert state["rules"].rules[0].enabled is False


def test_workspace_controller_saves_upload_and_loads_dataframe(tmp_path) -> None:
    state: dict = {}
    session = SessionController(state)
    workspace = WorkspaceController(session=session, workspace_root=tmp_path)
    upload = _BytesUpload(b"a,b\n1,x\n", name="sample.csv")

    saved_path = workspace.save_upload(upload, upload.name)
    dataframe = workspace.load_dataframe(saved_path)
    session.set_uploaded_dataframe(dataframe, str(saved_path))

    assert saved_path.exists()
    assert state["uploaded_file_path"] == str(saved_path)
    pd.testing.assert_frame_equal(dataframe, pd.DataFrame({"a": [1], "b": ["x"]}))


class _BytesUpload:
    def __init__(self, payload: bytes, name: str) -> None:
        self.payload = payload
        self.name = name
        self.position = 0

    def seek(self, position: int) -> None:
        self.position = position

    def read(self) -> bytes:
        return self.payload
