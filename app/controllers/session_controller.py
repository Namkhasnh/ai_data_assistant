from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

import pandas as pd


class SessionController:
    """Centralized access to Streamlit session state."""

    default_keys: dict[str, Any] = {
        "run_id": None,
        "uploaded_df": None,
        "uploaded_file_path": None,
        "metadata": None,
        "semantic_report": None,
        "rules": None,
        "draft_rules": None,
        "rules_dirty": False,
        "standardized_df": None,
        "enriched_df": None,
        "warnings": [],
    }

    def __init__(self, state: MutableMapping[str, Any]) -> None:
        self.state = state
        self.initialize()

    @classmethod
    def from_streamlit(cls) -> SessionController:
        import streamlit as st

        return cls(st.session_state)

    def initialize(self) -> None:
        for key, value in self.default_keys.items():
            if key not in self.state:
                self.state[key] = list(value) if isinstance(value, list) else value

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.state[key] = value

    def append_warning(self, warning: str) -> None:
        warnings = list(self.state.get("warnings", []))
        warnings.append(warning)
        self.state["warnings"] = warnings

    def set_uploaded_dataframe(self, dataframe: pd.DataFrame, file_path: str) -> None:
        self.state["uploaded_df"] = dataframe
        self.state["uploaded_file_path"] = file_path

    def set_metadata(self, metadata: Any) -> None:
        self.state["metadata"] = metadata

    def set_semantic_report(self, semantic_report: Any) -> None:
        self.state["semantic_report"] = semantic_report

    def set_rules(self, rules: Any) -> None:
        self.state["rules"] = rules
        self.state["draft_rules"] = rules.model_copy(deep=True) if hasattr(rules, "model_copy") else rules
        self.state["rules_dirty"] = False

    def set_draft_rules(self, draft_rules: Any) -> None:
        self.state["draft_rules"] = draft_rules
        self.state["rules_dirty"] = True

    def mark_rules_saved(self) -> None:
        self.state["rules"] = self.state.get("draft_rules")
        self.state["rules_dirty"] = False

    def set_standardized_dataframe(self, dataframe: pd.DataFrame) -> None:
        self.state["standardized_df"] = dataframe

    def set_enriched_dataframe(self, dataframe: pd.DataFrame) -> None:
        self.state["enriched_df"] = dataframe

    def clear_run_outputs(self) -> None:
        for key in (
            "metadata",
            "semantic_report",
            "rules",
            "draft_rules",
            "standardized_df",
            "enriched_df",
        ):
            self.state[key] = None
        self.state["rules_dirty"] = False
        self.state["warnings"] = []
