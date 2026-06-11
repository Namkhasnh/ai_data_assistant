from __future__ import annotations

import json
from typing import Any

from models.rule import RuleSet


def draft_rules_from_rule_set(rule_set: RuleSet) -> list[dict[str, Any]]:
    """Return editable rule dictionaries without mutating the source RuleSet."""

    return [rule.model_dump(mode="json") for rule in rule_set.rules]


def update_rule_enabled(
    draft_rules: list[dict[str, Any]],
    rule_id: str,
    enabled: bool,
) -> list[dict[str, Any]]:
    updated = [dict(rule) for rule in draft_rules]
    for rule in updated:
        if rule.get("id") == rule_id:
            rule["enabled"] = enabled
            break
    return updated


def update_rule_parameters(
    draft_rules: list[dict[str, Any]],
    rule_id: str,
    parameters: dict[str, Any],
) -> list[dict[str, Any]]:
    updated = [dict(rule) for rule in draft_rules]
    for rule in updated:
        if rule.get("id") == rule_id:
            rule["parameters"] = parameters
            break
    return updated


def draft_rules_to_rule_set(draft_rules: list[dict[str, Any]] | RuleSet) -> RuleSet:
    if isinstance(draft_rules, RuleSet):
        return draft_rules
    return RuleSet.model_validate({"rules": draft_rules})


def rules_json_preview(draft_rules: list[dict[str, Any]] | RuleSet) -> str:
    rule_set = draft_rules_to_rule_set(draft_rules)
    return json.dumps(rule_set.model_dump(mode="json"), indent=2, ensure_ascii=False)


def render_rule_editor(draft_rules: list[dict[str, Any]] | RuleSet) -> RuleSet:
    """Render rule controls and return edited draft rules without persisting."""

    import streamlit as st

    editable_rules = draft_rules_from_rule_set(draft_rules_to_rule_set(draft_rules))
    for index, rule in enumerate(editable_rules):
        with st.expander(f"{rule.get('id', f'rule_{index}')} ({rule.get('type', 'unknown')})"):
            rule["enabled"] = st.checkbox(
                "Enabled",
                value=bool(rule.get("enabled", True)),
                key=f"rule_enabled_{rule.get('id', index)}",
            )
            raw_parameters = st.text_area(
                "Parameters JSON",
                value=json.dumps(rule.get("parameters", {}), indent=2, ensure_ascii=False),
                key=f"rule_parameters_{rule.get('id', index)}",
            )
            try:
                parameters = json.loads(raw_parameters)
                if isinstance(parameters, dict):
                    rule["parameters"] = parameters
                else:
                    st.warning("Parameters must be a JSON object.")
            except json.JSONDecodeError as exc:
                st.warning(f"Invalid parameters JSON: {exc}")
            st.code(json.dumps(rule, indent=2, ensure_ascii=False), language="json")
    return draft_rules_to_rule_set(editable_rules)
