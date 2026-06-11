from __future__ import annotations

from app.components.rule_editor import (
    draft_rules_from_rule_set,
    draft_rules_to_rule_set,
    rules_json_preview,
    update_rule_enabled,
    update_rule_parameters,
)
from models.rule import Rule, RuleSet


def test_rule_editor_helpers_update_drafts_without_mutating_source() -> None:
    rule_set = RuleSet(
        rules=[
            Rule(
                id="rule_001",
                type="transformation",
                column="title",
                parameters={"operation": "trim"},
                enabled=True,
            )
        ]
    )

    draft = draft_rules_from_rule_set(rule_set)
    disabled = update_rule_enabled(draft, "rule_001", False)
    updated = update_rule_parameters(disabled, "rule_001", {"operation": "uppercase"})
    updated_rule_set = draft_rules_to_rule_set(updated)

    assert rule_set.rules[0].enabled is True
    assert rule_set.rules[0].parameters == {"operation": "trim"}
    assert updated_rule_set.rules[0].enabled is False
    assert updated_rule_set.rules[0].parameters == {"operation": "uppercase"}
    assert '"rule_001"' in rules_json_preview(updated_rule_set)
