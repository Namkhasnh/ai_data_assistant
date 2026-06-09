from __future__ import annotations

import pytest

from core.rules.base_rule import RuleApplicationError
from core.rules.mapping_rule import MappingRule
from core.rules.rule_registry import RuleRegistry
from models.rule import Rule


def test_rule_registry_dispatches_registered_rule_types() -> None:
    registry = RuleRegistry()
    registry.register("mapping", MappingRule)
    rule = Rule(
        id="normalize",
        type="mapping",
        column="value",
        parameters={"mapping": {"a": "b"}},
    )

    instance = registry.create(rule)

    assert isinstance(instance, MappingRule)
    assert registry.available_rules() == ["mapping"]


def test_default_rule_registry_exposes_core_rule_types() -> None:
    registry = RuleRegistry.default()

    assert registry.available_rules() == [
        "mapping",
        "regex",
        "transformation",
        "validation",
    ]


def test_rule_registry_rejects_unknown_rule_type() -> None:
    registry = RuleRegistry.default()
    rule = Rule(id="unknown", type="missing_value", column="value")

    with pytest.raises(RuleApplicationError, match="Unsupported rule type"):
        registry.create(rule)
