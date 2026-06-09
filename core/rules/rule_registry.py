from __future__ import annotations

from core.rules.base_rule import BaseRule, RuleApplicationError
from core.rules.mapping_rule import MappingRule
from core.rules.regex_rule import RegexRule
from core.rules.transformation_rule import TransformationRule
from core.rules.validation_rule import ValidationRule
from models.rule import Rule


class RuleRegistry:
    """Plugin registry for deterministic rule executors."""

    def __init__(self) -> None:
        self._rules: dict[str, type[BaseRule]] = {}

    def register(self, rule_type: str, rule_class: type[BaseRule]) -> None:
        self._rules[rule_type.strip().lower()] = rule_class

    def create(self, rule: Rule) -> BaseRule:
        rule_class = self._rules.get(rule.type)
        if rule_class is None:
            raise RuleApplicationError(f"Unsupported rule type: {rule.type}")
        return rule_class()

    def available_rules(self) -> list[str]:
        return sorted(self._rules)

    @classmethod
    def default(cls) -> RuleRegistry:
        registry = cls()
        registry.register("mapping", MappingRule)
        registry.register("regex", RegexRule)
        registry.register("transformation", TransformationRule)
        registry.register("validation", ValidationRule)
        return registry
