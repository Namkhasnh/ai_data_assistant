from __future__ import annotations

from core.rule_generation.generator_registry import GeneratorRegistry
from core.rule_generation.regex_rule_generator import RegexRuleGenerator


def test_generator_registry_dispatches_registered_generators() -> None:
    registry = GeneratorRegistry()
    registry.register("regex", RegexRuleGenerator)

    generator = registry.create("regex")

    assert isinstance(generator, RegexRuleGenerator)
    assert registry.available_generators() == ["regex"]


def test_default_generator_registry_exposes_core_generators() -> None:
    registry = GeneratorRegistry.default()

    assert registry.available_generators() == [
        "mapping",
        "regex",
        "transformation",
        "validation",
    ]
