from __future__ import annotations

from core.rule_generation.base_generator import BaseRuleGenerator
from core.rule_generation.mapping_rule_generator import MappingRuleGenerator
from core.rule_generation.regex_rule_generator import RegexRuleGenerator
from core.rule_generation.transformation_rule_generator import TransformationRuleGenerator
from core.rule_generation.validation_rule_generator import ValidationRuleGenerator


class GeneratorRegistry:
    """Plugin registry for deterministic rule generators."""

    def __init__(self) -> None:
        self._generators: dict[str, type[BaseRuleGenerator]] = {}

    def register(self, name: str, generator_class: type[BaseRuleGenerator]) -> None:
        self._generators[name.strip().lower()] = generator_class

    def create(self, name: str) -> BaseRuleGenerator:
        generator_class = self._generators[name.strip().lower()]
        return generator_class()

    def available_generators(self) -> list[str]:
        return sorted(self._generators)

    @classmethod
    def default(cls) -> GeneratorRegistry:
        registry = cls()
        registry.register("mapping", MappingRuleGenerator)
        registry.register("regex", RegexRuleGenerator)
        registry.register("transformation", TransformationRuleGenerator)
        registry.register("validation", ValidationRuleGenerator)
        return registry
