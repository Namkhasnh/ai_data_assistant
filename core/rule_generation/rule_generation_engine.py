from __future__ import annotations

import json

from core.rule_generation.generator_registry import GeneratorRegistry
from models.dataset import DatasetMetadata
from models.rule import Rule, RuleSet
from models.rule_generation import RuleGenerationReport
from models.semantic_tag import SemanticDetectionReport


class RuleGenerationEngine:
    """Deterministic rule suggestion engine."""

    def __init__(
        self,
        registry: GeneratorRegistry | None = None,
        enabled_generators: list[str] | None = None,
    ) -> None:
        self.registry = registry or GeneratorRegistry.default()
        self.enabled_generators = enabled_generators or self.registry.available_generators()

    def generate(
        self,
        metadata: DatasetMetadata,
        semantic_report: SemanticDetectionReport,
    ) -> tuple[RuleSet, RuleGenerationReport]:
        generated_rules: list[Rule] = []
        generated_by_generator: dict[str, int] = {}
        generator_warnings: dict[str, list[str]] = {}
        warnings: list[str] = []

        for generator_name in self.enabled_generators:
            generator = self.registry.create(generator_name)
            rules = generator.generate(metadata, semantic_report)
            generated_rules.extend(rules)
            generated_by_generator[generator.name] = len(rules)
            if generator.warnings:
                generator_warnings[generator.name] = generator.warnings
                warnings.extend(generator.warnings)

        deduplicated_rules, duplicate_count = self._deduplicate_rules(generated_rules)
        sorted_rules = sorted(deduplicated_rules, key=lambda rule: (rule.priority, rule.id))
        skipped_columns = self._skipped_columns(metadata, sorted_rules)

        rule_set = RuleSet(rules=sorted_rules)
        report = RuleGenerationReport(
            total_generated_rules=len(sorted_rules),
            generated_by_generator=generated_by_generator,
            skipped_columns=skipped_columns,
            duplicate_rules_removed=duplicate_count,
            warnings=warnings,
            generator_warnings=generator_warnings,
        )
        return rule_set, report

    @staticmethod
    def _deduplicate_rules(rules: list[Rule]) -> tuple[list[Rule], int]:
        seen: set[str] = set()
        deduplicated: list[Rule] = []
        for rule in rules:
            key = json.dumps(rule.model_dump(mode="json"), sort_keys=True, ensure_ascii=False)
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(rule)
        return deduplicated, len(rules) - len(deduplicated)

    @staticmethod
    def _skipped_columns(metadata: DatasetMetadata, rules: list[Rule]) -> list[str]:
        rule_columns = {rule.column for rule in rules}
        return [
            column.name
            for column in metadata.columns
            if column.name not in rule_columns
        ]
