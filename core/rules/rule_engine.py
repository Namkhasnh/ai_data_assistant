from __future__ import annotations

import time

import pandas as pd

from core.rules.base_rule import RuleApplicationError
from core.rules.rule_registry import RuleRegistry
from models.rule import ExecutionReport, Rule, RuleExecutionResult, RuleSet


class RuleEngine:
    """Sequential deterministic rule execution engine."""

    def __init__(self, registry: RuleRegistry | None = None) -> None:
        self.registry = registry or RuleRegistry.default()

    def execute(self, df: pd.DataFrame, rule_set: RuleSet) -> tuple[pd.DataFrame, ExecutionReport]:
        """Apply enabled rules in priority order and return dataframe plus report."""

        result = df.copy(deep=True)
        report_results: list[RuleExecutionResult] = []

        for rule in self._ordered_rules(rule_set.rules):
            started_at = time.perf_counter()
            if not rule.enabled:
                report_results.append(
                    RuleExecutionResult(
                        rule_id=rule.id,
                        rule_type=rule.type,
                        status="skipped",
                        affected_rows=0,
                        execution_time_ms=self._elapsed_ms(started_at),
                        message="Rule disabled",
                    )
                )
                continue

            before = result.copy(deep=True)
            try:
                rule_instance = self.registry.create(rule)
                result = rule_instance.apply(result, rule)
                report_results.append(
                    RuleExecutionResult(
                        rule_id=rule.id,
                        rule_type=rule.type,
                        status="applied",
                        affected_rows=self._affected_rows(before, result),
                        execution_time_ms=self._elapsed_ms(started_at),
                        message="Rule applied",
                    )
                )
            except Exception as exc:
                result = before
                report_results.append(
                    RuleExecutionResult(
                        rule_id=rule.id,
                        rule_type=rule.type,
                        status="failed",
                        affected_rows=0,
                        execution_time_ms=self._elapsed_ms(started_at),
                        message=str(exc),
                    )
                )

        return result, ExecutionReport(results=report_results)

    @staticmethod
    def _ordered_rules(rules: list[Rule]) -> list[Rule]:
        return sorted(rules, key=lambda rule: rule.priority)

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return round((time.perf_counter() - started_at) * 1000, 4)

    @staticmethod
    def _affected_rows(before: pd.DataFrame, after: pd.DataFrame) -> int:
        all_columns = list(dict.fromkeys([*before.columns, *after.columns]))
        before_aligned = before.reindex(columns=all_columns)
        after_aligned = after.reindex(columns=all_columns)
        equal_values = before_aligned.eq(after_aligned).fillna(False)
        equal_or_both_null = equal_values | (
            before_aligned.isna() & after_aligned.isna()
        )
        return int((~equal_or_both_null).any(axis=1).sum())
