from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Pattern

import pandas as pd

from core.knowledge_packages.base_package import BasePackage


@dataclass(frozen=True)
class CompanySizeAliasPattern:
    """Compiled alias pattern for one configured company size group."""

    group: str
    alias: str
    pattern: Pattern[str]


class CompanySizePackage(BasePackage):
    """Generate deterministic company size groups from company information."""

    package_id = "company_size"
    name = "Company Size Package"
    description = "Generate company size attributes from company information."
    version = "1.0"
    enabled = True
    priority = 1100
    required_columns: tuple[str, ...] = ()
    produced_columns = ("company_size_group",)

    default_company_size_groups_path = Path("knowledge/jobs/company_size_groups.json")
    semantic_types = ("COMPANY_NAME", "COMPANY_DESCRIPTION", "JOB_DESCRIPTION")
    employee_count_pattern = re.compile(
        r"(?<!\d)(\d{1,7})(?:\+)?\s*(?:employees?|staff|people|nhân viên)(?!\w)",
        re.IGNORECASE,
    )
    employee_count_plus_pattern = re.compile(
        r"(?<!\d)(\d{1,7})\+\s*(?:employees?|staff|people|nhân viên)(?!\w)",
        re.IGNORECASE,
    )

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Append company_size_group using numeric extraction before aliases."""

        output = dataframe.copy(deep=True)
        usable_columns = self._usable_columns(
            dataframe=output,
            runtime_context=runtime_context,
        )
        if not usable_columns:
            self.request_skip(
                "Company size package skipped; no usable semantic columns."
            )
            return output

        if "company_size_group" in output.columns:
            return output

        alias_patterns = self._load_alias_patterns(knowledge_config)
        output["company_size_group"] = [
            self._detect_company_size_group(
                row=row,
                usable_columns=usable_columns,
                alias_patterns=alias_patterns,
            )
            for _, row in output.iterrows()
        ]
        return output

    def _usable_columns(
        self,
        dataframe: pd.DataFrame,
        runtime_context: Mapping[str, Any] | None,
    ) -> list[str]:
        semantic_columns = self._semantic_columns(runtime_context)
        candidate_columns: list[str] = []
        for semantic_type in self.semantic_types:
            candidate_columns.extend(semantic_columns.get(semantic_type, []))
        return sorted({column for column in candidate_columns if column in dataframe.columns})

    def _semantic_columns(
        self,
        runtime_context: Mapping[str, Any] | None,
    ) -> Mapping[str, list[str]]:
        if runtime_context is None:
            return {}
        semantic_columns = runtime_context.get("semantic_columns", {})
        if not isinstance(semantic_columns, Mapping):
            return {}
        return {
            str(semantic_type): [
                str(column)
                for column in columns
                if isinstance(column, str) and column
            ]
            for semantic_type, columns in semantic_columns.items()
            if isinstance(columns, list)
        }

    def _load_alias_patterns(
        self,
        knowledge_config: Mapping[str, Any] | None,
    ) -> list[CompanySizeAliasPattern]:
        path = self._config_path(knowledge_config)
        payload = self._load_json_object(path)
        patterns: list[CompanySizeAliasPattern] = []
        for group, entry in payload.items():
            if not isinstance(entry, dict):
                continue
            aliases = entry.get("aliases", [])
            if not isinstance(aliases, list):
                continue
            normalized_aliases = {
                value.strip()
                for value in aliases
                if isinstance(value, str) and value.strip()
            }
            for alias in sorted(normalized_aliases):
                patterns.append(
                    CompanySizeAliasPattern(
                        group=str(group),
                        alias=alias,
                        pattern=self._compile_term_pattern(alias),
                    )
                )
        return sorted(patterns, key=lambda item: (item.group.casefold(), item.alias.casefold()))

    def _config_path(self, knowledge_config: Mapping[str, Any] | None) -> Path:
        if knowledge_config is None:
            return self.default_company_size_groups_path
        for key in ("company_size_groups_file", "knowledge_file"):
            configured_path = knowledge_config.get(key)
            if isinstance(configured_path, str) and configured_path.strip():
                return Path(configured_path)
        return self.default_company_size_groups_path

    def _load_json_object(self, path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self.warnings.append(
                f"Knowledge package {self.package_id} missing knowledge file: {path}"
            )
            return {}
        except json.JSONDecodeError as exc:
            self.warnings.append(
                f"Knowledge package {self.package_id} invalid knowledge file: {path}: {exc}"
            )
            return {}
        if not isinstance(payload, dict):
            self.warnings.append(
                f"Knowledge package {self.package_id} knowledge file must be a JSON object: {path}"
            )
            return {}
        return payload

    def _compile_term_pattern(self, term: str) -> Pattern[str]:
        escaped_term = re.escape(term)
        return re.compile(rf"(?<!\w){escaped_term}(?!\w)", re.IGNORECASE)

    def _detect_company_size_group(
        self,
        row: pd.Series,
        usable_columns: list[str],
        alias_patterns: list[CompanySizeAliasPattern],
    ) -> str | None:
        text = self._aggregated_text(row=row, usable_columns=usable_columns)
        if not text:
            return None

        numeric_group = self._group_from_employee_count(text)
        if numeric_group is not None:
            return numeric_group
        return self._group_from_aliases(text, alias_patterns)

    def _group_from_employee_count(self, text: str) -> str | None:
        plus_counts = [
            int(match.group(1))
            for match in self.employee_count_plus_pattern.finditer(text)
        ]
        if plus_counts:
            return self._group_from_count(max(plus_counts), is_plus=True)

        counts = [
            int(match.group(1))
            for match in self.employee_count_pattern.finditer(text)
        ]
        if not counts:
            return None
        return self._group_from_count(max(counts), is_plus=False)

    def _group_from_count(self, count: int, is_plus: bool) -> str:
        if is_plus and count >= 1000:
            return "Enterprise"
        if count <= 20:
            return "Startup"
        if count <= 100:
            return "Small"
        if count <= 500:
            return "Medium"
        if count <= 1000:
            return "Large"
        return "Enterprise"

    def _group_from_aliases(
        self,
        text: str,
        alias_patterns: list[CompanySizeAliasPattern],
    ) -> str | None:
        scores: dict[str, int] = {}
        for alias_pattern in alias_patterns:
            if alias_pattern.pattern.search(text):
                scores[alias_pattern.group] = scores.get(alias_pattern.group, 0) + 1
        if not scores:
            return None
        highest_score = max(scores.values())
        candidate_groups = [
            group for group, score in scores.items() if score == highest_score
        ]
        return sorted(candidate_groups)[0]

    def _aggregated_text(self, row: pd.Series, usable_columns: list[str]) -> str:
        normalized_values = {
            text
            for text in (
                self._normalized_text(row.get(column)) for column in usable_columns
            )
            if text
        }
        return "\n".join(sorted(normalized_values, key=lambda value: value.casefold()))

    def _normalized_text(self, value: object) -> str | None:
        if value is None:
            return None
        try:
            if bool(pd.isna(value)):
                return None
        except (TypeError, ValueError):
            pass
        text = re.sub(r"\s+", " ", str(value).strip())
        return text or None
