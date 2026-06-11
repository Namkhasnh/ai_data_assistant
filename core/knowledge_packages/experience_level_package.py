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
class ExperienceLevelDefinition:
    """Configured experience level and minimum years."""

    level: str
    min_years: int | None


@dataclass(frozen=True)
class ExperienceAliasPattern:
    """Compiled alias pattern for one configured experience level."""

    level: str
    min_years: int | None
    pattern: Pattern[str]


@dataclass(frozen=True)
class ExperienceSignal:
    """Detected experience attributes for one row."""

    years: int | None
    level: str | None


class ExperienceLevelPackage(BasePackage):
    """Generate deterministic experience attributes from job text columns."""

    package_id = "experience_level"
    name = "Experience Level Package"
    description = "Generate experience attributes from job text columns."
    version = "1.0"
    enabled = True
    priority = 700
    required_columns: tuple[str, ...] = ()
    produced_columns = ("experience_years", "experience_level")

    default_experience_levels_path = Path("knowledge/jobs/experience_levels.json")
    semantic_types = ("JOB_TITLE", "JOB_DESCRIPTION", "BENEFITS")
    years_pattern = re.compile(
        r"(?<!\d)(\d{1,2})(?:\+)?\s*(?:years?|yrs?|năm)(?!\w)",
        re.IGNORECASE,
    )

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Append experience_years and experience_level using text signals."""

        output = dataframe.copy(deep=True)
        usable_columns = self._usable_columns(
            dataframe=output,
            runtime_context=runtime_context,
        )
        if not usable_columns:
            self.request_skip(
                "Experience level package skipped; no usable semantic columns."
            )
            return output

        level_definitions, alias_patterns = self._load_experience_levels(
            knowledge_config,
        )
        signals = [
            self._detect_experience_signal(
                row=row,
                usable_columns=usable_columns,
                level_definitions=level_definitions,
                alias_patterns=alias_patterns,
            )
            for _, row in output.iterrows()
        ]

        if "experience_years" not in output.columns:
            output["experience_years"] = [signal.years for signal in signals]
        if "experience_level" not in output.columns:
            output["experience_level"] = [signal.level for signal in signals]
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

    def _load_experience_levels(
        self,
        knowledge_config: Mapping[str, Any] | None,
    ) -> tuple[list[ExperienceLevelDefinition], list[ExperienceAliasPattern]]:
        path = self._config_path(knowledge_config)
        payload = self._load_json_object(path)
        level_definitions: list[ExperienceLevelDefinition] = []
        alias_patterns: list[ExperienceAliasPattern] = []
        for level, entry in payload.items():
            if not isinstance(entry, dict):
                continue
            min_years = self._min_years(entry.get("min_years"))
            level_text = str(level)
            level_definitions.append(
                ExperienceLevelDefinition(level=level_text, min_years=min_years)
            )
            aliases = entry.get("aliases", [])
            if not isinstance(aliases, list):
                continue
            for alias in sorted({value.strip() for value in aliases if isinstance(value, str) and value.strip()}):
                alias_patterns.append(
                    ExperienceAliasPattern(
                        level=level_text,
                        min_years=min_years,
                        pattern=self._compile_term_pattern(alias),
                    )
                )

        return (
            sorted(
                level_definitions,
                key=lambda item: (
                    -1 if item.min_years is None else item.min_years,
                    item.level.casefold(),
                ),
            ),
            sorted(
                alias_patterns,
                key=lambda item: (
                    -1 if item.min_years is None else item.min_years,
                    item.level.casefold(),
                    item.pattern.pattern,
                ),
            ),
        )

    def _config_path(self, knowledge_config: Mapping[str, Any] | None) -> Path:
        if knowledge_config is None:
            return self.default_experience_levels_path
        for key in ("experience_levels_file", "knowledge_file"):
            configured_path = knowledge_config.get(key)
            if isinstance(configured_path, str) and configured_path.strip():
                return Path(configured_path)
        return self.default_experience_levels_path

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

    def _min_years(self, value: object) -> int | None:
        if value is None:
            return None
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None

    def _compile_term_pattern(self, term: str) -> Pattern[str]:
        escaped_term = re.escape(term)
        return re.compile(rf"(?<!\w){escaped_term}(?!\w)", re.IGNORECASE)

    def _detect_experience_signal(
        self,
        row: pd.Series,
        usable_columns: list[str],
        level_definitions: list[ExperienceLevelDefinition],
        alias_patterns: list[ExperienceAliasPattern],
    ) -> ExperienceSignal:
        text = self._aggregated_text(row=row, usable_columns=usable_columns)
        if not text:
            return ExperienceSignal(years=None, level=None)

        years = self._extract_years(text)
        alias_level = self._detect_alias_level(text, alias_patterns)
        if years is not None:
            numeric_level = self._level_from_years(years, level_definitions)
            compatible_alias_level = self._compatible_alias_level(
                years=years,
                alias_level=alias_level,
                level_definitions=level_definitions,
            )
            return ExperienceSignal(
                years=years,
                level=compatible_alias_level or numeric_level,
            )

        if alias_level is None:
            return ExperienceSignal(years=None, level=None)
        alias_min_years = self._min_years_for_level(alias_level, level_definitions)
        return ExperienceSignal(
            years=alias_min_years if alias_min_years == 0 else None,
            level=alias_level,
        )

    def _extract_years(self, text: str) -> int | None:
        values = [int(match.group(1)) for match in self.years_pattern.finditer(text)]
        if not values:
            return None
        return max(values)

    def _detect_alias_level(
        self,
        text: str,
        alias_patterns: list[ExperienceAliasPattern],
    ) -> str | None:
        matches = [
            alias_pattern
            for alias_pattern in alias_patterns
            if alias_pattern.pattern.search(text)
        ]
        if not matches:
            return None
        return sorted(
            matches,
            key=lambda item: (
                -1 if item.min_years is None else -item.min_years,
                item.level.casefold(),
            ),
        )[0].level

    def _level_from_years(
        self,
        years: int,
        level_definitions: list[ExperienceLevelDefinition],
    ) -> str | None:
        candidates = [
            definition
            for definition in level_definitions
            if definition.min_years is not None and definition.min_years <= years
        ]
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda item: (-int(item.min_years or 0), item.level.casefold()),
        )[0].level

    def _compatible_alias_level(
        self,
        years: int,
        alias_level: str | None,
        level_definitions: list[ExperienceLevelDefinition],
    ) -> str | None:
        if alias_level is None:
            return None
        alias_min_years = self._min_years_for_level(alias_level, level_definitions)
        if alias_min_years is None or years < alias_min_years:
            return None
        next_min_years = self._next_min_years(alias_min_years, level_definitions)
        if next_min_years is not None and years > next_min_years:
            return None
        return alias_level

    def _min_years_for_level(
        self,
        level: str,
        level_definitions: list[ExperienceLevelDefinition],
    ) -> int | None:
        for definition in level_definitions:
            if definition.level == level:
                return definition.min_years
        return None

    def _next_min_years(
        self,
        min_years: int,
        level_definitions: list[ExperienceLevelDefinition],
    ) -> int | None:
        next_values = sorted(
            {
                int(definition.min_years)
                for definition in level_definitions
                if definition.min_years is not None and definition.min_years > min_years
            }
        )
        if not next_values:
            return None
        return next_values[0]

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
