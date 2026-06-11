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
class LanguageAliasPattern:
    """Compiled alias pattern for one configured language."""

    language: str
    alias: str
    pattern: Pattern[str]


class LanguagePackage(BasePackage):
    """Detect deterministic language requirements from job text columns."""

    package_id = "language"
    name = "Language Package"
    description = "Generate language attributes from job text columns."
    version = "1.0"
    enabled = True
    priority = 1000
    required_columns: tuple[str, ...] = ()
    produced_columns = ("languages",)

    default_languages_path = Path("knowledge/jobs/languages.json")
    semantic_types = (
        "JOB_TITLE",
        "JOB_DESCRIPTION",
        "BENEFITS",
        "JOB_REQUIREMENTS",
        "EDUCATION",
    )

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Append a list-valued languages column using alias matching."""

        output = dataframe.copy(deep=True)
        usable_columns = self._usable_columns(
            dataframe=output,
            runtime_context=runtime_context,
        )
        if not usable_columns:
            self.request_skip("Language package skipped; no usable semantic columns.")
            return output

        if "languages" in output.columns:
            return output

        alias_patterns = self._load_alias_patterns(knowledge_config)
        output["languages"] = [
            self._detect_languages(
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
    ) -> list[LanguageAliasPattern]:
        path = self._config_path(knowledge_config)
        payload = self._load_json_object(path)
        patterns: list[LanguageAliasPattern] = []
        for language, entry in payload.items():
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
                    LanguageAliasPattern(
                        language=str(language),
                        alias=alias,
                        pattern=self._compile_term_pattern(alias),
                    )
                )
        return sorted(
            patterns,
            key=lambda item: (item.language.casefold(), item.alias.casefold()),
        )

    def _config_path(self, knowledge_config: Mapping[str, Any] | None) -> Path:
        if knowledge_config is None:
            return self.default_languages_path
        for key in ("languages_file", "knowledge_file"):
            configured_path = knowledge_config.get(key)
            if isinstance(configured_path, str) and configured_path.strip():
                return Path(configured_path)
        return self.default_languages_path

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

    def _detect_languages(
        self,
        row: pd.Series,
        usable_columns: list[str],
        alias_patterns: list[LanguageAliasPattern],
    ) -> list[str]:
        text = self._aggregated_text(row=row, usable_columns=usable_columns)
        if not text:
            return []

        matched_languages = {
            alias_pattern.language
            for alias_pattern in alias_patterns
            if alias_pattern.pattern.search(text)
        }
        return sorted(matched_languages, key=lambda value: value.casefold())

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
