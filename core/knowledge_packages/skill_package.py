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
class SkillPattern:
    """Compiled alias pattern for one canonical skill."""

    canonical_skill: str
    pattern: Pattern[str]


class SkillPackage(BasePackage):
    """Extract deterministic skill lists from job text columns."""

    package_id = "skill"
    name = "Skill Package"
    description = "Generate semantic skill attributes from free-text job columns."
    version = "1.0"
    enabled = True
    priority = 400
    required_columns: tuple[str, ...] = ()
    produced_columns = ("skills",)

    default_skills_path = Path("knowledge/jobs/skills.json")
    requirements_semantic_type = "JOB_REQUIREMENTS"
    description_semantic_type = "JOB_DESCRIPTION"

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Append a list-valued skills column without modifying source text."""

        output = dataframe.copy(deep=True)
        usable_columns = self._usable_columns(
            dataframe=output,
            runtime_context=runtime_context,
        )
        if not usable_columns:
            self.request_skip("Skill package skipped; no usable semantic columns.")
            return output

        if "skills" in output.columns:
            return output

        skill_patterns = self._load_skill_patterns(knowledge_config)
        output["skills"] = [
            self._extract_skills(row=row, usable_columns=usable_columns, skill_patterns=skill_patterns)
            for _, row in output.iterrows()
        ]
        return output

    def _usable_columns(
        self,
        dataframe: pd.DataFrame,
        runtime_context: Mapping[str, Any] | None,
    ) -> list[str]:
        semantic_columns = self._semantic_columns(runtime_context)
        candidate_columns = [
            *semantic_columns.get(self.requirements_semantic_type, []),
            *semantic_columns.get(self.description_semantic_type, []),
        ]
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

    def _load_skill_patterns(
        self,
        knowledge_config: Mapping[str, Any] | None,
    ) -> list[SkillPattern]:
        path = self._config_path(knowledge_config)
        payload = self._load_json_object(path)
        patterns: list[SkillPattern] = []
        for canonical_skill, entry in payload.items():
            if not isinstance(entry, dict):
                continue
            aliases = entry.get("aliases", [])
            terms = [canonical_skill]
            if isinstance(aliases, list):
                terms.extend(alias for alias in aliases if isinstance(alias, str))
            for term in terms:
                normalized_term = term.strip()
                if normalized_term:
                    patterns.append(
                        SkillPattern(
                            canonical_skill=str(canonical_skill),
                            pattern=self._compile_term_pattern(normalized_term),
                        )
                    )
        return sorted(patterns, key=lambda item: (item.canonical_skill.casefold(), item.pattern.pattern))

    def _config_path(self, knowledge_config: Mapping[str, Any] | None) -> Path:
        if knowledge_config is None:
            return self.default_skills_path
        for key in ("skills_file", "knowledge_file"):
            configured_path = knowledge_config.get(key)
            if isinstance(configured_path, str) and configured_path.strip():
                return Path(configured_path)
        return self.default_skills_path

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

    def _extract_skills(
        self,
        row: pd.Series,
        usable_columns: list[str],
        skill_patterns: list[SkillPattern],
    ) -> list[str]:
        text = "\n".join(
            str(value)
            for value in (row.get(column) for column in usable_columns)
            if value is not None and not pd.isna(value)
        )
        if not text:
            return []

        matched_skills = {
            skill_pattern.canonical_skill
            for skill_pattern in skill_patterns
            if skill_pattern.pattern.search(text)
        }
        return sorted(matched_skills, key=lambda value: value.casefold())
