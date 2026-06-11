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
class WorkModePattern:
    """Compiled alias pattern for one configured work mode."""

    mode: str
    priority: int
    pattern: Pattern[str]


class WorkModePackage(BasePackage):
    """Detect deterministic work mode values from job text columns."""

    package_id = "work_mode"
    name = "Work Mode Package"
    description = "Generate work mode attributes from job text columns."
    version = "1.0"
    enabled = True
    priority = 500
    required_columns: tuple[str, ...] = ()
    produced_columns = ("work_mode",)

    default_work_modes_path = Path("knowledge/jobs/work_modes.json")
    semantic_types = ("JOB_DESCRIPTION", "BENEFITS", "WORKING_TIME", "WORK_LOCATION")

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Append a work_mode column using semantic text inputs and local knowledge."""

        output = dataframe.copy(deep=True)
        usable_columns = self._usable_columns(
            dataframe=output,
            runtime_context=runtime_context,
        )
        if not usable_columns:
            self.request_skip("Work mode package skipped; no usable semantic columns.")
            return output

        if "work_mode" in output.columns:
            return output

        work_mode_patterns = self._load_work_mode_patterns(knowledge_config)
        output["work_mode"] = [
            self._detect_work_mode(
                row=row,
                usable_columns=usable_columns,
                work_mode_patterns=work_mode_patterns,
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

    def _load_work_mode_patterns(
        self,
        knowledge_config: Mapping[str, Any] | None,
    ) -> list[WorkModePattern]:
        path = self._config_path(knowledge_config)
        payload = self._load_json_object(path)
        patterns: list[WorkModePattern] = []
        for mode, entry in payload.items():
            if not isinstance(entry, dict):
                continue
            priority = self._priority(entry.get("priority"))
            aliases = entry.get("aliases", [])
            terms = [str(mode)]
            if isinstance(aliases, list):
                terms.extend(alias for alias in aliases if isinstance(alias, str))
            for term in sorted({value.strip() for value in terms if value.strip()}):
                patterns.append(
                    WorkModePattern(
                        mode=str(mode),
                        priority=priority,
                        pattern=self._compile_term_pattern(term),
                    )
                )
        return sorted(
            patterns,
            key=lambda item: (item.priority, item.mode.casefold(), item.pattern.pattern),
        )

    def _config_path(self, knowledge_config: Mapping[str, Any] | None) -> Path:
        if knowledge_config is None:
            return self.default_work_modes_path
        for key in ("work_modes_file", "knowledge_file"):
            configured_path = knowledge_config.get(key)
            if isinstance(configured_path, str) and configured_path.strip():
                return Path(configured_path)
        return self.default_work_modes_path

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

    def _priority(self, value: object) -> int:
        if isinstance(value, int):
            return value
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return 1000

    def _compile_term_pattern(self, term: str) -> Pattern[str]:
        escaped_term = re.escape(term)
        return re.compile(rf"(?<!\w){escaped_term}(?!\w)", re.IGNORECASE)

    def _detect_work_mode(
        self,
        row: pd.Series,
        usable_columns: list[str],
        work_mode_patterns: list[WorkModePattern],
    ) -> str | None:
        text = self._aggregated_text(row=row, usable_columns=usable_columns)
        if not text:
            return None

        matches = [
            work_mode_pattern
            for work_mode_pattern in work_mode_patterns
            if work_mode_pattern.pattern.search(text)
        ]
        if not matches:
            return None
        return sorted(
            matches,
            key=lambda item: (item.priority, item.mode.casefold()),
        )[0].mode

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
