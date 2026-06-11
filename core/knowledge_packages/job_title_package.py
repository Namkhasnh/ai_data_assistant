from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from core.knowledge_packages.base_package import BasePackage


class JobTitlePackage(BasePackage):
    """Generate semantic business attributes from standardized job titles."""

    package_id = "job_title"
    name = "Job Title Package"
    description = "Generate semantic business attributes from standardized job titles."
    version = "1.0"
    enabled = True
    priority = 100
    required_columns: tuple[str, ...] = ()
    produced_columns = ("job_group", "specialization", "seniority", "tech_domain")

    title_semantic_type = "JOB_TITLE"
    standardized_title_semantic_type = "STANDARDIZED_JOB_TITLE"
    default_job_titles_path = Path("knowledge/jobs/job_titles.json")
    default_seniority_path = Path("knowledge/jobs/seniority.json")

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Append job-title semantic columns using external knowledge files."""

        output = dataframe.copy(deep=True)
        title_column = self._first_semantic_column(
            dataframe=output,
            runtime_context=runtime_context,
            semantic_type=self.title_semantic_type,
        )
        standardized_title_column = self._first_semantic_column(
            dataframe=output,
            runtime_context=runtime_context,
            semantic_type=self.standardized_title_semantic_type,
        )
        if title_column is None or standardized_title_column is None:
            self.request_skip("Job title package skipped; no usable semantic columns.")
            return output

        writable_columns: set[str] = set()
        for column in self.produced_columns:
            if column not in output.columns:
                output[column] = pd.NA
                writable_columns.add(column)

        job_titles = self._load_job_titles(knowledge_config)
        seniority_terms = self._load_seniority_terms(knowledge_config)

        for row_index, row in output.iterrows():
            standardized_title = row.get(standardized_title_column)
            title = row.get(title_column)

            job_attributes = self._lookup_job_attributes(standardized_title, job_titles)
            if job_attributes is None:
                self.record_unknown_value(standardized_title)
            else:
                self._assign_if_available(
                    output,
                    row_index,
                    "job_group",
                    job_attributes,
                    writable_columns,
                )
                self._assign_if_available(
                    output,
                    row_index,
                    "specialization",
                    job_attributes,
                    writable_columns,
                )
                self._assign_if_available(
                    output,
                    row_index,
                    "tech_domain",
                    job_attributes,
                    writable_columns,
                )

            seniority = self._detect_seniority(title, seniority_terms)
            if seniority is not None and "seniority" in writable_columns:
                output.at[row_index, "seniority"] = seniority

        return output

    def _first_semantic_column(
        self,
        dataframe: pd.DataFrame,
        runtime_context: Mapping[str, Any] | None,
        semantic_type: str,
    ) -> str | None:
        semantic_columns = self._semantic_columns(runtime_context)
        for column in semantic_columns.get(semantic_type, []):
            if column in dataframe.columns:
                return column
        return None

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

    def _load_job_titles(
        self,
        knowledge_config: Mapping[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        path = self._config_path(
            knowledge_config=knowledge_config,
            key="job_titles_file",
            default_path=self.default_job_titles_path,
        )
        payload = self._load_json_object(path)
        lookup: dict[str, dict[str, Any]] = {}
        for canonical_title, entry in payload.items():
            if not isinstance(entry, dict):
                continue
            attributes = {
                "job_group": entry.get("job_group"),
                "specialization": entry.get("specialization"),
                "tech_domain": entry.get("tech_domain"),
            }
            self._add_lookup_entry(lookup, canonical_title, attributes)
            aliases = entry.get("aliases", [])
            if isinstance(aliases, list):
                for alias in aliases:
                    self._add_lookup_entry(lookup, alias, attributes)
        return lookup

    def _load_seniority_terms(
        self,
        knowledge_config: Mapping[str, Any] | None,
    ) -> dict[str, str]:
        path = self._config_path(
            knowledge_config=knowledge_config,
            key="seniority_file",
            default_path=self.default_seniority_path,
        )
        payload = self._load_json_object(path)
        return {
            str(source).strip(): str(target).strip()
            for source, target in payload.items()
            if str(source).strip() and str(target).strip()
        }

    def _config_path(
        self,
        knowledge_config: Mapping[str, Any] | None,
        key: str,
        default_path: Path,
    ) -> Path:
        if knowledge_config is None:
            return default_path
        configured_path = knowledge_config.get(key)
        if isinstance(configured_path, str) and configured_path.strip():
            return Path(configured_path)
        return default_path

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

    def _add_lookup_entry(
        self,
        lookup: dict[str, dict[str, Any]],
        source_value: object,
        attributes: dict[str, Any],
    ) -> None:
        if source_value is None or pd.isna(source_value):
            return
        source_text = str(source_value).strip()
        if source_text:
            lookup[source_text.casefold()] = attributes

    def _lookup_job_attributes(
        self,
        standardized_title: object,
        job_titles: Mapping[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        if standardized_title is None or pd.isna(standardized_title):
            return None
        title_text = str(standardized_title).strip()
        if not title_text:
            return None
        return job_titles.get(title_text.casefold())

    def _assign_if_available(
        self,
        output: pd.DataFrame,
        row_index: Any,
        column: str,
        attributes: Mapping[str, Any],
        writable_columns: set[str],
    ) -> None:
        value = attributes.get(column)
        if value is not None and column in writable_columns:
            output.at[row_index, column] = value

    def _detect_seniority(
        self,
        title: object,
        seniority_terms: Mapping[str, str],
    ) -> str | None:
        if title is None or pd.isna(title):
            return None
        title_text = str(title)
        for source, target in sorted(seniority_terms.items(), key=lambda item: item[0].casefold()):
            pattern = re.compile(rf"(?<!\w){re.escape(source)}(?!\w)", re.IGNORECASE)
            if pattern.search(title_text):
                return target
        return None
