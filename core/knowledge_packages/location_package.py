from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from core.knowledge_packages.base_package import BasePackage


class LocationPackage(BasePackage):
    """Generate semantic location attributes from standardized locations."""

    package_id = "location"
    name = "Location Package"
    description = "Generate semantic location attributes from standardized locations."
    version = "1.0"
    enabled = True
    priority = 200
    required_columns: tuple[str, ...] = ()
    produced_columns = ("city", "province", "region", "country")

    location_semantic_type = "JOB_LOCATION"
    standardized_location_semantic_type = "STANDARDIZED_LOCATION"
    default_provinces_path = Path("knowledge/locations/provinces.json")

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Append semantic location columns using external knowledge files."""

        output = dataframe.copy(deep=True)
        location_column = self._first_semantic_column(
            dataframe=output,
            runtime_context=runtime_context,
            semantic_type=self.location_semantic_type,
        )
        standardized_location_column = self._first_semantic_column(
            dataframe=output,
            runtime_context=runtime_context,
            semantic_type=self.standardized_location_semantic_type,
        )
        if location_column is None or standardized_location_column is None:
            self.request_skip("Location package skipped; no usable semantic columns.")
            return output

        writable_columns: set[str] = set()
        for column in self.produced_columns:
            if column not in output.columns:
                output[column] = pd.NA
                writable_columns.add(column)

        provinces = self._load_provinces(knowledge_config)

        for row_index, row in output.iterrows():
            standardized_location = row.get(standardized_location_column)
            location_attributes = self._lookup_location_attributes(
                standardized_location,
                provinces,
            )
            if location_attributes is None:
                self.record_unknown_value(standardized_location)
                continue

            for column in self.produced_columns:
                self._assign_if_available(
                    output=output,
                    row_index=row_index,
                    column=column,
                    attributes=location_attributes,
                    writable_columns=writable_columns,
                )

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

    def _load_provinces(
        self,
        knowledge_config: Mapping[str, Any] | None,
    ) -> dict[str, dict[str, Any]]:
        path = self._config_path(knowledge_config)
        payload = self._load_json_object(path)
        lookup: dict[str, dict[str, Any]] = {}
        for canonical_location, entry in payload.items():
            if not isinstance(entry, dict):
                continue
            attributes = {
                "city": entry.get("city"),
                "province": entry.get("province"),
                "region": entry.get("region"),
                "country": entry.get("country"),
            }
            self._add_lookup_entry(lookup, canonical_location, attributes)
            aliases = entry.get("aliases", [])
            if isinstance(aliases, list):
                for alias in aliases:
                    self._add_lookup_entry(lookup, alias, attributes)
        return lookup

    def _config_path(self, knowledge_config: Mapping[str, Any] | None) -> Path:
        if knowledge_config is None:
            return self.default_provinces_path
        for key in ("provinces_file", "knowledge_file"):
            configured_path = knowledge_config.get(key)
            if isinstance(configured_path, str) and configured_path.strip():
                return Path(configured_path)
        return self.default_provinces_path

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

    def _lookup_location_attributes(
        self,
        standardized_location: object,
        provinces: Mapping[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        if standardized_location is None or pd.isna(standardized_location):
            return None
        location_text = str(standardized_location).strip()
        if not location_text:
            return None
        return provinces.get(location_text.casefold())

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
