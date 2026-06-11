from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from core.knowledge_packages.base_package import BasePackage
from core.knowledge_packages.package_registry import PackageRegistry
from models.knowledge_package import KnowledgePackageReport, KnowledgePackageResult


class KnowledgePackageEngine:
    """Apply optional knowledge packages without depending on concrete packages."""

    def __init__(self, registry: PackageRegistry) -> None:
        self.registry = registry
        self.warnings: list[str] = []

    def apply_packages(
        self,
        dataframe: pd.DataFrame,
        package_names: Sequence[str] | None = None,
        package_configs: Mapping[str, Mapping[str, Any]] | None = None,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> KnowledgePackageResult:
        """Apply selected packages to a copy of the dataframe."""

        self.warnings = []
        output_dataframe = dataframe.copy(deep=True)
        applied_packages: list[str] = []
        skipped_packages: list[str] = []
        produced_columns: list[str] = []
        produced_columns_by_package: dict[str, list[str]] = {}
        unknown_values_by_package: dict[str, list[str]] = {}

        packages, unknown_package_names = self._resolve_packages(package_names)
        for package_name in unknown_package_names:
            self._warn(f"Unknown knowledge package skipped: {package_name}")
            skipped_packages.append(package_name)

        for package in packages:
            package_id = package.package_id
            package_warnings: list[str] = []
            package.reset_run_state()

            if not package.enabled:
                self._warn(f"Knowledge package disabled: {package_id}")
                skipped_packages.append(package_id)
                continue

            missing_columns = [
                column for column in package.required_columns if column not in output_dataframe.columns
            ]
            if missing_columns:
                self._warn(
                    f"Knowledge package {package_id} skipped; missing required columns: "
                    f"{', '.join(missing_columns)}"
                )
                skipped_packages.append(package_id)
                continue

            knowledge_config = self._package_config(
                package=package,
                package_configs=package_configs,
                package_warnings=package_warnings,
            )
            self.warnings.extend(package_warnings)
            missing_files = self._missing_knowledge_files(knowledge_config)
            if missing_files:
                self._warn(
                    f"Knowledge package {package_id} skipped; missing knowledge files: "
                    f"{', '.join(str(path) for path in missing_files)}"
                )
                skipped_packages.append(package_id)
                continue

            package_input = output_dataframe.copy(deep=True)
            try:
                package_output = package.apply_with_context(
                    package_input,
                    knowledge_config,
                    runtime_context=runtime_context,
                )
            except Exception as exc:  # noqa: BLE001 - packages are optional.
                self._warn(f"Knowledge package {package_id} failed: {exc}")
                skipped_packages.append(package_id)
                continue
            self._extend_package_warnings(package)

            if not isinstance(package_output, pd.DataFrame):
                self._warn(f"Knowledge package {package_id} returned a non-dataframe result")
                skipped_packages.append(package_id)
                continue
            if package.skip_requested:
                skipped_packages.append(package_id)
                continue

            safe_columns = self._safe_produced_columns(
                package=package,
                before=output_dataframe,
                after=package_output,
            )
            if not safe_columns:
                self._warn(f"Knowledge package {package_id} produced no appendable columns")
                skipped_packages.append(package_id)
                continue

            for column in safe_columns:
                output_dataframe[column] = package_output[column].copy(deep=True)
                produced_columns.append(column)
            applied_packages.append(package_id)
            produced_columns_by_package[package_id] = safe_columns
            unknown_values = self._normalized_unknown_values(package)
            if unknown_values:
                unknown_values_by_package[package_id] = unknown_values
                self._warn(f"Unknown values encountered in package '{package_id}'.")

        report = KnowledgePackageReport(
            applied_packages=applied_packages,
            skipped_packages=skipped_packages,
            warnings=list(self.warnings),
            produced_columns=produced_columns,
            produced_columns_by_package=produced_columns_by_package,
            unknown_values_by_package=unknown_values_by_package,
        )
        return KnowledgePackageResult(dataframe=output_dataframe, report=report)

    def _resolve_packages(
        self,
        package_names: Sequence[str] | None,
    ) -> tuple[list[BasePackage], list[str]]:
        if package_names is None:
            return self.registry.list_packages(), []

        packages: list[BasePackage] = []
        unknown_package_names: list[str] = []
        seen: set[str] = set()
        for package_name in package_names:
            if package_name in seen:
                continue
            seen.add(package_name)
            package = self.registry.get(package_name)
            if package is None:
                unknown_package_names.append(package_name)
            else:
                packages.append(package)
        return (
            sorted(packages, key=lambda package: (package.priority, package.package_id)),
            unknown_package_names,
        )

    def _package_config(
        self,
        package: BasePackage,
        package_configs: Mapping[str, Mapping[str, Any]] | None,
        package_warnings: list[str],
    ) -> Mapping[str, Any] | None:
        if package_configs is None:
            package_warnings.append(
                f"Knowledge package {package.package_id} has no package configuration"
            )
            return None

        knowledge_config = package_configs.get(package.package_id)
        if knowledge_config is None:
            package_warnings.append(
                f"Knowledge package {package.package_id} has no package configuration"
            )
        return knowledge_config

    def _missing_knowledge_files(
        self,
        knowledge_config: Mapping[str, Any] | None,
    ) -> list[Path]:
        if knowledge_config is None:
            return []

        candidate_paths: list[Path] = []
        for key in ("knowledge_file", "knowledge_path"):
            value = knowledge_config.get(key)
            if isinstance(value, str):
                candidate_paths.append(Path(value))
        knowledge_files = knowledge_config.get("knowledge_files")
        if isinstance(knowledge_files, Sequence) and not isinstance(knowledge_files, str):
            candidate_paths.extend(Path(value) for value in knowledge_files if isinstance(value, str))

        return [path for path in candidate_paths if not path.exists()]

    def _safe_produced_columns(
        self,
        package: BasePackage,
        before: pd.DataFrame,
        after: pd.DataFrame,
    ) -> list[str]:
        package_id = package.package_id
        before_columns = list(before.columns)
        before_column_set = set(before_columns)
        after_column_set = set(after.columns)

        missing_source_columns = [
            column for column in before_columns if column not in after_column_set
        ]
        if missing_source_columns:
            self._warn(
                f"Knowledge package {package_id} attempted to remove source columns: "
                f"{', '.join(str(column) for column in missing_source_columns)}"
            )

        for column in before_columns:
            if column in after_column_set and not before[column].equals(after[column]):
                self._warn(
                    f"Knowledge package {package_id} attempted to modify source column: {column}"
                )

        undeclared_columns = [
            column
            for column in after.columns
            if column not in before_column_set and column not in package.produced_columns
        ]
        for column in undeclared_columns:
            self._warn(
                f"Knowledge package {package_id} produced undeclared column skipped: {column}"
            )

        safe_columns: list[str] = []
        for column in package.produced_columns:
            if column in before_column_set:
                self._warn(
                    f"Knowledge package {package_id} attempted to overwrite existing column: {column}"
                )
                continue
            if column not in after_column_set:
                self._warn(
                    f"Knowledge package {package_id} did not produce declared column: {column}"
                )
                continue
            safe_columns.append(column)
        return safe_columns

    def _warn(self, warning: str) -> None:
        self.warnings.append(warning)

    def _extend_package_warnings(self, package: BasePackage) -> None:
        for warning in package.warnings:
            self._warn(warning)

    def _normalized_unknown_values(self, package: BasePackage) -> list[str]:
        values = {
            str(value).strip()
            for value in getattr(package, "unknown_values", [])
            if value is not None and str(value).strip()
        }
        return sorted(values)
