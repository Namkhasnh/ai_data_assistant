from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from core.knowledge_packages.base_package import BasePackage
from core.knowledge_packages.job_title_package import JobTitlePackage
from core.knowledge_packages.knowledge_package_engine import KnowledgePackageEngine
from core.knowledge_packages.location_package import LocationPackage
from core.knowledge_packages.package_registry import PackageRegistry
from core.knowledge_packages.salary_package import SalaryPackage


class HealthcareTestPackage(BasePackage):
    package_id = "healthcare_test"
    name = "Healthcare Test Package"
    priority = 400
    required_columns = ("test_code",)
    produced_columns = ("test_name",)

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
    ) -> pd.DataFrame:
        output = dataframe.copy(deep=True)
        output["test_name"] = output["test_code"].map({"CBC": "Complete Blood Count"})
        return output


def test_salary_package_generates_semantic_columns_through_engine() -> None:
    dataframe = pd.DataFrame(
        {
            "salary_min": [20, 15, 12, None],
            "salary_max": [30, 15, 12, None],
            "salary_type": ["range", "fixed", "range", "negotiable"],
        }
    )
    original = dataframe.copy(deep=True)

    result = _engine_with_salary_package().apply_packages(
        dataframe,
        package_names=["salary"],
        package_configs={"salary": {}},
        runtime_context=_salary_runtime_context(),
    )

    assert list(result.dataframe.columns) == [
        "salary_min",
        "salary_max",
        "salary_type",
        "salary_avg",
        "currency",
        "salary_unit",
    ]
    assert result.dataframe.loc[0, "salary_avg"] == 25
    assert result.dataframe.loc[1, "salary_avg"] == 15
    assert result.dataframe.loc[2, "salary_avg"] == 12
    assert pd.isna(result.dataframe.loc[3, "salary_avg"])
    assert list(result.dataframe["currency"]) == ["VND", "VND", "VND", "VND"]
    assert list(result.dataframe["salary_unit"]) == [
        "million",
        "million",
        "million",
        "million",
    ]
    assert result.report.applied_packages == ["salary"]
    assert result.report.produced_columns_by_package == {
        "salary": ["salary_avg", "currency", "salary_unit"]
    }
    assert result.report.unknown_values_by_package == {}
    pd.testing.assert_frame_equal(dataframe, original)


def test_salary_package_fixed_salary_uses_salary_min_even_when_max_differs() -> None:
    dataframe = pd.DataFrame(
        {
            "salary_min": [15],
            "salary_max": [20],
            "salary_type": ["fixed"],
        }
    )

    result = _engine_with_salary_package().apply_packages(
        dataframe,
        package_configs={"salary": {}},
        runtime_context=_salary_runtime_context(),
    )

    assert result.dataframe.loc[0, "salary_avg"] == 15
    assert result.dataframe.loc[0, "currency"] == "VND"
    assert result.dataframe.loc[0, "salary_unit"] == "million"


def test_salary_package_does_not_overwrite_existing_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "salary_min": [20],
            "salary_max": [30],
            "salary_type": ["range"],
            "salary_avg": [999],
            "currency": ["Existing Currency"],
        }
    )

    result = _engine_with_salary_package().apply_packages(
        dataframe,
        package_configs={"salary": {}},
        runtime_context=_salary_runtime_context(),
    )

    assert result.dataframe.loc[0, "salary_avg"] == 999
    assert result.dataframe.loc[0, "currency"] == "Existing Currency"
    assert result.dataframe.loc[0, "salary_unit"] == "million"
    assert result.report.produced_columns_by_package == {"salary": ["salary_unit"]}
    assert any("attempted to overwrite existing column: salary_avg" in warning for warning in result.report.warnings)
    assert any("attempted to overwrite existing column: currency" in warning for warning in result.report.warnings)


def test_salary_package_missing_required_columns_are_handled_by_engine() -> None:
    dataframe = pd.DataFrame(
        {
            "salary_min": [20],
            "salary_max": [30],
        }
    )

    result = _engine_with_salary_package().apply_packages(
        dataframe,
        package_configs={"salary": {}},
        runtime_context={
            "semantic_columns": {
                "SALARY_MIN": ["salary_min"],
                "SALARY_MAX": ["salary_max"],
                "SALARY_TYPE": ["salary_type"],
            }
        },
    )

    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["salary"]
    assert result.report.warnings == ["Salary package skipped; no usable semantic columns."]
    assert list(result.dataframe.columns) == ["salary_min", "salary_max"]


def test_package_execution_order_is_deterministic_with_job_location_and_salary(
    tmp_path: Path,
) -> None:
    provinces_path = _write_location_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "title": ["Senior CV Engineer"],
            "standardized_title": ["Computer Vision Engineer"],
            "location": ["Quận Cầu Giấy, Hà Nội"],
            "standardized_location": ["Hà Nội"],
            "salary_min": [20],
            "salary_max": [30],
            "salary_type": ["range"],
        }
    )
    registry = PackageRegistry()
    registry.register(SalaryPackage())
    registry.register(LocationPackage())
    registry.register(JobTitlePackage())

    result = KnowledgePackageEngine(registry).apply_packages(
        dataframe,
        package_configs={
            "job_title": {},
            "location": {"provinces_file": str(provinces_path)},
            "salary": {},
        },
        runtime_context={
            "semantic_columns": {
                "JOB_TITLE": ["title"],
                "STANDARDIZED_JOB_TITLE": ["standardized_title"],
                "JOB_LOCATION": ["location"],
                "STANDARDIZED_LOCATION": ["standardized_location"],
                "SALARY_MIN": ["salary_min"],
                "SALARY_MAX": ["salary_max"],
                "SALARY_TYPE": ["salary_type"],
            }
        },
    )

    assert result.report.applied_packages == ["job_title", "location", "salary"]
    assert result.report.produced_columns_by_package == {
        "job_title": ["job_group", "specialization", "seniority", "tech_domain"],
        "location": ["city", "province", "region", "country"],
        "salary": ["salary_avg", "currency", "salary_unit"],
    }
    assert list(result.dataframe.columns) == [
        "title",
        "standardized_title",
        "location",
        "standardized_location",
        "salary_min",
        "salary_max",
        "salary_type",
        "job_group",
        "specialization",
        "seniority",
        "tech_domain",
        "city",
        "province",
        "region",
        "country",
        "salary_avg",
        "currency",
        "salary_unit",
    ]


def test_salary_package_metadata_matches_phase_9d_contract() -> None:
    metadata = SalaryPackage().metadata

    assert metadata.package_id == "salary"
    assert metadata.name == "Salary Package"
    assert metadata.description == (
        "Generate semantic business attributes from standardized salary columns."
    )
    assert metadata.version == "1.0"
    assert metadata.enabled is True
    assert metadata.priority == 300
    assert metadata.required_columns == []
    assert metadata.produced_columns == ["salary_avg", "currency", "salary_unit"]


def test_healthcare_test_package_can_be_added_without_salary_or_engine_changes() -> None:
    dataframe = pd.DataFrame({"test_code": ["CBC"]})
    registry = PackageRegistry()
    registry.register(HealthcareTestPackage())

    result = KnowledgePackageEngine(registry).apply_packages(
        dataframe,
        package_names=["healthcare_test"],
        package_configs={"healthcare_test": {}},
    )

    assert result.report.applied_packages == ["healthcare_test"]
    assert result.report.produced_columns_by_package == {
        "healthcare_test": ["test_name"]
    }
    assert result.dataframe.loc[0, "test_name"] == "Complete Blood Count"


def _engine_with_salary_package() -> KnowledgePackageEngine:
    registry = PackageRegistry()
    registry.register(SalaryPackage())
    return KnowledgePackageEngine(registry)


def _salary_runtime_context(
    salary_min_column: str = "salary_min",
    salary_max_column: str = "salary_max",
    salary_type_column: str = "salary_type",
) -> dict[str, dict[str, list[str]]]:
    return {
        "semantic_columns": {
            "SALARY_MIN": [salary_min_column],
            "SALARY_MAX": [salary_max_column],
            "SALARY_TYPE": [salary_type_column],
        }
    }


def _write_location_knowledge(tmp_path: Path) -> Path:
    provinces_path = tmp_path / "provinces.json"
    provinces_path.write_text(
        json.dumps(
            {
                "Hà Nội": {
                    "aliases": ["Ha Noi", "Hanoi"],
                    "city": "Hà Nội",
                    "province": "Hà Nội",
                    "region": "North",
                    "country": "Vietnam",
                }
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return provinces_path
