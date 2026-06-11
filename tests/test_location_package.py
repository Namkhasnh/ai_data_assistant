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


class SalaryPackage(BasePackage):
    package_id = "salary"
    name = "Salary Package"
    priority = 300
    required_columns = ("salary",)
    produced_columns = ("salary_band",)

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
    ) -> pd.DataFrame:
        output = dataframe.copy(deep=True)
        output["salary_band"] = output["salary"].map({"20-30 triệu": "mid"})
        return output


def test_location_package_generates_semantic_columns_through_engine(
    tmp_path: Path,
) -> None:
    provinces_path = _write_location_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "location": [
                "Quận Cầu Giấy, Hà Nội",
                "Quận 1, Hồ Chí Minh",
                "Atlantis",
            ],
            "standardized_location": [
                "Hà Nội",
                "HCM",
                "Atlantis",
            ],
        }
    )
    original = dataframe.copy(deep=True)

    result = _engine_with_location_package().apply_packages(
        dataframe,
        package_names=["location"],
        package_configs={"location": {"provinces_file": str(provinces_path)}},
        runtime_context=_location_runtime_context(),
    )

    assert list(result.dataframe.columns) == [
        "location",
        "standardized_location",
        "city",
        "province",
        "region",
        "country",
    ]
    assert result.dataframe.loc[0, "location"] == "Quận Cầu Giấy, Hà Nội"
    assert result.dataframe.loc[0, "standardized_location"] == "Hà Nội"
    assert result.dataframe.loc[0, "city"] == "Hà Nội"
    assert result.dataframe.loc[0, "province"] == "Hà Nội"
    assert result.dataframe.loc[0, "region"] == "North"
    assert result.dataframe.loc[0, "country"] == "Vietnam"
    assert result.dataframe.loc[1, "city"] == "TP Hồ Chí Minh"
    assert result.dataframe.loc[1, "province"] == "TP Hồ Chí Minh"
    assert result.dataframe.loc[1, "region"] == "South"
    assert result.dataframe.loc[1, "country"] == "Vietnam"
    assert pd.isna(result.dataframe.loc[2, "city"])
    assert pd.isna(result.dataframe.loc[2, "province"])
    assert pd.isna(result.dataframe.loc[2, "region"])
    assert pd.isna(result.dataframe.loc[2, "country"])
    assert result.report.applied_packages == ["location"]
    assert result.report.produced_columns_by_package == {
        "location": ["city", "province", "region", "country"]
    }
    assert result.report.unknown_values_by_package == {"location": ["Atlantis"]}
    assert result.report.warnings.count(
        "Unknown values encountered in package 'location'."
    ) == 1
    pd.testing.assert_frame_equal(dataframe, original)


def test_location_package_does_not_overwrite_existing_columns(tmp_path: Path) -> None:
    provinces_path = _write_location_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "location": ["Quận Cầu Giấy, Hà Nội"],
            "standardized_location": ["Hà Nội"],
            "region": ["Existing Region"],
        }
    )

    result = _engine_with_location_package().apply_packages(
        dataframe,
        package_configs={"location": {"provinces_file": str(provinces_path)}},
        runtime_context=_location_runtime_context(),
    )

    assert result.dataframe.loc[0, "region"] == "Existing Region"
    assert result.dataframe.loc[0, "city"] == "Hà Nội"
    assert result.dataframe.loc[0, "province"] == "Hà Nội"
    assert result.dataframe.loc[0, "country"] == "Vietnam"
    assert result.report.produced_columns_by_package == {
        "location": ["city", "province", "country"]
    }
    assert any("attempted to overwrite existing column: region" in warning for warning in result.report.warnings)


def test_location_package_aggregates_duplicate_unknown_values_once(
    tmp_path: Path,
) -> None:
    provinces_path = _write_location_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "location": ["Atlantis", "Nowhere", "Atlantis"] * 200,
            "standardized_location": ["Atlantis", "Nowhere", "Atlantis"] * 200,
        }
    )

    result = _engine_with_location_package().apply_packages(
        dataframe,
        package_configs={"location": {"provinces_file": str(provinces_path)}},
        runtime_context=_location_runtime_context(),
    )

    assert result.report.unknown_values_by_package == {
        "location": ["Atlantis", "Nowhere"]
    }
    assert result.report.warnings.count(
        "Unknown values encountered in package 'location'."
    ) == 1
    assert all(pd.isna(value) for value in result.dataframe["city"])


def test_location_package_missing_knowledge_file_warns_without_exception(
    tmp_path: Path,
) -> None:
    missing_provinces_path = tmp_path / "missing_provinces.json"
    dataframe = pd.DataFrame(
        {
            "location": ["Quận Cầu Giấy, Hà Nội"],
            "standardized_location": ["Hà Nội"],
        }
    )

    result = _engine_with_location_package().apply_packages(
        dataframe,
        package_configs={"location": {"provinces_file": str(missing_provinces_path)}},
        runtime_context=_location_runtime_context(),
    )

    assert result.report.applied_packages == ["location"]
    assert any("missing knowledge file" in warning for warning in result.report.warnings)
    assert result.report.unknown_values_by_package == {"location": ["Hà Nội"]}
    assert pd.isna(result.dataframe.loc[0, "city"])
    assert pd.isna(result.dataframe.loc[0, "province"])
    assert pd.isna(result.dataframe.loc[0, "region"])
    assert pd.isna(result.dataframe.loc[0, "country"])


def test_location_package_metadata_matches_phase_9c_contract() -> None:
    metadata = LocationPackage().metadata

    assert metadata.package_id == "location"
    assert metadata.name == "Location Package"
    assert metadata.description == (
        "Generate semantic location attributes from standardized locations."
    )
    assert metadata.version == "1.0"
    assert metadata.enabled is True
    assert metadata.priority == 200
    assert metadata.required_columns == []
    assert metadata.produced_columns == ["city", "province", "region", "country"]


def test_package_execution_order_is_deterministic_with_job_title_and_location(
    tmp_path: Path,
) -> None:
    provinces_path = _write_location_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "title": ["Senior CV Engineer"],
            "standardized_title": ["Computer Vision Engineer"],
            "location": ["Quận Cầu Giấy, Hà Nội"],
            "standardized_location": ["Hà Nội"],
        }
    )
    registry = PackageRegistry()
    registry.register(LocationPackage())
    registry.register(JobTitlePackage())

    result = KnowledgePackageEngine(registry).apply_packages(
        dataframe,
        package_configs={
            "job_title": {},
            "location": {"provinces_file": str(provinces_path)},
        },
        runtime_context={
            "semantic_columns": {
                "JOB_TITLE": ["title"],
                "STANDARDIZED_JOB_TITLE": ["standardized_title"],
                "JOB_LOCATION": ["location"],
                "STANDARDIZED_LOCATION": ["standardized_location"],
            }
        },
    )

    assert result.report.applied_packages == ["job_title", "location"]
    assert result.report.produced_columns_by_package == {
        "job_title": ["job_group", "specialization", "seniority", "tech_domain"],
        "location": ["city", "province", "region", "country"],
    }
    assert list(result.dataframe.columns) == [
        "title",
        "standardized_title",
        "location",
        "standardized_location",
        "job_group",
        "specialization",
        "seniority",
        "tech_domain",
        "city",
        "province",
        "region",
        "country",
    ]


def test_salary_package_can_be_added_without_location_or_engine_changes() -> None:
    dataframe = pd.DataFrame({"salary": ["20-30 triệu"]})
    registry = PackageRegistry()
    registry.register(SalaryPackage())

    result = KnowledgePackageEngine(registry).apply_packages(
        dataframe,
        package_names=["salary"],
        package_configs={"salary": {}},
    )

    assert result.report.applied_packages == ["salary"]
    assert result.report.produced_columns_by_package == {"salary": ["salary_band"]}
    assert result.dataframe.loc[0, "salary_band"] == "mid"


def _engine_with_location_package() -> KnowledgePackageEngine:
    registry = PackageRegistry()
    registry.register(LocationPackage())
    return KnowledgePackageEngine(registry)


def _location_runtime_context(
    location_column: str = "location",
    standardized_location_column: str = "standardized_location",
) -> dict[str, dict[str, list[str]]]:
    return {
        "semantic_columns": {
            "JOB_LOCATION": [location_column],
            "STANDARDIZED_LOCATION": [standardized_location_column],
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
                },
                "TP Hồ Chí Minh": {
                    "aliases": ["HCM", "TPHCM", "Ho Chi Minh City"],
                    "city": "TP Hồ Chí Minh",
                    "province": "TP Hồ Chí Minh",
                    "region": "South",
                    "country": "Vietnam",
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return provinces_path
