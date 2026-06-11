from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.knowledge_packages.knowledge_package_engine import KnowledgePackageEngine
from core.knowledge_packages.location_package import LocationPackage
from core.knowledge_packages.package_registry import PackageRegistry
from core.semantic_resolver.semantic_column_resolver import SemanticColumnResolver
from models.semantic_column_registry import SemanticColumnRegistry


def test_location_package_semantic_dataset_variants_match() -> None:
    dataset_a = pd.DataFrame(
        {"location": ["Quận Cầu Giấy, Hà Nội"], "standardized_location": ["Hà Nội"]}
    )
    dataset_b = pd.DataFrame(
        {"city_name": ["Quận Cầu Giấy, Hà Nội"], "canonical_city_name": ["Hà Nội"]}
    )
    dataset_c = pd.DataFrame(
        {"Địa điểm": ["Quận Cầu Giấy, Hà Nội"], "Địa điểm chuẩn hóa": ["Hà Nội"]}
    )

    result_a = _run_location_package(
        dataset_a,
        {
            "JOB_LOCATION": ["location"],
            "STANDARDIZED_LOCATION": ["standardized_location"],
        },
    )
    result_b = _run_location_package(
        dataset_b,
        {
            "JOB_LOCATION": ["city_name"],
            "STANDARDIZED_LOCATION": ["canonical_city_name"],
        },
    )
    result_c = _run_location_package(
        dataset_c,
        {
            "JOB_LOCATION": ["Địa điểm"],
            "STANDARDIZED_LOCATION": ["Địa điểm chuẩn hóa"],
        },
    )

    expected = {
        "city": "Hà Nội",
        "province": "Hà Nội",
        "region": "North",
        "country": "Vietnam",
    }
    assert _semantic_output(result_a.dataframe) == expected
    assert _semantic_output(result_b.dataframe) == expected
    assert _semantic_output(result_c.dataframe) == expected


def test_location_package_has_no_physical_column_assumptions() -> None:
    dataframe = pd.DataFrame(
        {"location": ["Quận Cầu Giấy, Hà Nội"], "standardized_location": ["Hà Nội"]}
    )

    result = _run_location_package(dataframe, {})

    assert list(result.dataframe.columns) == ["location", "standardized_location"]
    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["location"]
    assert result.report.warnings == ["Location package skipped; no usable semantic columns."]


def test_location_package_runtime_context_none_skips_gracefully() -> None:
    dataframe = pd.DataFrame(
        {"location": ["Quận Cầu Giấy, Hà Nội"], "standardized_location": ["Hà Nội"]}
    )
    registry = PackageRegistry()
    registry.register(LocationPackage())

    result = KnowledgePackageEngine(registry).apply_packages(
        dataframe,
        package_names=["location"],
        package_configs={"location": {}},
        runtime_context=None,
    )

    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["location"]
    assert result.report.warnings == ["Location package skipped; no usable semantic columns."]


def test_location_package_does_not_import_semantic_resolver() -> None:
    source = Path("core/knowledge_packages/location_package.py").read_text(encoding="utf-8")

    assert "semantic_column_resolver" not in source
    assert "SemanticColumnResolver" not in source


def _run_location_package(
    dataframe: pd.DataFrame,
    columns_by_semantic_type: dict[str, list[str]],
) -> object:
    package_registry = PackageRegistry()
    package_registry.register(LocationPackage())
    return KnowledgePackageEngine(package_registry).apply_packages(
        dataframe,
        package_names=["location"],
        package_configs={"location": {}},
        runtime_context=_runtime_context(columns_by_semantic_type),
    )


def _runtime_context(columns_by_semantic_type: dict[str, list[str]]) -> dict[str, dict[str, list[str]]]:
    resolver = SemanticColumnResolver(
        SemanticColumnRegistry(columns_by_semantic_type=columns_by_semantic_type)
    )
    return {
        "semantic_columns": {
            semantic_type: resolver.get_columns([semantic_type])
            for semantic_type in resolver.available_semantic_types()
        }
    }


def _semantic_output(dataframe: pd.DataFrame) -> dict[str, object]:
    return {
        "city": dataframe.loc[0, "city"],
        "province": dataframe.loc[0, "province"],
        "region": dataframe.loc[0, "region"],
        "country": dataframe.loc[0, "country"],
    }
