from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.knowledge_packages.knowledge_package_engine import KnowledgePackageEngine
from core.knowledge_packages.package_registry import PackageRegistry
from core.knowledge_packages.salary_package import SalaryPackage
from core.semantic_resolver.semantic_column_resolver import SemanticColumnResolver
from models.semantic_column_registry import SemanticColumnRegistry


def test_salary_package_semantic_dataset_variants_match() -> None:
    dataset_a = pd.DataFrame(
        {"salary_min": [20], "salary_max": [30], "salary_type": ["range"]}
    )
    dataset_b = pd.DataFrame(
        {"min_salary": [20], "max_salary": [30], "pay_type": ["range"]}
    )
    dataset_c = pd.DataFrame(
        {"Lương tối thiểu": [20], "Lương tối đa": [30], "Loại lương": ["range"]}
    )

    result_a = _run_salary_package(
        dataset_a,
        {
            "SALARY_MIN": ["salary_min"],
            "SALARY_MAX": ["salary_max"],
            "SALARY_TYPE": ["salary_type"],
        },
    )
    result_b = _run_salary_package(
        dataset_b,
        {
            "SALARY_MIN": ["min_salary"],
            "SALARY_MAX": ["max_salary"],
            "SALARY_TYPE": ["pay_type"],
        },
    )
    result_c = _run_salary_package(
        dataset_c,
        {
            "SALARY_MIN": ["Lương tối thiểu"],
            "SALARY_MAX": ["Lương tối đa"],
            "SALARY_TYPE": ["Loại lương"],
        },
    )

    expected = {
        "salary_avg": 25,
        "currency": "VND",
        "salary_unit": "million",
    }
    assert _semantic_output(result_a.dataframe) == expected
    assert _semantic_output(result_b.dataframe) == expected
    assert _semantic_output(result_c.dataframe) == expected


def test_salary_package_has_no_physical_column_assumptions() -> None:
    dataframe = pd.DataFrame(
        {"salary_min": [20], "salary_max": [30], "salary_type": ["range"]}
    )

    result = _run_salary_package(dataframe, {})

    assert list(result.dataframe.columns) == ["salary_min", "salary_max", "salary_type"]
    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["salary"]
    assert result.report.warnings == ["Salary package skipped; no usable semantic columns."]


def test_salary_package_runtime_context_none_skips_gracefully() -> None:
    dataframe = pd.DataFrame(
        {"salary_min": [20], "salary_max": [30], "salary_type": ["range"]}
    )
    registry = PackageRegistry()
    registry.register(SalaryPackage())

    result = KnowledgePackageEngine(registry).apply_packages(
        dataframe,
        package_names=["salary"],
        package_configs={"salary": {}},
        runtime_context=None,
    )

    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["salary"]
    assert result.report.warnings == ["Salary package skipped; no usable semantic columns."]


def test_salary_package_does_not_import_semantic_resolver() -> None:
    source = Path("core/knowledge_packages/salary_package.py").read_text(encoding="utf-8")

    assert "semantic_column_resolver" not in source
    assert "SemanticColumnResolver" not in source


def _run_salary_package(
    dataframe: pd.DataFrame,
    columns_by_semantic_type: dict[str, list[str]],
) -> object:
    package_registry = PackageRegistry()
    package_registry.register(SalaryPackage())
    return KnowledgePackageEngine(package_registry).apply_packages(
        dataframe,
        package_names=["salary"],
        package_configs={"salary": {}},
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
        "salary_avg": dataframe.loc[0, "salary_avg"],
        "currency": dataframe.loc[0, "currency"],
        "salary_unit": dataframe.loc[0, "salary_unit"],
    }
