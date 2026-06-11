from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.knowledge_packages.job_title_package import JobTitlePackage
from core.knowledge_packages.knowledge_package_engine import KnowledgePackageEngine
from core.knowledge_packages.package_registry import PackageRegistry
from core.semantic_resolver.semantic_column_resolver import SemanticColumnResolver
from models.semantic_column_registry import SemanticColumnRegistry


def test_job_title_package_semantic_dataset_variants_match() -> None:
    dataset_a = pd.DataFrame(
        {"title": ["Senior CV Engineer"], "standardized_title": ["Computer Vision Engineer"]}
    )
    dataset_b = pd.DataFrame(
        {"job_name": ["Senior CV Engineer"], "canonical_job_name": ["Computer Vision Engineer"]}
    )
    dataset_c = pd.DataFrame(
        {"Tiêu đề": ["Senior CV Engineer"], "Tiêu đề chuẩn hóa": ["Computer Vision Engineer"]}
    )

    result_a = _run_job_title_package(
        dataset_a,
        {
            "JOB_TITLE": ["title"],
            "STANDARDIZED_JOB_TITLE": ["standardized_title"],
        },
    )
    result_b = _run_job_title_package(
        dataset_b,
        {
            "JOB_TITLE": ["job_name"],
            "STANDARDIZED_JOB_TITLE": ["canonical_job_name"],
        },
    )
    result_c = _run_job_title_package(
        dataset_c,
        {
            "JOB_TITLE": ["Tiêu đề"],
            "STANDARDIZED_JOB_TITLE": ["Tiêu đề chuẩn hóa"],
        },
    )

    expected = {
        "job_group": "AI Engineer",
        "specialization": "Computer Vision",
        "seniority": "Senior",
        "tech_domain": "AI",
    }
    assert _semantic_output(result_a.dataframe) == expected
    assert _semantic_output(result_b.dataframe) == expected
    assert _semantic_output(result_c.dataframe) == expected
    assert result_a.report.produced_columns_by_package == {
        "job_title": ["job_group", "specialization", "seniority", "tech_domain"]
    }


def test_job_title_package_has_no_physical_column_assumptions() -> None:
    dataframe = pd.DataFrame(
        {"title": ["Senior CV Engineer"], "standardized_title": ["Computer Vision Engineer"]}
    )

    result = _run_job_title_package(dataframe, {})

    assert list(result.dataframe.columns) == ["title", "standardized_title"]
    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["job_title"]
    assert result.report.warnings == ["Job title package skipped; no usable semantic columns."]


def test_job_title_package_runtime_context_none_skips_gracefully() -> None:
    dataframe = pd.DataFrame(
        {"title": ["Senior CV Engineer"], "standardized_title": ["Computer Vision Engineer"]}
    )
    registry = PackageRegistry()
    registry.register(JobTitlePackage())

    result = KnowledgePackageEngine(registry).apply_packages(
        dataframe,
        package_names=["job_title"],
        package_configs={"job_title": {}},
        runtime_context=None,
    )

    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["job_title"]
    assert result.report.warnings == ["Job title package skipped; no usable semantic columns."]


def test_job_title_package_does_not_import_semantic_resolver() -> None:
    source = Path("core/knowledge_packages/job_title_package.py").read_text(encoding="utf-8")

    assert "semantic_column_resolver" not in source
    assert "SemanticColumnResolver" not in source


def _run_job_title_package(
    dataframe: pd.DataFrame,
    columns_by_semantic_type: dict[str, list[str]],
) -> object:
    package_registry = PackageRegistry()
    package_registry.register(JobTitlePackage())
    return KnowledgePackageEngine(package_registry).apply_packages(
        dataframe,
        package_names=["job_title"],
        package_configs={"job_title": {}},
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
        "job_group": dataframe.loc[0, "job_group"],
        "specialization": dataframe.loc[0, "specialization"],
        "seniority": dataframe.loc[0, "seniority"],
        "tech_domain": dataframe.loc[0, "tech_domain"],
    }
