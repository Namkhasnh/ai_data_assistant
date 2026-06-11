from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from core.knowledge_packages.base_package import BasePackage
from core.knowledge_packages.industry_domain_package import IndustryDomainPackage
from core.knowledge_packages.job_title_package import JobTitlePackage
from core.knowledge_packages.knowledge_package_engine import KnowledgePackageEngine
from core.knowledge_packages.location_package import LocationPackage
from core.knowledge_packages.package_registry import PackageRegistry
from core.knowledge_packages.salary_package import SalaryPackage
from core.knowledge_packages.skill_package import SkillPackage
from core.knowledge_packages.work_mode_package import WorkModePackage


class BankingAbbreviationPackage(BasePackage):
    package_id = "banking_abbreviation"
    name = "Banking Abbreviation Package"
    priority = 50
    required_columns: tuple[str, ...] = ()
    produced_columns = ("banking_category",)

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        output = dataframe.copy(deep=True)
        output["banking_category"] = "General"
        return output


def test_industry_domain_package_appends_domain_from_semantic_columns(
    tmp_path: Path,
) -> None:
    industry_domains_path = _write_industry_domain_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "company_name": ["Vietcombank"],
            "job_description": ["Basel II credit risk scoring."],
        }
    )
    original = dataframe.copy(deep=True)

    result = _engine_with_industry_domain_package().apply_packages(
        dataframe,
        package_names=["industry_domain"],
        package_configs={
            "industry_domain": {
                "industry_domains_file": str(industry_domains_path)
            }
        },
        runtime_context=_runtime_context(
            company_names=["company_name"],
            descriptions=["job_description"],
        ),
    )

    assert list(result.dataframe.columns) == [
        "company_name",
        "job_description",
        "industry_domain",
    ]
    assert result.dataframe.loc[0, "industry_domain"] == "Banking"
    assert result.report.applied_packages == ["industry_domain"]
    assert result.report.produced_columns_by_package == {
        "industry_domain": ["industry_domain"]
    }
    assert result.report.unknown_values_by_package == {}
    assert result.report.warnings == []
    pd.testing.assert_frame_equal(dataframe, original)


def test_industry_domain_package_does_not_overwrite_existing_domain(
    tmp_path: Path,
) -> None:
    industry_domains_path = _write_industry_domain_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "company_name": ["Vinmec"],
            "industry_domain": ["Existing Domain"],
        }
    )

    result = _engine_with_industry_domain_package().apply_packages(
        dataframe,
        package_configs={
            "industry_domain": {
                "industry_domains_file": str(industry_domains_path)
            }
        },
        runtime_context=_runtime_context(company_names=["company_name"]),
    )

    assert result.dataframe.loc[0, "industry_domain"] == "Existing Domain"
    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["industry_domain"]
    assert any(
        "attempted to overwrite existing column: industry_domain" in warning
        for warning in result.report.warnings
    )


def test_industry_domain_package_skips_without_usable_semantic_columns() -> None:
    dataframe = pd.DataFrame({"company_name": ["Vietcombank"]})

    result = _engine_with_industry_domain_package().apply_packages(
        dataframe,
        package_configs={"industry_domain": {}},
        runtime_context={"semantic_columns": {}},
    )

    assert list(result.dataframe.columns) == ["company_name"]
    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["industry_domain"]
    assert result.report.warnings == [
        "Industry domain package skipped; no usable semantic columns."
    ]


def test_industry_domain_package_runtime_context_none_skips_gracefully() -> None:
    dataframe = pd.DataFrame({"company_name": ["Vietcombank"]})

    result = _engine_with_industry_domain_package().apply_packages(
        dataframe,
        package_names=["industry_domain"],
        package_configs={"industry_domain": {}},
        runtime_context=None,
    )

    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["industry_domain"]
    assert result.report.warnings == [
        "Industry domain package skipped; no usable semantic columns."
    ]


def test_industry_domain_package_matches_case_insensitive_text(
    tmp_path: Path,
) -> None:
    industry_domains_path = _write_industry_domain_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "job_description": [
                "HOSPITAL information systems for PATIENT records at VINMEC."
            ]
        }
    )

    result = _engine_with_industry_domain_package().apply_packages(
        dataframe,
        package_configs={
            "industry_domain": {
                "industry_domains_file": str(industry_domains_path)
            }
        },
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "industry_domain"] == "Healthcare"


def test_industry_domain_package_matches_company_name(tmp_path: Path) -> None:
    industry_domains_path = _write_industry_domain_knowledge(tmp_path)
    dataframe = pd.DataFrame({"company": ["Shopee"]})

    result = _run_industry_domain_package(
        dataframe=dataframe,
        industry_domains_path=industry_domains_path,
        runtime_context=_runtime_context(company_names=["company"]),
    )

    assert result.dataframe.loc[0, "industry_domain"] == "E-commerce"


def test_industry_domain_package_matches_title(tmp_path: Path) -> None:
    industry_domains_path = _write_industry_domain_knowledge(tmp_path)
    dataframe = pd.DataFrame({"title": ["Insurance Data Engineer"]})

    result = _run_industry_domain_package(
        dataframe=dataframe,
        industry_domains_path=industry_domains_path,
        runtime_context=_runtime_context(titles=["title"]),
    )

    assert result.dataframe.loc[0, "industry_domain"] == "Insurance"


def test_industry_domain_package_matches_description(tmp_path: Path) -> None:
    industry_domains_path = _write_industry_domain_knowledge(tmp_path)
    dataframe = pd.DataFrame({"description": ["Hospital patient records."]})

    result = _run_industry_domain_package(
        dataframe=dataframe,
        industry_domains_path=industry_domains_path,
        runtime_context=_runtime_context(descriptions=["description"]),
    )

    assert result.dataframe.loc[0, "industry_domain"] == "Healthcare"


def test_industry_domain_package_matches_benefits(tmp_path: Path) -> None:
    industry_domains_path = _write_industry_domain_knowledge(tmp_path)
    dataframe = pd.DataFrame({"benefits": ["University education support."]})

    result = _run_industry_domain_package(
        dataframe=dataframe,
        industry_domains_path=industry_domains_path,
        runtime_context=_runtime_context(benefits=["benefits"]),
    )

    assert result.dataframe.loc[0, "industry_domain"] == "Education"


def test_industry_domain_package_matches_work_location(tmp_path: Path) -> None:
    industry_domains_path = _write_industry_domain_knowledge(tmp_path)
    dataframe = pd.DataFrame({"work_location": ["Warehouse near Binh Duong."]})

    result = _run_industry_domain_package(
        dataframe=dataframe,
        industry_domains_path=industry_domains_path,
        runtime_context=_runtime_context(work_locations=["work_location"]),
    )

    assert result.dataframe.loc[0, "industry_domain"] == "Logistics"


def test_industry_domain_package_uses_multi_hit_voting(tmp_path: Path) -> None:
    industry_domains_path = _write_industry_domain_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {"job_description": ["Hospital patient insurance reimbursement."]}
    )

    result = _run_industry_domain_package(
        dataframe=dataframe,
        industry_domains_path=industry_domains_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "industry_domain"] == "Healthcare"


def test_industry_domain_package_resolves_ties_alphabetically(
    tmp_path: Path,
) -> None:
    industry_domains_path = _write_industry_domain_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {"job_description": ["Hospital patient insurance Prudential."]}
    )

    result = _run_industry_domain_package(
        dataframe=dataframe,
        industry_domains_path=industry_domains_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "industry_domain"] == "Healthcare"


def test_industry_domain_package_deduplicates_text_inputs(
    tmp_path: Path,
) -> None:
    industry_domains_path = _write_industry_domain_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "company_name": ["Vietcombank"],
            "job_description": ["Vietcombank"],
            "benefits": ["Vietcombank"],
        }
    )

    result = _run_industry_domain_package(
        dataframe=dataframe,
        industry_domains_path=industry_domains_path,
        runtime_context=_runtime_context(
            company_names=["company_name"],
            descriptions=["job_description"],
            benefits=["benefits"],
        ),
    )

    assert result.dataframe.loc[0, "industry_domain"] == "Banking"
    assert result.report.warnings == []


def test_industry_domain_package_unknown_text_remains_quiet(
    tmp_path: Path,
) -> None:
    industry_domains_path = _write_industry_domain_knowledge(tmp_path)
    dataframe = pd.DataFrame({"job_description": ["No industry signals here."]})

    result = _run_industry_domain_package(
        dataframe=dataframe,
        industry_domains_path=industry_domains_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert pd.isna(result.dataframe.loc[0, "industry_domain"])
    assert result.report.warnings == []
    assert result.report.unknown_values_by_package == {}


def test_industry_domain_package_output_is_deterministic(tmp_path: Path) -> None:
    industry_domains_path = _write_industry_domain_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "company_name": ["Vietcombank"],
            "job_description": ["Basel II credit risk scoring."],
        }
    )
    kwargs = {
        "dataframe": dataframe,
        "industry_domains_path": industry_domains_path,
        "runtime_context": _runtime_context(
            company_names=["company_name"],
            descriptions=["job_description"],
        ),
    }

    first = _run_industry_domain_package(**kwargs)
    second = _run_industry_domain_package(**kwargs)

    pd.testing.assert_frame_equal(first.dataframe, second.dataframe)
    assert first.report == second.report


def test_industry_domain_package_metadata_matches_phase_10_4e_contract() -> None:
    package = IndustryDomainPackage()
    metadata = package.metadata

    assert package.package_id == "industry_domain"
    assert package.name == "Industry Domain Package"
    assert package.description == (
        "Generate industry domain attributes from job text columns."
    )
    assert package.version == "1.0"
    assert package.enabled is True
    assert package.priority == 600
    assert package.required_columns == ()
    assert package.produced_columns == ("industry_domain",)
    assert metadata.package_id == "industry_domain"
    assert metadata.produced_columns == ["industry_domain"]


def test_industry_domain_package_does_not_import_semantic_resolver() -> None:
    source = Path("core/knowledge_packages/industry_domain_package.py").read_text(
        encoding="utf-8"
    )

    assert "semantic_column_resolver" not in source
    assert "SemanticColumnResolver" not in source


def test_current_package_order_includes_industry_domain() -> None:
    registry = PackageRegistry()
    registry.register(IndustryDomainPackage())
    registry.register(WorkModePackage())
    registry.register(SkillPackage())
    registry.register(SalaryPackage())
    registry.register(LocationPackage())
    registry.register(JobTitlePackage())

    assert [package.package_id for package in registry.list_packages()] == [
        "job_title",
        "location",
        "salary",
        "skill",
        "work_mode",
        "industry_domain",
    ]


def test_unrelated_package_can_be_added_without_industry_or_engine_changes() -> None:
    dataframe = pd.DataFrame({"bank_code": ["VCB"]})
    registry = PackageRegistry()
    registry.register(BankingAbbreviationPackage())

    result = KnowledgePackageEngine(registry).apply_packages(
        dataframe,
        package_names=["banking_abbreviation"],
        package_configs={"banking_abbreviation": {}},
    )

    assert result.report.applied_packages == ["banking_abbreviation"]
    assert result.report.produced_columns_by_package == {
        "banking_abbreviation": ["banking_category"]
    }
    assert result.dataframe.loc[0, "banking_category"] == "General"


def _engine_with_industry_domain_package() -> KnowledgePackageEngine:
    registry = PackageRegistry()
    registry.register(IndustryDomainPackage())
    return KnowledgePackageEngine(registry)


def _run_industry_domain_package(
    dataframe: pd.DataFrame,
    industry_domains_path: Path,
    runtime_context: dict[str, dict[str, list[str]]],
) -> object:
    return _engine_with_industry_domain_package().apply_packages(
        dataframe,
        package_configs={
            "industry_domain": {
                "industry_domains_file": str(industry_domains_path)
            }
        },
        runtime_context=runtime_context,
    )


def _runtime_context(
    company_names: list[str] | None = None,
    titles: list[str] | None = None,
    descriptions: list[str] | None = None,
    benefits: list[str] | None = None,
    work_locations: list[str] | None = None,
) -> dict[str, dict[str, list[str]]]:
    return {
        "semantic_columns": {
            "COMPANY_NAME": company_names or [],
            "JOB_TITLE": titles or [],
            "JOB_DESCRIPTION": descriptions or [],
            "BENEFITS": benefits or [],
            "WORK_LOCATION": work_locations or [],
        }
    }


def _write_industry_domain_knowledge(tmp_path: Path) -> Path:
    industry_domains_path = tmp_path / "industry_domains.json"
    industry_domains_path.write_text(
        json.dumps(
            {
                "Banking": {
                    "aliases": [
                        "agribank",
                        "bank",
                        "banking",
                        "basel ii",
                        "bidv",
                        "credit",
                        "loan",
                        "payment",
                        "vietcombank",
                    ],
                },
                "E-commerce": {
                    "aliases": [
                        "e-commerce",
                        "ecommerce",
                        "lazada",
                        "marketplace",
                        "online retail",
                        "shopee",
                        "tiki",
                    ],
                },
                "Education": {
                    "aliases": [
                        "edtech",
                        "education",
                        "school",
                        "student",
                        "university",
                    ],
                },
                "Healthcare": {
                    "aliases": [
                        "clinic",
                        "healthcare",
                        "hospital",
                        "medical",
                        "patient",
                        "telemedicine",
                        "vinmec",
                    ],
                },
                "Insurance": {
                    "aliases": [
                        "claim",
                        "claims",
                        "insurance",
                        "policy",
                        "premium",
                        "prudential",
                        "underwriting",
                    ],
                },
                "Logistics": {
                    "aliases": [
                        "delivery",
                        "freight",
                        "logistics",
                        "shipping",
                        "warehouse",
                    ],
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return industry_domains_path
