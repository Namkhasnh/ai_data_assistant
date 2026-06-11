from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from core.knowledge_packages.base_package import BasePackage
from core.knowledge_packages.company_size_package import CompanySizePackage
from core.knowledge_packages.education_package import EducationPackage
from core.knowledge_packages.employment_type_package import EmploymentTypePackage
from core.knowledge_packages.experience_level_package import ExperienceLevelPackage
from core.knowledge_packages.industry_domain_package import IndustryDomainPackage
from core.knowledge_packages.job_title_package import JobTitlePackage
from core.knowledge_packages.knowledge_package_engine import KnowledgePackageEngine
from core.knowledge_packages.language_package import LanguagePackage
from core.knowledge_packages.location_package import LocationPackage
from core.knowledge_packages.package_registry import PackageRegistry
from core.knowledge_packages.salary_package import SalaryPackage
from core.knowledge_packages.skill_package import SkillPackage
from core.knowledge_packages.work_mode_package import WorkModePackage


class FakePackage(BasePackage):
    package_id = "fake"
    name = "Fake Package"
    priority = 50
    required_columns: tuple[str, ...] = ()
    produced_columns = ("fake_column",)

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        output = dataframe.copy(deep=True)
        output["fake_column"] = "fake"
        return output


def test_company_size_package_appends_company_size_group(tmp_path: Path) -> None:
    company_size_groups_path = _write_company_size_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "company_name": ["Example Startup"],
            "company_description": ["15 employees building a SaaS product."],
        }
    )
    original = dataframe.copy(deep=True)

    result = _engine_with_company_size_package().apply_packages(
        dataframe,
        package_names=["company_size"],
        package_configs={
            "company_size": {
                "company_size_groups_file": str(company_size_groups_path)
            }
        },
        runtime_context=_runtime_context(
            company_names=["company_name"],
            company_descriptions=["company_description"],
        ),
    )

    assert list(result.dataframe.columns) == [
        "company_name",
        "company_description",
        "company_size_group",
    ]
    assert result.dataframe.loc[0, "company_size_group"] == "Startup"
    assert result.report.applied_packages == ["company_size"]
    assert result.report.produced_columns_by_package == {
        "company_size": ["company_size_group"]
    }
    assert result.report.unknown_values_by_package == {}
    assert result.report.warnings == []
    pd.testing.assert_frame_equal(dataframe, original)


def test_company_size_package_does_not_overwrite_existing_column(
    tmp_path: Path,
) -> None:
    company_size_groups_path = _write_company_size_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "company_description": ["300 employees."],
            "company_size_group": ["Existing Size"],
        }
    )

    result = _engine_with_company_size_package().apply_packages(
        dataframe,
        package_configs={
            "company_size": {
                "company_size_groups_file": str(company_size_groups_path)
            }
        },
        runtime_context=_runtime_context(company_descriptions=["company_description"]),
    )

    assert result.dataframe.loc[0, "company_size_group"] == "Existing Size"
    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["company_size"]
    assert any(
        "attempted to overwrite existing column: company_size_group" in warning
        for warning in result.report.warnings
    )


def test_company_size_package_skips_without_usable_semantic_columns() -> None:
    dataframe = pd.DataFrame({"company_description": ["15 employees."]})

    result = _engine_with_company_size_package().apply_packages(
        dataframe,
        package_configs={"company_size": {}},
        runtime_context={"semantic_columns": {}},
    )

    assert list(result.dataframe.columns) == ["company_description"]
    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["company_size"]
    assert result.report.warnings == [
        "Company size package skipped; no usable semantic columns."
    ]


def test_company_size_package_runtime_context_none_skips_gracefully() -> None:
    dataframe = pd.DataFrame({"company_description": ["15 employees."]})

    result = _engine_with_company_size_package().apply_packages(
        dataframe,
        package_names=["company_size"],
        package_configs={"company_size": {}},
        runtime_context=None,
    )

    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["company_size"]
    assert result.report.warnings == [
        "Company size package skipped; no usable semantic columns."
    ]


def test_company_size_package_extracts_numeric_employee_groups(
    tmp_path: Path,
) -> None:
    company_size_groups_path = _write_company_size_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "company_description": [
                "15 employees",
                "80 employees",
                "300 employees",
                "1000 employees",
                "1000+ employees",
            ]
        }
    )

    result = _run_company_size_package(
        dataframe=dataframe,
        company_size_groups_path=company_size_groups_path,
        runtime_context=_runtime_context(company_descriptions=["company_description"]),
    )

    assert result.dataframe["company_size_group"].tolist() == [
        "Startup",
        "Small",
        "Medium",
        "Large",
        "Enterprise",
    ]


def test_company_size_package_matches_case_insensitive_aliases(
    tmp_path: Path,
) -> None:
    company_size_groups_path = _write_company_size_knowledge(tmp_path)
    dataframe = pd.DataFrame({"company_description": ["GLOBAL MULTINATIONAL COMPANY"]})

    result = _run_company_size_package(
        dataframe=dataframe,
        company_size_groups_path=company_size_groups_path,
        runtime_context=_runtime_context(company_descriptions=["company_description"]),
    )

    assert result.dataframe.loc[0, "company_size_group"] == "Enterprise"


def test_company_size_package_matches_aliases(tmp_path: Path) -> None:
    company_size_groups_path = _write_company_size_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "company_description": [
                "Global multinational company",
                "Early-stage startup",
                "Small business",
                "Mid-sized technology company",
                "Large-scale operation",
            ]
        }
    )

    result = _run_company_size_package(
        dataframe=dataframe,
        company_size_groups_path=company_size_groups_path,
        runtime_context=_runtime_context(company_descriptions=["company_description"]),
    )

    assert result.dataframe["company_size_group"].tolist() == [
        "Enterprise",
        "Startup",
        "Small",
        "Medium",
        "Large",
    ]


def test_company_size_package_numeric_signal_overrides_aliases(
    tmp_path: Path,
) -> None:
    company_size_groups_path = _write_company_size_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "company_description": [
                "Global multinational company with 80 employees",
                "Startup with 300 employees",
                "Multinational enterprise with 15 employees",
            ]
        }
    )

    result = _run_company_size_package(
        dataframe=dataframe,
        company_size_groups_path=company_size_groups_path,
        runtime_context=_runtime_context(company_descriptions=["company_description"]),
    )

    assert result.dataframe["company_size_group"].tolist() == [
        "Small",
        "Medium",
        "Startup",
    ]


def test_company_size_package_reads_all_semantic_input_types(tmp_path: Path) -> None:
    company_size_groups_path = _write_company_size_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "company_name": ["Seed-stage Labs"],
            "company_description": ["Global company"],
            "job_description": ["300 employees across engineering."],
        }
    )

    result = _run_company_size_package(
        dataframe=dataframe,
        company_size_groups_path=company_size_groups_path,
        runtime_context=_runtime_context(
            company_names=["company_name"],
            company_descriptions=["company_description"],
            job_descriptions=["job_description"],
        ),
    )

    assert result.dataframe.loc[0, "company_size_group"] == "Medium"


def test_company_size_package_deduplicates_text_inputs(tmp_path: Path) -> None:
    company_size_groups_path = _write_company_size_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "company_name": ["Startup"],
            "company_description": ["Startup"],
            "job_description": ["Startup"],
        }
    )

    result = _run_company_size_package(
        dataframe=dataframe,
        company_size_groups_path=company_size_groups_path,
        runtime_context=_runtime_context(
            company_names=["company_name"],
            company_descriptions=["company_description"],
            job_descriptions=["job_description"],
        ),
    )

    assert result.dataframe.loc[0, "company_size_group"] == "Startup"
    assert result.report.warnings == []


def test_company_size_package_unknown_text_remains_quiet(tmp_path: Path) -> None:
    company_size_groups_path = _write_company_size_knowledge(tmp_path)
    dataframe = pd.DataFrame({"company_description": ["Good working environment"]})

    result = _run_company_size_package(
        dataframe=dataframe,
        company_size_groups_path=company_size_groups_path,
        runtime_context=_runtime_context(company_descriptions=["company_description"]),
    )

    assert result.dataframe.loc[0, "company_size_group"] is None
    assert result.report.warnings == []
    assert result.report.unknown_values_by_package == {}


def test_company_size_package_output_is_deterministic(tmp_path: Path) -> None:
    company_size_groups_path = _write_company_size_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {"company_description": ["Global multinational company with 80 employees"]}
    )
    kwargs = {
        "dataframe": dataframe,
        "company_size_groups_path": company_size_groups_path,
        "runtime_context": _runtime_context(company_descriptions=["company_description"]),
    }

    first = _run_company_size_package(**kwargs)
    second = _run_company_size_package(**kwargs)

    pd.testing.assert_frame_equal(first.dataframe, second.dataframe)
    assert first.report == second.report


def test_company_size_package_metadata_matches_phase_10_4j_contract() -> None:
    package = CompanySizePackage()
    metadata = package.metadata

    assert package.package_id == "company_size"
    assert package.name == "Company Size Package"
    assert package.description == "Generate company size attributes from company information."
    assert package.version == "1.0"
    assert package.enabled is True
    assert package.priority == 1100
    assert package.required_columns == ()
    assert package.produced_columns == ("company_size_group",)
    assert metadata.package_id == "company_size"
    assert metadata.produced_columns == ["company_size_group"]


def test_company_size_package_does_not_import_semantic_resolver() -> None:
    source = Path("core/knowledge_packages/company_size_package.py").read_text(
        encoding="utf-8"
    )

    assert "semantic_column_resolver" not in source
    assert "SemanticColumnResolver" not in source


def test_company_size_package_does_not_depend_on_prior_package_execution(
    tmp_path: Path,
) -> None:
    company_size_groups_path = _write_company_size_knowledge(tmp_path)
    dataframe = pd.DataFrame({"description": ["15 employees"]})

    result = _engine_with_company_size_package().apply_packages(
        dataframe,
        package_names=["company_size"],
        package_configs={
            "company_size": {
                "company_size_groups_file": str(company_size_groups_path)
            }
        },
        runtime_context=_runtime_context(job_descriptions=["description"]),
    )

    assert result.report.applied_packages == ["company_size"]
    assert result.dataframe.loc[0, "company_size_group"] == "Startup"


def test_current_package_order_includes_company_size() -> None:
    registry = PackageRegistry()
    registry.register(CompanySizePackage())
    registry.register(LanguagePackage())
    registry.register(EducationPackage())
    registry.register(EmploymentTypePackage())
    registry.register(ExperienceLevelPackage())
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
        "experience_level",
        "employment_type",
        "education",
        "language",
        "company_size",
    ]


def test_unrelated_package_can_be_added_without_company_size_or_engine_changes() -> None:
    dataframe = pd.DataFrame({"source": ["value"]})
    registry = PackageRegistry()
    registry.register(FakePackage())

    result = KnowledgePackageEngine(registry).apply_packages(
        dataframe,
        package_names=["fake"],
        package_configs={"fake": {}},
    )

    assert result.report.applied_packages == ["fake"]
    assert result.report.produced_columns_by_package == {"fake": ["fake_column"]}
    assert result.dataframe.loc[0, "fake_column"] == "fake"


def _engine_with_company_size_package() -> KnowledgePackageEngine:
    registry = PackageRegistry()
    registry.register(CompanySizePackage())
    return KnowledgePackageEngine(registry)


def _run_company_size_package(
    dataframe: pd.DataFrame,
    company_size_groups_path: Path,
    runtime_context: dict[str, dict[str, list[str]]],
) -> object:
    return _engine_with_company_size_package().apply_packages(
        dataframe,
        package_configs={
            "company_size": {
                "company_size_groups_file": str(company_size_groups_path)
            }
        },
        runtime_context=runtime_context,
    )


def _runtime_context(
    company_names: list[str] | None = None,
    company_descriptions: list[str] | None = None,
    job_descriptions: list[str] | None = None,
) -> dict[str, dict[str, list[str]]]:
    return {
        "semantic_columns": {
            "COMPANY_NAME": company_names or [],
            "COMPANY_DESCRIPTION": company_descriptions or [],
            "JOB_DESCRIPTION": job_descriptions or [],
        }
    }


def _write_company_size_knowledge(tmp_path: Path) -> Path:
    company_size_groups_path = tmp_path / "company_size_groups.json"
    company_size_groups_path.write_text(
        json.dumps(
            {
                "Enterprise": {
                    "aliases": [
                        "enterprise",
                        "fortune 500",
                        "global company",
                        "large corporation",
                        "multinational",
                    ],
                },
                "Large": {
                    "aliases": [
                        "large company",
                        "large-scale",
                        "national company",
                    ],
                },
                "Medium": {
                    "aliases": [
                        "medium company",
                        "mid-sized",
                        "scale-up",
                    ],
                },
                "Small": {
                    "aliases": [
                        "small business",
                        "small company",
                        "sme",
                    ],
                },
                "Startup": {
                    "aliases": [
                        "early-stage",
                        "seed-stage",
                        "startup",
                    ],
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return company_size_groups_path
