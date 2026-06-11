from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from core.knowledge_packages.base_package import BasePackage
from core.knowledge_packages.employment_type_package import EmploymentTypePackage
from core.knowledge_packages.experience_level_package import ExperienceLevelPackage
from core.knowledge_packages.industry_domain_package import IndustryDomainPackage
from core.knowledge_packages.job_title_package import JobTitlePackage
from core.knowledge_packages.knowledge_package_engine import KnowledgePackageEngine
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


def test_employment_type_package_appends_employment_type(
    tmp_path: Path,
) -> None:
    employment_types_path = _write_employment_type_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "title": ["Backend Engineer"],
            "job_description": ["Full-time permanent position."],
        }
    )
    original = dataframe.copy(deep=True)

    result = _engine_with_employment_type_package().apply_packages(
        dataframe,
        package_names=["employment_type"],
        package_configs={
            "employment_type": {
                "employment_types_file": str(employment_types_path)
            }
        },
        runtime_context=_runtime_context(
            titles=["title"],
            descriptions=["job_description"],
        ),
    )

    assert list(result.dataframe.columns) == [
        "title",
        "job_description",
        "employment_type",
    ]
    assert result.dataframe.loc[0, "employment_type"] == "Full-time"
    assert result.report.applied_packages == ["employment_type"]
    assert result.report.produced_columns_by_package == {
        "employment_type": ["employment_type"]
    }
    assert result.report.unknown_values_by_package == {}
    assert result.report.warnings == []
    pd.testing.assert_frame_equal(dataframe, original)


def test_employment_type_package_does_not_overwrite_existing_column(
    tmp_path: Path,
) -> None:
    employment_types_path = _write_employment_type_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "job_description": ["Full-time permanent position."],
            "employment_type": ["Existing Type"],
        }
    )

    result = _engine_with_employment_type_package().apply_packages(
        dataframe,
        package_configs={
            "employment_type": {
                "employment_types_file": str(employment_types_path)
            }
        },
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "employment_type"] == "Existing Type"
    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["employment_type"]
    assert any(
        "attempted to overwrite existing column: employment_type" in warning
        for warning in result.report.warnings
    )


def test_employment_type_package_skips_without_usable_semantic_columns() -> None:
    dataframe = pd.DataFrame({"job_description": ["Full-time permanent position."]})

    result = _engine_with_employment_type_package().apply_packages(
        dataframe,
        package_configs={"employment_type": {}},
        runtime_context={"semantic_columns": {}},
    )

    assert list(result.dataframe.columns) == ["job_description"]
    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["employment_type"]
    assert result.report.warnings == [
        "Employment type package skipped; no usable semantic columns."
    ]


def test_employment_type_package_runtime_context_none_skips_gracefully() -> None:
    dataframe = pd.DataFrame({"job_description": ["Full-time permanent position."]})

    result = _engine_with_employment_type_package().apply_packages(
        dataframe,
        package_names=["employment_type"],
        package_configs={"employment_type": {}},
        runtime_context=None,
    )

    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["employment_type"]
    assert result.report.warnings == [
        "Employment type package skipped; no usable semantic columns."
    ]


def test_employment_type_package_matches_case_insensitive_aliases(
    tmp_path: Path,
) -> None:
    employment_types_path = _write_employment_type_knowledge(tmp_path)
    dataframe = pd.DataFrame({"job_description": ["FULL-TIME PERMANENT position."]})

    result = _run_employment_type_package(
        dataframe=dataframe,
        employment_types_path=employment_types_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "employment_type"] == "Full-time"


def test_employment_type_package_matches_english_aliases(tmp_path: Path) -> None:
    employment_types_path = _write_employment_type_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "job_description": [
                "Freelance React Developer.",
                "6-month contract with extension.",
                "Internship opportunity for AI students.",
                "Part-time 20 hours/week.",
                "Seasonal warehouse staff.",
                "Temporary project role.",
            ]
        }
    )

    result = _run_employment_type_package(
        dataframe=dataframe,
        employment_types_path=employment_types_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe["employment_type"].tolist() == [
        "Freelance",
        "Contract",
        "Internship",
        "Part-time",
        "Seasonal",
        "Temporary",
    ]


def test_employment_type_package_matches_vietnamese_aliases(tmp_path: Path) -> None:
    employment_types_path = _write_employment_type_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "job_description": [
                "Thực tập AI Engineer.",
                "Làm việc toàn thời gian.",
                "Vị trí bán thời gian.",
                "Hợp đồng 12 tháng.",
            ]
        }
    )

    result = _run_employment_type_package(
        dataframe=dataframe,
        employment_types_path=employment_types_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe["employment_type"].tolist() == [
        "Internship",
        "Full-time",
        "Part-time",
        "Contract",
    ]


def test_employment_type_package_matches_title(tmp_path: Path) -> None:
    employment_types_path = _write_employment_type_knowledge(tmp_path)
    dataframe = pd.DataFrame({"title": ["Freelance React Developer"]})

    result = _run_employment_type_package(
        dataframe=dataframe,
        employment_types_path=employment_types_path,
        runtime_context=_runtime_context(titles=["title"]),
    )

    assert result.dataframe.loc[0, "employment_type"] == "Freelance"


def test_employment_type_package_matches_description(tmp_path: Path) -> None:
    employment_types_path = _write_employment_type_knowledge(tmp_path)
    dataframe = pd.DataFrame({"description": ["Full-time permanent position."]})

    result = _run_employment_type_package(
        dataframe=dataframe,
        employment_types_path=employment_types_path,
        runtime_context=_runtime_context(descriptions=["description"]),
    )

    assert result.dataframe.loc[0, "employment_type"] == "Full-time"


def test_employment_type_package_matches_benefits(tmp_path: Path) -> None:
    employment_types_path = _write_employment_type_knowledge(tmp_path)
    dataframe = pd.DataFrame({"benefits": ["Permanent employee insurance."]})

    result = _run_employment_type_package(
        dataframe=dataframe,
        employment_types_path=employment_types_path,
        runtime_context=_runtime_context(benefits=["benefits"]),
    )

    assert result.dataframe.loc[0, "employment_type"] == "Full-time"


def test_employment_type_package_matches_working_time(tmp_path: Path) -> None:
    employment_types_path = _write_employment_type_knowledge(tmp_path)
    dataframe = pd.DataFrame({"working_time": ["Part time, 20 hours/week."]})

    result = _run_employment_type_package(
        dataframe=dataframe,
        employment_types_path=employment_types_path,
        runtime_context=_runtime_context(working_times=["working_time"]),
    )

    assert result.dataframe.loc[0, "employment_type"] == "Part-time"


def test_employment_type_package_matches_explicit_employment_type_column(
    tmp_path: Path,
) -> None:
    employment_types_path = _write_employment_type_knowledge(tmp_path)
    dataframe = pd.DataFrame({"job_type": ["Contract"]})

    result = _run_employment_type_package(
        dataframe=dataframe,
        employment_types_path=employment_types_path,
        runtime_context=_runtime_context(employment_types=["job_type"]),
    )

    assert result.dataframe.loc[0, "employment_type"] == "Contract"


def test_employment_type_package_uses_multi_hit_voting(tmp_path: Path) -> None:
    employment_types_path = _write_employment_type_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {"job_description": ["Full-time permanent role with contract review."]}
    )

    result = _run_employment_type_package(
        dataframe=dataframe,
        employment_types_path=employment_types_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "employment_type"] == "Full-time"


def test_employment_type_package_resolves_ties_alphabetically(
    tmp_path: Path,
) -> None:
    employment_types_path = _write_employment_type_knowledge(tmp_path)
    dataframe = pd.DataFrame({"job_description": ["Intern part-time role."]})

    result = _run_employment_type_package(
        dataframe=dataframe,
        employment_types_path=employment_types_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "employment_type"] == "Internship"


def test_employment_type_package_deduplicates_text_inputs(tmp_path: Path) -> None:
    employment_types_path = _write_employment_type_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "title": ["Freelance React Developer"],
            "job_description": ["Freelance React Developer"],
            "benefits": ["Freelance React Developer"],
        }
    )

    result = _run_employment_type_package(
        dataframe=dataframe,
        employment_types_path=employment_types_path,
        runtime_context=_runtime_context(
            titles=["title"],
            descriptions=["job_description"],
            benefits=["benefits"],
        ),
    )

    assert result.dataframe.loc[0, "employment_type"] == "Freelance"
    assert result.report.warnings == []


def test_employment_type_package_unknown_text_remains_quiet(
    tmp_path: Path,
) -> None:
    employment_types_path = _write_employment_type_knowledge(tmp_path)
    dataframe = pd.DataFrame({"job_description": ["No employment type signal."]})

    result = _run_employment_type_package(
        dataframe=dataframe,
        employment_types_path=employment_types_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "employment_type"] is None
    assert result.report.warnings == []
    assert result.report.unknown_values_by_package == {}


def test_employment_type_package_output_is_deterministic(tmp_path: Path) -> None:
    employment_types_path = _write_employment_type_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {"job_description": ["Full-time permanent position."]}
    )
    kwargs = {
        "dataframe": dataframe,
        "employment_types_path": employment_types_path,
        "runtime_context": _runtime_context(descriptions=["job_description"]),
    }

    first = _run_employment_type_package(**kwargs)
    second = _run_employment_type_package(**kwargs)

    pd.testing.assert_frame_equal(first.dataframe, second.dataframe)
    assert first.report == second.report


def test_employment_type_package_metadata_matches_phase_10_4g_contract() -> None:
    package = EmploymentTypePackage()
    metadata = package.metadata

    assert package.package_id == "employment_type"
    assert package.name == "Employment Type Package"
    assert package.description == "Generate employment type attributes from job text columns."
    assert package.version == "1.0"
    assert package.enabled is True
    assert package.priority == 800
    assert package.required_columns == ()
    assert package.produced_columns == ("employment_type",)
    assert metadata.package_id == "employment_type"
    assert metadata.produced_columns == ["employment_type"]


def test_employment_type_package_does_not_import_semantic_resolver() -> None:
    source = Path("core/knowledge_packages/employment_type_package.py").read_text(
        encoding="utf-8"
    )

    assert "semantic_column_resolver" not in source
    assert "SemanticColumnResolver" not in source


def test_employment_type_package_does_not_depend_on_prior_package_execution(
    tmp_path: Path,
) -> None:
    employment_types_path = _write_employment_type_knowledge(tmp_path)
    dataframe = pd.DataFrame({"description": ["Full-time permanent position."]})

    result = _engine_with_employment_type_package().apply_packages(
        dataframe,
        package_names=["employment_type"],
        package_configs={
            "employment_type": {
                "employment_types_file": str(employment_types_path)
            }
        },
        runtime_context=_runtime_context(descriptions=["description"]),
    )

    assert result.report.applied_packages == ["employment_type"]
    assert result.dataframe.loc[0, "employment_type"] == "Full-time"


def test_current_package_order_includes_employment_type() -> None:
    registry = PackageRegistry()
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
    ]


def test_unrelated_package_can_be_added_without_employment_or_engine_changes() -> None:
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


def _engine_with_employment_type_package() -> KnowledgePackageEngine:
    registry = PackageRegistry()
    registry.register(EmploymentTypePackage())
    return KnowledgePackageEngine(registry)


def _run_employment_type_package(
    dataframe: pd.DataFrame,
    employment_types_path: Path,
    runtime_context: dict[str, dict[str, list[str]]],
) -> object:
    return _engine_with_employment_type_package().apply_packages(
        dataframe,
        package_configs={
            "employment_type": {
                "employment_types_file": str(employment_types_path)
            }
        },
        runtime_context=runtime_context,
    )


def _runtime_context(
    titles: list[str] | None = None,
    descriptions: list[str] | None = None,
    benefits: list[str] | None = None,
    working_times: list[str] | None = None,
    employment_types: list[str] | None = None,
) -> dict[str, dict[str, list[str]]]:
    return {
        "semantic_columns": {
            "JOB_TITLE": titles or [],
            "JOB_DESCRIPTION": descriptions or [],
            "BENEFITS": benefits or [],
            "WORKING_TIME": working_times or [],
            "EMPLOYMENT_TYPE": employment_types or [],
        }
    }


def _write_employment_type_knowledge(tmp_path: Path) -> Path:
    employment_types_path = tmp_path / "employment_types.json"
    employment_types_path.write_text(
        json.dumps(
            {
                "Contract": {
                    "aliases": [
                        "12-month contract",
                        "6-month contract",
                        "contract",
                        "hợp đồng",
                    ],
                },
                "Freelance": {
                    "aliases": [
                        "contractor",
                        "freelance",
                    ],
                },
                "Full-time": {
                    "aliases": [
                        "full time",
                        "full-time",
                        "permanent",
                        "toàn thời gian",
                    ],
                },
                "Internship": {
                    "aliases": [
                        "intern",
                        "internship",
                        "thực tập",
                    ],
                },
                "Part-time": {
                    "aliases": [
                        "bán thời gian",
                        "part time",
                        "part-time",
                    ],
                },
                "Seasonal": {
                    "aliases": [
                        "seasonal",
                    ],
                },
                "Temporary": {
                    "aliases": [
                        "temp",
                        "temporary",
                    ],
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return employment_types_path
