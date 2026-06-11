from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from core.knowledge_packages.base_package import BasePackage
from core.knowledge_packages.education_package import EducationPackage
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


def test_education_package_appends_education_level(tmp_path: Path) -> None:
    education_levels_path = _write_education_level_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "title": ["Backend Engineer"],
            "job_description": ["Bachelor degree required."],
        }
    )
    original = dataframe.copy(deep=True)

    result = _engine_with_education_package().apply_packages(
        dataframe,
        package_names=["education"],
        package_configs={
            "education": {"education_levels_file": str(education_levels_path)}
        },
        runtime_context=_runtime_context(
            titles=["title"],
            descriptions=["job_description"],
        ),
    )

    assert list(result.dataframe.columns) == [
        "title",
        "job_description",
        "education_level",
    ]
    assert result.dataframe.loc[0, "education_level"] == "Bachelor"
    assert result.report.applied_packages == ["education"]
    assert result.report.produced_columns_by_package == {
        "education": ["education_level"]
    }
    assert result.report.unknown_values_by_package == {}
    assert result.report.warnings == []
    pd.testing.assert_frame_equal(dataframe, original)


def test_education_package_does_not_overwrite_existing_column(tmp_path: Path) -> None:
    education_levels_path = _write_education_level_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "job_description": ["Master degree required."],
            "education_level": ["Existing Education"],
        }
    )

    result = _engine_with_education_package().apply_packages(
        dataframe,
        package_configs={
            "education": {"education_levels_file": str(education_levels_path)}
        },
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "education_level"] == "Existing Education"
    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["education"]
    assert any(
        "attempted to overwrite existing column: education_level" in warning
        for warning in result.report.warnings
    )


def test_education_package_skips_without_usable_semantic_columns() -> None:
    dataframe = pd.DataFrame({"job_description": ["Bachelor degree required."]})

    result = _engine_with_education_package().apply_packages(
        dataframe,
        package_configs={"education": {}},
        runtime_context={"semantic_columns": {}},
    )

    assert list(result.dataframe.columns) == ["job_description"]
    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["education"]
    assert result.report.warnings == [
        "Education package skipped; no usable semantic columns."
    ]


def test_education_package_runtime_context_none_skips_gracefully() -> None:
    dataframe = pd.DataFrame({"job_description": ["Bachelor degree required."]})

    result = _engine_with_education_package().apply_packages(
        dataframe,
        package_names=["education"],
        package_configs={"education": {}},
        runtime_context=None,
    )

    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["education"]
    assert result.report.warnings == [
        "Education package skipped; no usable semantic columns."
    ]


def test_education_package_matches_case_insensitive_aliases(tmp_path: Path) -> None:
    education_levels_path = _write_education_level_knowledge(tmp_path)
    dataframe = pd.DataFrame({"job_description": ["MASTER DEGREE required."]})

    result = _run_education_package(
        dataframe=dataframe,
        education_levels_path=education_levels_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "education_level"] == "Master"


def test_education_package_matches_english_aliases(tmp_path: Path) -> None:
    education_levels_path = _write_education_level_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "job_description": [
                "High school diploma accepted.",
                "Associate degree preferred.",
                "Bachelor degree required.",
                "AWS Professional Certificate required.",
                "MBA preferred.",
                "Master degree required.",
                "PhD preferred.",
            ]
        }
    )

    result = _run_education_package(
        dataframe=dataframe,
        education_levels_path=education_levels_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe["education_level"].tolist() == [
        "High School",
        "Associate",
        "Bachelor",
        "Professional Certificate",
        "MBA",
        "Master",
        "PhD",
    ]


def test_education_package_matches_vietnamese_aliases(tmp_path: Path) -> None:
    education_levels_path = _write_education_level_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "job_description": [
                "Tốt nghiệp đại học CNTT.",
                "Ưu tiên thạc sĩ.",
                "Yêu cầu tiến sĩ.",
                "Tốt nghiệp cao đẳng.",
            ]
        }
    )

    result = _run_education_package(
        dataframe=dataframe,
        education_levels_path=education_levels_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe["education_level"].tolist() == [
        "Bachelor",
        "Master",
        "PhD",
        "Associate",
    ]


def test_education_package_matches_all_semantic_input_types(tmp_path: Path) -> None:
    education_levels_path = _write_education_level_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "title": ["MBA Product Manager"],
            "description": ["Bachelor degree required."],
            "benefits": ["Professional certificate budget."],
            "requirements": ["Master degree preferred."],
            "education": ["PhD is a plus."],
        }
    )

    result = _run_education_package(
        dataframe=dataframe,
        education_levels_path=education_levels_path,
        runtime_context=_runtime_context(
            titles=["title"],
            descriptions=["description"],
            benefits=["benefits"],
            requirements=["requirements"],
            education=["education"],
        ),
    )

    assert result.dataframe.loc[0, "education_level"] == "PhD"


def test_education_package_uses_rank_for_multi_level_mentions(
    tmp_path: Path,
) -> None:
    education_levels_path = _write_education_level_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "job_description": [
                "Bachelor degree required. Master is a plus.",
                "Master or PhD preferred.",
            ]
        }
    )

    result = _run_education_package(
        dataframe=dataframe,
        education_levels_path=education_levels_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe["education_level"].tolist() == ["Master", "PhD"]


def test_education_package_uses_alphabetical_fallback_for_equal_score_and_rank(
    tmp_path: Path,
) -> None:
    education_levels_path = _write_education_level_knowledge(tmp_path)
    dataframe = pd.DataFrame({"job_description": ["Master or MBA preferred."]})

    result = _run_education_package(
        dataframe=dataframe,
        education_levels_path=education_levels_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "education_level"] == "MBA"


def test_education_package_deduplicates_text_inputs(tmp_path: Path) -> None:
    education_levels_path = _write_education_level_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "job_description": ["Bachelor degree required."],
            "job_requirements": ["Bachelor degree required."],
            "education": ["Bachelor degree required."],
        }
    )

    result = _run_education_package(
        dataframe=dataframe,
        education_levels_path=education_levels_path,
        runtime_context=_runtime_context(
            descriptions=["job_description"],
            requirements=["job_requirements"],
            education=["education"],
        ),
    )

    assert result.dataframe.loc[0, "education_level"] == "Bachelor"
    assert result.report.warnings == []


def test_education_package_unknown_text_remains_quiet(tmp_path: Path) -> None:
    education_levels_path = _write_education_level_knowledge(tmp_path)
    dataframe = pd.DataFrame({"job_description": ["Good communication skills."]})

    result = _run_education_package(
        dataframe=dataframe,
        education_levels_path=education_levels_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "education_level"] is None
    assert result.report.warnings == []
    assert result.report.unknown_values_by_package == {}


def test_education_package_output_is_deterministic(tmp_path: Path) -> None:
    education_levels_path = _write_education_level_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {"job_description": ["Bachelor degree required. Master is a plus."]}
    )
    kwargs = {
        "dataframe": dataframe,
        "education_levels_path": education_levels_path,
        "runtime_context": _runtime_context(descriptions=["job_description"]),
    }

    first = _run_education_package(**kwargs)
    second = _run_education_package(**kwargs)

    pd.testing.assert_frame_equal(first.dataframe, second.dataframe)
    assert first.report == second.report


def test_education_package_metadata_matches_phase_10_4h_contract() -> None:
    package = EducationPackage()
    metadata = package.metadata

    assert package.package_id == "education"
    assert package.name == "Education Package"
    assert package.description == "Generate education attributes from job text columns."
    assert package.version == "1.0"
    assert package.enabled is True
    assert package.priority == 900
    assert package.required_columns == ()
    assert package.produced_columns == ("education_level",)
    assert metadata.package_id == "education"
    assert metadata.produced_columns == ["education_level"]


def test_education_package_does_not_import_semantic_resolver() -> None:
    source = Path("core/knowledge_packages/education_package.py").read_text(
        encoding="utf-8"
    )

    assert "semantic_column_resolver" not in source
    assert "SemanticColumnResolver" not in source


def test_education_package_does_not_depend_on_prior_package_execution(
    tmp_path: Path,
) -> None:
    education_levels_path = _write_education_level_knowledge(tmp_path)
    dataframe = pd.DataFrame({"description": ["Bachelor degree required."]})

    result = _engine_with_education_package().apply_packages(
        dataframe,
        package_names=["education"],
        package_configs={
            "education": {"education_levels_file": str(education_levels_path)}
        },
        runtime_context=_runtime_context(descriptions=["description"]),
    )

    assert result.report.applied_packages == ["education"]
    assert result.dataframe.loc[0, "education_level"] == "Bachelor"


def test_current_package_order_includes_education() -> None:
    registry = PackageRegistry()
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
    ]


def test_unrelated_package_can_be_added_without_education_or_engine_changes() -> None:
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


def _engine_with_education_package() -> KnowledgePackageEngine:
    registry = PackageRegistry()
    registry.register(EducationPackage())
    return KnowledgePackageEngine(registry)


def _run_education_package(
    dataframe: pd.DataFrame,
    education_levels_path: Path,
    runtime_context: dict[str, dict[str, list[str]]],
) -> object:
    return _engine_with_education_package().apply_packages(
        dataframe,
        package_configs={
            "education": {"education_levels_file": str(education_levels_path)}
        },
        runtime_context=runtime_context,
    )


def _runtime_context(
    titles: list[str] | None = None,
    descriptions: list[str] | None = None,
    benefits: list[str] | None = None,
    requirements: list[str] | None = None,
    education: list[str] | None = None,
) -> dict[str, dict[str, list[str]]]:
    return {
        "semantic_columns": {
            "JOB_TITLE": titles or [],
            "JOB_DESCRIPTION": descriptions or [],
            "BENEFITS": benefits or [],
            "JOB_REQUIREMENTS": requirements or [],
            "EDUCATION": education or [],
        }
    }


def _write_education_level_knowledge(tmp_path: Path) -> Path:
    education_levels_path = tmp_path / "education_levels.json"
    education_levels_path.write_text(
        json.dumps(
            {
                "Associate": {
                    "aliases": [
                        "associate",
                        "associate degree",
                        "cao đẳng",
                    ],
                    "rank": 2,
                },
                "Bachelor": {
                    "aliases": [
                        "bachelor",
                        "bachelor degree",
                        "cử nhân",
                        "đại học",
                    ],
                    "rank": 3,
                },
                "High School": {
                    "aliases": [
                        "high school",
                        "secondary school",
                        "trung học phổ thông",
                    ],
                    "rank": 1,
                },
                "MBA": {
                    "aliases": [
                        "mba",
                    ],
                    "rank": 4,
                },
                "Master": {
                    "aliases": [
                        "master",
                        "master degree",
                        "thạc sĩ",
                    ],
                    "rank": 4,
                },
                "PhD": {
                    "aliases": [
                        "doctorate",
                        "phd",
                        "tiến sĩ",
                    ],
                    "rank": 5,
                },
                "Professional Certificate": {
                    "aliases": [
                        "aws certification",
                        "google professional certificate",
                        "professional certificate",
                    ],
                    "rank": 3,
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return education_levels_path
