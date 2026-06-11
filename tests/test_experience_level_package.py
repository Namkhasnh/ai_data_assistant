from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from core.knowledge_packages.base_package import BasePackage
from core.knowledge_packages.experience_level_package import ExperienceLevelPackage
from core.knowledge_packages.industry_domain_package import IndustryDomainPackage
from core.knowledge_packages.job_title_package import JobTitlePackage
from core.knowledge_packages.knowledge_package_engine import KnowledgePackageEngine
from core.knowledge_packages.location_package import LocationPackage
from core.knowledge_packages.package_registry import PackageRegistry
from core.knowledge_packages.salary_package import SalaryPackage
from core.knowledge_packages.skill_package import SkillPackage
from core.knowledge_packages.work_mode_package import WorkModePackage


class HealthcareTestPackage(BasePackage):
    package_id = "healthcare_test"
    name = "Healthcare Test Package"
    priority = 50
    required_columns: tuple[str, ...] = ()
    produced_columns = ("test_name",)

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        output = dataframe.copy(deep=True)
        output["test_name"] = "Complete Blood Count"
        return output


def test_experience_level_package_appends_experience_columns(
    tmp_path: Path,
) -> None:
    experience_levels_path = _write_experience_level_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "title": ["Backend Engineer"],
            "job_description": ["2 years experience with APIs."],
        }
    )
    original = dataframe.copy(deep=True)

    result = _engine_with_experience_level_package().apply_packages(
        dataframe,
        package_names=["experience_level"],
        package_configs={
            "experience_level": {
                "experience_levels_file": str(experience_levels_path)
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
        "experience_years",
        "experience_level",
    ]
    assert result.dataframe.loc[0, "experience_years"] == 2
    assert result.dataframe.loc[0, "experience_level"] == "Middle"
    assert result.report.applied_packages == ["experience_level"]
    assert result.report.produced_columns_by_package == {
        "experience_level": ["experience_years", "experience_level"]
    }
    assert result.report.unknown_values_by_package == {}
    assert result.report.warnings == []
    pd.testing.assert_frame_equal(dataframe, original)


def test_experience_level_package_does_not_overwrite_existing_columns(
    tmp_path: Path,
) -> None:
    experience_levels_path = _write_experience_level_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "title": ["Senior Backend Engineer"],
            "experience_years": [99],
            "experience_level": ["Existing Level"],
        }
    )

    result = _engine_with_experience_level_package().apply_packages(
        dataframe,
        package_configs={
            "experience_level": {
                "experience_levels_file": str(experience_levels_path)
            }
        },
        runtime_context=_runtime_context(titles=["title"]),
    )

    assert result.dataframe.loc[0, "experience_years"] == 99
    assert result.dataframe.loc[0, "experience_level"] == "Existing Level"
    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["experience_level"]
    assert any(
        "attempted to overwrite existing column: experience_years" in warning
        for warning in result.report.warnings
    )
    assert any(
        "attempted to overwrite existing column: experience_level" in warning
        for warning in result.report.warnings
    )


def test_experience_level_package_skips_without_usable_semantic_columns() -> None:
    dataframe = pd.DataFrame({"title": ["Senior Backend Engineer"]})

    result = _engine_with_experience_level_package().apply_packages(
        dataframe,
        package_configs={"experience_level": {}},
        runtime_context={"semantic_columns": {}},
    )

    assert list(result.dataframe.columns) == ["title"]
    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["experience_level"]
    assert result.report.warnings == [
        "Experience level package skipped; no usable semantic columns."
    ]


def test_experience_level_package_runtime_context_none_skips_gracefully() -> None:
    dataframe = pd.DataFrame({"title": ["Senior Backend Engineer"]})

    result = _engine_with_experience_level_package().apply_packages(
        dataframe,
        package_names=["experience_level"],
        package_configs={"experience_level": {}},
        runtime_context=None,
    )

    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["experience_level"]
    assert result.report.warnings == [
        "Experience level package skipped; no usable semantic columns."
    ]


def test_experience_level_package_matches_case_insensitive_aliases(
    tmp_path: Path,
) -> None:
    experience_levels_path = _write_experience_level_knowledge(tmp_path)
    dataframe = pd.DataFrame({"title": ["SENIOR Backend Engineer"]})

    result = _run_experience_level_package(
        dataframe=dataframe,
        experience_levels_path=experience_levels_path,
        runtime_context=_runtime_context(titles=["title"]),
    )

    assert pd.isna(result.dataframe.loc[0, "experience_years"])
    assert result.dataframe.loc[0, "experience_level"] == "Senior"


def test_experience_level_package_matches_english_aliases(tmp_path: Path) -> None:
    experience_levels_path = _write_experience_level_knowledge(tmp_path)
    dataframe = pd.DataFrame({"job_description": ["Fresh graduate welcome."]})

    result = _run_experience_level_package(
        dataframe=dataframe,
        experience_levels_path=experience_levels_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "experience_years"] == 0
    assert result.dataframe.loc[0, "experience_level"] == "Junior"


def test_experience_level_package_matches_vietnamese_aliases(tmp_path: Path) -> None:
    experience_levels_path = _write_experience_level_knowledge(tmp_path)
    dataframe = pd.DataFrame({"title": ["Thực tập AI Engineer"]})

    result = _run_experience_level_package(
        dataframe=dataframe,
        experience_levels_path=experience_levels_path,
        runtime_context=_runtime_context(titles=["title"]),
    )

    assert result.dataframe.loc[0, "experience_years"] == 0
    assert result.dataframe.loc[0, "experience_level"] == "Intern"


def test_experience_level_package_extracts_years_with_plus_sign(
    tmp_path: Path,
) -> None:
    experience_levels_path = _write_experience_level_knowledge(tmp_path)
    dataframe = pd.DataFrame({"job_description": ["5+ years experience."]})

    result = _run_experience_level_package(
        dataframe=dataframe,
        experience_levels_path=experience_levels_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "experience_years"] == 5
    assert result.dataframe.loc[0, "experience_level"] == "Senior"


def test_experience_level_package_extracts_at_least_years(tmp_path: Path) -> None:
    experience_levels_path = _write_experience_level_knowledge(tmp_path)
    dataframe = pd.DataFrame({"job_description": ["At least 8 years with Python."]})

    result = _run_experience_level_package(
        dataframe=dataframe,
        experience_levels_path=experience_levels_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "experience_years"] == 8
    assert result.dataframe.loc[0, "experience_level"] == "Lead"


def test_experience_level_package_uses_largest_year_mention(
    tmp_path: Path,
) -> None:
    experience_levels_path = _write_experience_level_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {"job_description": ["3 years Java. 8 years Python."]}
    )

    result = _run_experience_level_package(
        dataframe=dataframe,
        experience_levels_path=experience_levels_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "experience_years"] == 8
    assert result.dataframe.loc[0, "experience_level"] == "Lead"


def test_experience_level_package_detects_title_only_levels(
    tmp_path: Path,
) -> None:
    experience_levels_path = _write_experience_level_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "title": [
                "Principal Software Architect",
                "Lead Data Scientist",
                "Intern AI Engineer",
            ]
        }
    )

    result = _run_experience_level_package(
        dataframe=dataframe,
        experience_levels_path=experience_levels_path,
        runtime_context=_runtime_context(titles=["title"]),
    )

    assert pd.isna(result.dataframe.loc[0, "experience_years"])
    assert result.dataframe.loc[0, "experience_level"] == "Principal"
    assert pd.isna(result.dataframe.loc[1, "experience_years"])
    assert result.dataframe.loc[1, "experience_level"] == "Lead"
    assert result.dataframe.loc[2, "experience_years"] == 0
    assert result.dataframe.loc[2, "experience_level"] == "Intern"


def test_experience_level_package_numeric_signal_overrides_incompatible_alias(
    tmp_path: Path,
) -> None:
    experience_levels_path = _write_experience_level_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "title": ["Senior Data Engineer"],
            "job_description": ["2 years experience."],
        }
    )

    result = _run_experience_level_package(
        dataframe=dataframe,
        experience_levels_path=experience_levels_path,
        runtime_context=_runtime_context(
            titles=["title"],
            descriptions=["job_description"],
        ),
    )

    assert result.dataframe.loc[0, "experience_years"] == 2
    assert result.dataframe.loc[0, "experience_level"] == "Middle"


def test_experience_level_package_accepts_compatible_title_with_numeric_years(
    tmp_path: Path,
) -> None:
    experience_levels_path = _write_experience_level_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "title": ["Lead Data Scientist"],
            "job_description": ["10 years experience."],
        }
    )

    result = _run_experience_level_package(
        dataframe=dataframe,
        experience_levels_path=experience_levels_path,
        runtime_context=_runtime_context(
            titles=["title"],
            descriptions=["job_description"],
        ),
    )

    assert result.dataframe.loc[0, "experience_years"] == 10
    assert result.dataframe.loc[0, "experience_level"] == "Lead"


def test_experience_level_package_deduplicates_text_inputs(
    tmp_path: Path,
) -> None:
    experience_levels_path = _write_experience_level_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "title": ["Senior Backend Engineer"],
            "job_description": ["Senior Backend Engineer"],
            "benefits": ["Senior Backend Engineer"],
        }
    )

    result = _run_experience_level_package(
        dataframe=dataframe,
        experience_levels_path=experience_levels_path,
        runtime_context=_runtime_context(
            titles=["title"],
            descriptions=["job_description"],
            benefits=["benefits"],
        ),
    )

    assert pd.isna(result.dataframe.loc[0, "experience_years"])
    assert result.dataframe.loc[0, "experience_level"] == "Senior"
    assert result.report.warnings == []


def test_experience_level_package_unknown_text_remains_quiet(
    tmp_path: Path,
) -> None:
    experience_levels_path = _write_experience_level_knowledge(tmp_path)
    dataframe = pd.DataFrame({"job_description": ["No explicit experience signal."]})

    result = _run_experience_level_package(
        dataframe=dataframe,
        experience_levels_path=experience_levels_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert pd.isna(result.dataframe.loc[0, "experience_years"])
    assert result.dataframe.loc[0, "experience_level"] is None
    assert result.report.warnings == []
    assert result.report.unknown_values_by_package == {}


def test_experience_level_package_output_is_deterministic(tmp_path: Path) -> None:
    experience_levels_path = _write_experience_level_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "title": ["Senior Data Engineer"],
            "job_description": ["2 years experience."],
        }
    )
    kwargs = {
        "dataframe": dataframe,
        "experience_levels_path": experience_levels_path,
        "runtime_context": _runtime_context(
            titles=["title"],
            descriptions=["job_description"],
        ),
    }

    first = _run_experience_level_package(**kwargs)
    second = _run_experience_level_package(**kwargs)

    pd.testing.assert_frame_equal(first.dataframe, second.dataframe)
    assert first.report == second.report


def test_experience_level_package_metadata_matches_phase_10_4f_contract() -> None:
    package = ExperienceLevelPackage()
    metadata = package.metadata

    assert package.package_id == "experience_level"
    assert package.name == "Experience Level Package"
    assert package.description == "Generate experience attributes from job text columns."
    assert package.version == "1.0"
    assert package.enabled is True
    assert package.priority == 700
    assert package.required_columns == ()
    assert package.produced_columns == ("experience_years", "experience_level")
    assert metadata.package_id == "experience_level"
    assert metadata.produced_columns == ["experience_years", "experience_level"]


def test_experience_level_package_does_not_import_semantic_resolver() -> None:
    source = Path("core/knowledge_packages/experience_level_package.py").read_text(
        encoding="utf-8"
    )

    assert "semantic_column_resolver" not in source
    assert "SemanticColumnResolver" not in source


def test_current_package_order_includes_experience_level() -> None:
    registry = PackageRegistry()
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
    ]


def test_unrelated_package_can_be_added_without_experience_or_engine_changes() -> None:
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


def _engine_with_experience_level_package() -> KnowledgePackageEngine:
    registry = PackageRegistry()
    registry.register(ExperienceLevelPackage())
    return KnowledgePackageEngine(registry)


def _run_experience_level_package(
    dataframe: pd.DataFrame,
    experience_levels_path: Path,
    runtime_context: dict[str, dict[str, list[str]]],
) -> object:
    return _engine_with_experience_level_package().apply_packages(
        dataframe,
        package_configs={
            "experience_level": {
                "experience_levels_file": str(experience_levels_path)
            }
        },
        runtime_context=runtime_context,
    )


def _runtime_context(
    titles: list[str] | None = None,
    descriptions: list[str] | None = None,
    benefits: list[str] | None = None,
) -> dict[str, dict[str, list[str]]]:
    return {
        "semantic_columns": {
            "JOB_TITLE": titles or [],
            "JOB_DESCRIPTION": descriptions or [],
            "BENEFITS": benefits or [],
        }
    }


def _write_experience_level_knowledge(tmp_path: Path) -> Path:
    experience_levels_path = tmp_path / "experience_levels.json"
    experience_levels_path.write_text(
        json.dumps(
            {
                "Intern": {
                    "aliases": [
                        "intern",
                        "internship",
                        "thực tập",
                    ],
                    "min_years": 0,
                },
                "Junior": {
                    "aliases": [
                        "fresh graduate",
                        "junior",
                    ],
                    "min_years": 0,
                },
                "Middle": {
                    "aliases": [
                        "mid-level",
                        "middle",
                    ],
                    "min_years": 2,
                },
                "Senior": {
                    "aliases": [
                        "senior",
                    ],
                    "min_years": 5,
                },
                "Lead": {
                    "aliases": [
                        "lead",
                    ],
                    "min_years": 8,
                },
                "Principal": {
                    "aliases": [
                        "principal",
                    ],
                    "min_years": 10,
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return experience_levels_path
