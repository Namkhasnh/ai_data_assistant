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


def test_language_package_appends_languages_list(tmp_path: Path) -> None:
    languages_path = _write_language_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "title": ["Backend Engineer"],
            "job_description": ["English communication skill. TOEIC 800."],
        }
    )
    original = dataframe.copy(deep=True)

    result = _engine_with_language_package().apply_packages(
        dataframe,
        package_names=["language"],
        package_configs={"language": {"languages_file": str(languages_path)}},
        runtime_context=_runtime_context(
            titles=["title"],
            descriptions=["job_description"],
        ),
    )

    assert list(result.dataframe.columns) == ["title", "job_description", "languages"]
    assert result.dataframe.loc[0, "languages"] == ["English"]
    assert isinstance(result.dataframe.loc[0, "languages"], list)
    assert result.report.applied_packages == ["language"]
    assert result.report.produced_columns_by_package == {"language": ["languages"]}
    assert result.report.unknown_values_by_package == {}
    assert result.report.warnings == []
    pd.testing.assert_frame_equal(dataframe, original)


def test_language_package_does_not_overwrite_existing_column(tmp_path: Path) -> None:
    languages_path = _write_language_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "job_description": ["English communication skill."],
            "languages": [["Existing Language"]],
        }
    )

    result = _engine_with_language_package().apply_packages(
        dataframe,
        package_configs={"language": {"languages_file": str(languages_path)}},
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "languages"] == ["Existing Language"]
    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["language"]
    assert any(
        "attempted to overwrite existing column: languages" in warning
        for warning in result.report.warnings
    )


def test_language_package_skips_without_usable_semantic_columns() -> None:
    dataframe = pd.DataFrame({"job_description": ["English communication skill."]})

    result = _engine_with_language_package().apply_packages(
        dataframe,
        package_configs={"language": {}},
        runtime_context={"semantic_columns": {}},
    )

    assert list(result.dataframe.columns) == ["job_description"]
    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["language"]
    assert result.report.warnings == [
        "Language package skipped; no usable semantic columns."
    ]


def test_language_package_runtime_context_none_skips_gracefully() -> None:
    dataframe = pd.DataFrame({"job_description": ["English communication skill."]})

    result = _engine_with_language_package().apply_packages(
        dataframe,
        package_names=["language"],
        package_configs={"language": {}},
        runtime_context=None,
    )

    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["language"]
    assert result.report.warnings == [
        "Language package skipped; no usable semantic columns."
    ]


def test_language_package_matches_case_insensitive_aliases(tmp_path: Path) -> None:
    languages_path = _write_language_knowledge(tmp_path)
    dataframe = pd.DataFrame({"job_description": ["ENGLISH and JAPANESE required."]})

    result = _run_language_package(
        dataframe=dataframe,
        languages_path=languages_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "languages"] == ["English", "Japanese"]


def test_language_package_matches_english_and_japanese_aliases(
    tmp_path: Path,
) -> None:
    languages_path = _write_language_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "job_description": [
                "English communication skill. TOEIC 800. Japanese N2 preferred."
            ]
        }
    )

    result = _run_language_package(
        dataframe=dataframe,
        languages_path=languages_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "languages"] == ["English", "Japanese"]


def test_language_package_removes_duplicate_language_matches(tmp_path: Path) -> None:
    languages_path = _write_language_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "job_description": [
                "English communication. IELTS 6.5. TOEIC 800. C1 English."
            ]
        }
    )

    result = _run_language_package(
        dataframe=dataframe,
        languages_path=languages_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "languages"] == ["English"]


def test_language_package_matches_chinese_and_korean_aliases(tmp_path: Path) -> None:
    languages_path = _write_language_knowledge(tmp_path)
    dataframe = pd.DataFrame({"job_description": ["Mandarin HSK4 and Korean TOPIK 5."]})

    result = _run_language_package(
        dataframe=dataframe,
        languages_path=languages_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "languages"] == ["Chinese", "Korean"]


def test_language_package_matches_vietnamese_aliases(tmp_path: Path) -> None:
    languages_path = _write_language_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {"job_description": ["Tiếng Anh giao tiếp. TOEIC 800. Tiếng Nhật N2."]}
    )

    result = _run_language_package(
        dataframe=dataframe,
        languages_path=languages_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "languages"] == ["English", "Japanese"]


def test_language_package_returns_empty_list_for_unknown_text(tmp_path: Path) -> None:
    languages_path = _write_language_knowledge(tmp_path)
    dataframe = pd.DataFrame({"job_description": ["Good communication skills."]})

    result = _run_language_package(
        dataframe=dataframe,
        languages_path=languages_path,
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "languages"] == []
    assert result.report.warnings == []
    assert result.report.unknown_values_by_package == {}


def test_language_package_reads_all_semantic_input_types(tmp_path: Path) -> None:
    languages_path = _write_language_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "title": ["Japanese Bridge Engineer"],
            "description": ["English communication skill."],
            "benefits": ["French language club."],
            "requirements": ["German documents and Korean TOPIK."],
            "education": ["Chinese HSK5 accepted."],
        }
    )

    result = _run_language_package(
        dataframe=dataframe,
        languages_path=languages_path,
        runtime_context=_runtime_context(
            titles=["title"],
            descriptions=["description"],
            benefits=["benefits"],
            requirements=["requirements"],
            education=["education"],
        ),
    )

    assert result.dataframe.loc[0, "languages"] == [
        "Chinese",
        "English",
        "French",
        "German",
        "Japanese",
        "Korean",
    ]


def test_language_package_deduplicates_text_inputs(tmp_path: Path) -> None:
    languages_path = _write_language_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "job_description": ["English communication skill."],
            "requirements": ["English communication skill."],
            "education": ["English communication skill."],
        }
    )

    result = _run_language_package(
        dataframe=dataframe,
        languages_path=languages_path,
        runtime_context=_runtime_context(
            descriptions=["job_description"],
            requirements=["requirements"],
            education=["education"],
        ),
    )

    assert result.dataframe.loc[0, "languages"] == ["English"]
    assert result.report.warnings == []


def test_language_package_output_is_deterministic(tmp_path: Path) -> None:
    languages_path = _write_language_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "job_description": [
                "Japanese N2 preferred. English communication skill. TOEIC 800."
            ]
        }
    )
    kwargs = {
        "dataframe": dataframe,
        "languages_path": languages_path,
        "runtime_context": _runtime_context(descriptions=["job_description"]),
    }

    first = _run_language_package(**kwargs)
    second = _run_language_package(**kwargs)

    pd.testing.assert_frame_equal(first.dataframe, second.dataframe)
    assert first.report == second.report


def test_language_package_metadata_matches_phase_10_4i_contract() -> None:
    package = LanguagePackage()
    metadata = package.metadata

    assert package.package_id == "language"
    assert package.name == "Language Package"
    assert package.description == "Generate language attributes from job text columns."
    assert package.version == "1.0"
    assert package.enabled is True
    assert package.priority == 1000
    assert package.required_columns == ()
    assert package.produced_columns == ("languages",)
    assert metadata.package_id == "language"
    assert metadata.produced_columns == ["languages"]


def test_language_package_does_not_import_semantic_resolver() -> None:
    source = Path("core/knowledge_packages/language_package.py").read_text(
        encoding="utf-8"
    )

    assert "semantic_column_resolver" not in source
    assert "SemanticColumnResolver" not in source


def test_language_package_does_not_depend_on_prior_package_execution(
    tmp_path: Path,
) -> None:
    languages_path = _write_language_knowledge(tmp_path)
    dataframe = pd.DataFrame({"description": ["English communication skill."]})

    result = _engine_with_language_package().apply_packages(
        dataframe,
        package_names=["language"],
        package_configs={"language": {"languages_file": str(languages_path)}},
        runtime_context=_runtime_context(descriptions=["description"]),
    )

    assert result.report.applied_packages == ["language"]
    assert result.dataframe.loc[0, "languages"] == ["English"]


def test_current_package_order_includes_language() -> None:
    registry = PackageRegistry()
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
    ]


def test_unrelated_package_can_be_added_without_language_or_engine_changes() -> None:
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


def _engine_with_language_package() -> KnowledgePackageEngine:
    registry = PackageRegistry()
    registry.register(LanguagePackage())
    return KnowledgePackageEngine(registry)


def _run_language_package(
    dataframe: pd.DataFrame,
    languages_path: Path,
    runtime_context: dict[str, dict[str, list[str]]],
) -> object:
    return _engine_with_language_package().apply_packages(
        dataframe,
        package_configs={"language": {"languages_file": str(languages_path)}},
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


def _write_language_knowledge(tmp_path: Path) -> Path:
    languages_path = tmp_path / "languages.json"
    languages_path.write_text(
        json.dumps(
            {
                "Chinese": {
                    "aliases": [
                        "chinese",
                        "hsk",
                        "hsk3",
                        "hsk4",
                        "hsk5",
                        "hsk6",
                        "mandarin",
                        "tiếng trung",
                    ],
                },
                "English": {
                    "aliases": [
                        "b1",
                        "b2",
                        "c1",
                        "c2",
                        "english",
                        "ielts",
                        "ielts 6.5",
                        "tiếng anh",
                        "toeic",
                        "toeic 700",
                        "toeic 800",
                    ],
                },
                "French": {
                    "aliases": [
                        "french",
                        "tiếng pháp",
                    ],
                },
                "German": {
                    "aliases": [
                        "german",
                        "tiếng đức",
                    ],
                },
                "Japanese": {
                    "aliases": [
                        "japanese",
                        "jlpt",
                        "n1",
                        "n2",
                        "n3",
                        "n4",
                        "tiếng nhật",
                    ],
                },
                "Korean": {
                    "aliases": [
                        "korean",
                        "tiếng hàn",
                        "topik",
                        "topik 3",
                        "topik 4",
                        "topik 5",
                        "topik 6",
                    ],
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return languages_path
