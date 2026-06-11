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
from core.knowledge_packages.salary_package import SalaryPackage
from core.knowledge_packages.skill_package import SkillPackage


class HealthcareTestPackage(BasePackage):
    package_id = "healthcare_test"
    name = "Healthcare Test Package"
    priority = 500
    required_columns = ("test_code",)
    produced_columns = ("test_name",)

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
    ) -> pd.DataFrame:
        output = dataframe.copy(deep=True)
        output["test_name"] = output["test_code"].map({"CBC": "Complete Blood Count"})
        return output


def test_skill_package_extracts_unique_sorted_skills_from_both_text_columns(
    tmp_path: Path,
) -> None:
    skills_path = _write_skill_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "job_requirements": [
                "Python tốt. TensorFlow. Python là lợi thế. Biết cpp.",
            ],
            "job_description": [
                "Python. AWS. Docker. LangGraph. CrewAI. Claude Code.",
            ],
        }
    )
    original = dataframe.copy(deep=True)

    result = _engine_with_skill_package().apply_packages(
        dataframe,
        package_names=["skill"],
        package_configs={"skill": {"skills_file": str(skills_path)}},
        runtime_context=_skill_runtime_context(
            requirements=["job_requirements"],
            descriptions=["job_description"],
            extra={
                "JOB_TITLE": ["title"],
                "STANDARDIZED_JOB_TITLE": ["standardized_title"],
                "JOB_LOCATION": ["location"],
                "STANDARDIZED_LOCATION": ["standardized_location"],
                "SALARY_MIN": ["salary_min"],
                "SALARY_MAX": ["salary_max"],
                "SALARY_TYPE": ["salary_type"],
            },
        ),
    )

    assert list(result.dataframe.columns) == [
        "job_requirements",
        "job_description",
        "skills",
    ]
    assert result.dataframe.loc[0, "skills"] == [
        "AWS",
        "C++",
        "Docker",
        "Python",
        "TensorFlow",
    ]
    assert result.report.applied_packages == ["skill"]
    assert result.report.produced_columns_by_package == {"skill": ["skills"]}
    assert result.report.unknown_values_by_package == {}
    assert result.report.warnings == []
    pd.testing.assert_frame_equal(dataframe, original)


def test_skill_package_matches_case_insensitive_aliases(tmp_path: Path) -> None:
    skills_path = _write_skill_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "job_requirements": [
                "PYTHON, TF, amazon web services, SQL, pytorch, JAVA",
            ]
        }
    )

    result = _engine_with_skill_package().apply_packages(
        dataframe,
        package_configs={"skill": {"skills_file": str(skills_path)}},
        runtime_context=_skill_runtime_context(requirements=["job_requirements"]),
    )

    assert result.dataframe.loc[0, "skills"] == [
        "AWS",
        "Java",
        "Python",
        "PyTorch",
        "SQL",
        "TensorFlow",
    ]


def test_skill_package_returns_empty_list_when_no_known_skills_are_found(
    tmp_path: Path,
) -> None:
    skills_path = _write_skill_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "job_description": ["LangGraph CrewAI Claude Code"],
        }
    )

    result = _engine_with_skill_package().apply_packages(
        dataframe,
        package_configs={"skill": {"skills_file": str(skills_path)}},
        runtime_context=_skill_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "skills"] == []
    assert result.report.warnings == []
    assert result.report.unknown_values_by_package == {}


def test_skill_package_operates_with_single_usable_column(tmp_path: Path) -> None:
    skills_path = _write_skill_knowledge(tmp_path)
    dataframe = pd.DataFrame({"job_description": ["Có Docker và AWS."]})

    result = _engine_with_skill_package().apply_packages(
        dataframe,
        package_configs={"skill": {"skills_file": str(skills_path)}},
        runtime_context=_skill_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "skills"] == ["AWS", "Docker"]


def test_skill_package_skips_without_usable_text_columns() -> None:
    dataframe = pd.DataFrame({"title": ["Data Engineer"]})

    result = _engine_with_skill_package().apply_packages(
        dataframe,
        package_configs={"skill": {}},
    )

    assert list(result.dataframe.columns) == ["title"]
    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["skill"]
    assert "Skill package skipped; no usable semantic columns." in result.report.warnings


def test_skill_package_does_not_overwrite_existing_skills_column(
    tmp_path: Path,
) -> None:
    skills_path = _write_skill_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "job_requirements": ["Python Docker"],
            "skills": [["Existing Skill"]],
        }
    )

    result = _engine_with_skill_package().apply_packages(
        dataframe,
        package_configs={"skill": {"skills_file": str(skills_path)}},
        runtime_context=_skill_runtime_context(requirements=["job_requirements"]),
    )

    assert result.dataframe.loc[0, "skills"] == ["Existing Skill"]
    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["skill"]
    assert any("attempted to overwrite existing column: skills" in warning for warning in result.report.warnings)


def test_package_execution_order_is_deterministic_with_all_current_packages(
    tmp_path: Path,
) -> None:
    provinces_path = _write_location_knowledge(tmp_path)
    skills_path = _write_skill_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "title": ["Senior CV Engineer"],
            "standardized_title": ["Computer Vision Engineer"],
            "location": ["Quận Cầu Giấy, Hà Nội"],
            "standardized_location": ["Hà Nội"],
            "salary_min": [20],
            "salary_max": [30],
            "salary_type": ["range"],
            "job_requirements": ["Python tốt. TensorFlow."],
            "job_description": ["Python. AWS. Docker."],
        }
    )
    registry = PackageRegistry()
    registry.register(SkillPackage())
    registry.register(SalaryPackage())
    registry.register(LocationPackage())
    registry.register(JobTitlePackage())

    result = KnowledgePackageEngine(registry).apply_packages(
        dataframe,
        package_configs={
            "job_title": {},
            "location": {"provinces_file": str(provinces_path)},
            "salary": {},
            "skill": {"skills_file": str(skills_path)},
        },
        runtime_context=_skill_runtime_context(
            requirements=["job_requirements"],
            descriptions=["job_description"],
            extra={
                "JOB_TITLE": ["title"],
                "STANDARDIZED_JOB_TITLE": ["standardized_title"],
                "JOB_LOCATION": ["location"],
                "STANDARDIZED_LOCATION": ["standardized_location"],
                "SALARY_MIN": ["salary_min"],
                "SALARY_MAX": ["salary_max"],
                "SALARY_TYPE": ["salary_type"],
            },
        ),
    )

    assert result.report.applied_packages == ["job_title", "location", "salary", "skill"]
    assert result.report.produced_columns_by_package == {
        "job_title": ["job_group", "specialization", "seniority", "tech_domain"],
        "location": ["city", "province", "region", "country"],
        "salary": ["salary_avg", "currency", "salary_unit"],
        "skill": ["skills"],
    }
    assert result.dataframe.loc[0, "skills"] == [
        "AWS",
        "Docker",
        "Python",
        "TensorFlow",
    ]


def test_skill_package_metadata_matches_phase_10_1_contract() -> None:
    metadata = SkillPackage().metadata

    assert metadata.package_id == "skill"
    assert metadata.name == "Skill Package"
    assert metadata.description == "Generate semantic skill attributes from free-text job columns."
    assert metadata.version == "1.0"
    assert metadata.enabled is True
    assert metadata.priority == 400
    assert metadata.required_columns == []
    assert metadata.produced_columns == ["skills"]


def test_healthcare_test_package_can_be_added_without_skill_or_engine_changes() -> None:
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


def _engine_with_skill_package() -> KnowledgePackageEngine:
    registry = PackageRegistry()
    registry.register(SkillPackage())
    return KnowledgePackageEngine(registry)


def _skill_runtime_context(
    requirements: list[str] | None = None,
    descriptions: list[str] | None = None,
    extra: dict[str, list[str]] | None = None,
) -> dict[str, dict[str, list[str]]]:
    semantic_columns = {
        "JOB_REQUIREMENTS": requirements or [],
        "JOB_DESCRIPTION": descriptions or [],
    }
    if extra:
        semantic_columns.update(extra)
    return {
        "semantic_columns": semantic_columns
    }


def _write_skill_knowledge(tmp_path: Path) -> Path:
    skills_path = tmp_path / "skills.json"
    skills_path.write_text(
        json.dumps(
            {
                "Python": {
                    "aliases": ["python"],
                    "category": "programming_language",
                },
                "Java": {
                    "aliases": ["java"],
                    "category": "programming_language",
                },
                "C++": {
                    "aliases": ["c++", "cpp"],
                    "category": "programming_language",
                },
                "TensorFlow": {
                    "aliases": ["tensorflow", "tf"],
                    "category": "framework",
                },
                "PyTorch": {
                    "aliases": ["pytorch"],
                    "category": "framework",
                },
                "SQL": {
                    "aliases": ["sql"],
                    "category": "database",
                },
                "Docker": {
                    "aliases": ["docker"],
                    "category": "tool",
                },
                "AWS": {
                    "aliases": ["aws", "amazon web services"],
                    "category": "cloud",
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return skills_path


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
                }
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return provinces_path
