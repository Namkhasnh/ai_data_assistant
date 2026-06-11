from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.knowledge_packages.job_title_package import JobTitlePackage
from core.knowledge_packages.knowledge_package_engine import KnowledgePackageEngine
from core.knowledge_packages.package_registry import PackageRegistry


def test_job_title_package_generates_semantic_columns_through_engine(
    tmp_path: Path,
) -> None:
    job_titles_path, seniority_path = _write_job_title_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "title": [
                "Senior CV Engineer",
                "Junior Machine Learning Engineer",
                "Lead NLP Engineer",
                "AI Wizard",
                "Unknown Engineer",
                "AI Wizard",
            ],
            "standardized_title": [
                "Computer Vision Engineer",
                "ML Engineer",
                "NLP Engineer",
                "AI Wizard",
                "Unknown Engineer",
                "AI Wizard",
            ],
        }
    )
    original = dataframe.copy(deep=True)

    result = _engine_with_job_title_package().apply_packages(
        dataframe,
        package_names=["job_title"],
        package_configs={
            "job_title": {
                "job_titles_file": str(job_titles_path),
                "seniority_file": str(seniority_path),
            }
        },
        runtime_context=_job_title_runtime_context(),
    )

    assert list(result.dataframe.columns) == [
        "title",
        "standardized_title",
        "job_group",
        "specialization",
        "seniority",
        "tech_domain",
    ]
    assert result.dataframe.loc[0, "job_group"] == "AI Engineer"
    assert result.dataframe.loc[0, "specialization"] == "Computer Vision"
    assert result.dataframe.loc[0, "seniority"] == "Senior"
    assert result.dataframe.loc[0, "tech_domain"] == "AI"
    assert result.dataframe.loc[1, "specialization"] == "Machine Learning"
    assert result.dataframe.loc[1, "seniority"] == "Junior"
    assert result.dataframe.loc[1, "tech_domain"] == "AI"
    assert result.dataframe.loc[2, "specialization"] == "Natural Language Processing"
    assert result.dataframe.loc[2, "seniority"] == "Lead"
    assert result.dataframe.loc[2, "tech_domain"] == "AI"
    assert pd.isna(result.dataframe.loc[3, "job_group"])
    assert pd.isna(result.dataframe.loc[3, "specialization"])
    assert pd.isna(result.dataframe.loc[3, "seniority"])
    assert pd.isna(result.dataframe.loc[3, "tech_domain"])
    assert result.report.applied_packages == ["job_title"]
    assert result.report.produced_columns_by_package == {
        "job_title": ["job_group", "specialization", "seniority", "tech_domain"]
    }
    assert result.report.unknown_values_by_package == {
        "job_title": ["AI Wizard", "Unknown Engineer"]
    }
    assert result.report.warnings.count(
        "Unknown values encountered in package 'job_title'."
    ) == 1
    pd.testing.assert_frame_equal(dataframe, original)


def test_job_title_package_resolves_aliases_without_canonicalizing_source_columns(
    tmp_path: Path,
) -> None:
    job_titles_path, seniority_path = _write_job_title_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "title": ["Lead CV Engineer"],
            "standardized_title": ["CV Engineer"],
        }
    )

    result = _engine_with_job_title_package().apply_packages(
        dataframe,
        package_configs={
            "job_title": {
                "job_titles_file": str(job_titles_path),
                "seniority_file": str(seniority_path),
            }
        },
        runtime_context=_job_title_runtime_context(),
    )

    assert result.dataframe.loc[0, "title"] == "Lead CV Engineer"
    assert result.dataframe.loc[0, "standardized_title"] == "CV Engineer"
    assert result.dataframe.loc[0, "job_group"] == "AI Engineer"
    assert result.dataframe.loc[0, "specialization"] == "Computer Vision"
    assert result.dataframe.loc[0, "seniority"] == "Lead"
    assert result.dataframe.loc[0, "tech_domain"] == "AI"


def test_job_title_package_does_not_overwrite_existing_columns(tmp_path: Path) -> None:
    job_titles_path, seniority_path = _write_job_title_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "title": ["Senior CV Engineer"],
            "standardized_title": ["Computer Vision Engineer"],
            "job_group": ["Existing Group"],
            "tech_domain": ["Existing Domain"],
        }
    )

    result = _engine_with_job_title_package().apply_packages(
        dataframe,
        package_configs={
            "job_title": {
                "job_titles_file": str(job_titles_path),
                "seniority_file": str(seniority_path),
            }
        },
        runtime_context=_job_title_runtime_context(),
    )

    assert result.dataframe.loc[0, "job_group"] == "Existing Group"
    assert result.dataframe.loc[0, "tech_domain"] == "Existing Domain"
    assert result.dataframe.loc[0, "specialization"] == "Computer Vision"
    assert result.dataframe.loc[0, "seniority"] == "Senior"
    assert result.report.produced_columns_by_package == {
        "job_title": ["specialization", "seniority"]
    }
    assert any("attempted to overwrite existing column: job_group" in warning for warning in result.report.warnings)
    assert any("attempted to overwrite existing column: tech_domain" in warning for warning in result.report.warnings)


def test_job_title_package_aggregates_duplicate_unknown_values_once(
    tmp_path: Path,
) -> None:
    job_titles_path, seniority_path = _write_job_title_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "title": ["AI Wizard"] * 500,
            "standardized_title": ["AI Wizard"] * 500,
        }
    )

    result = _engine_with_job_title_package().apply_packages(
        dataframe,
        package_configs={
            "job_title": {
                "job_titles_file": str(job_titles_path),
                "seniority_file": str(seniority_path),
            }
        },
        runtime_context=_job_title_runtime_context(),
    )

    assert result.report.unknown_values_by_package == {"job_title": ["AI Wizard"]}
    assert result.report.warnings.count(
        "Unknown values encountered in package 'job_title'."
    ) == 1
    assert all(pd.isna(value) for value in result.dataframe["job_group"])
    assert all(pd.isna(value) for value in result.dataframe["tech_domain"])


def test_job_title_package_missing_knowledge_file_warns_without_exception(
    tmp_path: Path,
) -> None:
    missing_job_titles_path = tmp_path / "missing_job_titles.json"
    _, seniority_path = _write_job_title_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "title": ["Senior CV Engineer"],
            "standardized_title": ["Computer Vision Engineer"],
        }
    )

    result = _engine_with_job_title_package().apply_packages(
        dataframe,
        package_configs={
            "job_title": {
                "job_titles_file": str(missing_job_titles_path),
                "seniority_file": str(seniority_path),
            }
        },
        runtime_context=_job_title_runtime_context(),
    )

    assert result.report.applied_packages == ["job_title"]
    assert any("missing knowledge file" in warning for warning in result.report.warnings)
    assert result.report.unknown_values_by_package == {
        "job_title": ["Computer Vision Engineer"]
    }
    assert pd.isna(result.dataframe.loc[0, "job_group"])
    assert pd.isna(result.dataframe.loc[0, "specialization"])
    assert pd.isna(result.dataframe.loc[0, "tech_domain"])
    assert result.dataframe.loc[0, "seniority"] == "Senior"


def test_job_title_package_metadata_matches_phase_10_4d_1_contract() -> None:
    metadata = JobTitlePackage().metadata

    assert metadata.package_id == "job_title"
    assert metadata.name == "Job Title Package"
    assert metadata.description == (
        "Generate semantic business attributes from standardized job titles."
    )
    assert metadata.version == "1.0"
    assert metadata.enabled is True
    assert metadata.priority == 100
    assert metadata.required_columns == []
    assert metadata.produced_columns == [
        "job_group",
        "specialization",
        "seniority",
        "tech_domain",
    ]


def _engine_with_job_title_package() -> KnowledgePackageEngine:
    registry = PackageRegistry()
    registry.register(JobTitlePackage())
    return KnowledgePackageEngine(registry)


def _job_title_runtime_context(
    title_column: str = "title",
    standardized_title_column: str = "standardized_title",
) -> dict[str, dict[str, list[str]]]:
    return {
        "semantic_columns": {
            "JOB_TITLE": [title_column],
            "STANDARDIZED_JOB_TITLE": [standardized_title_column],
        }
    }


def _write_job_title_knowledge(tmp_path: Path) -> tuple[Path, Path]:
    job_titles_path = tmp_path / "job_titles.json"
    seniority_path = tmp_path / "seniority.json"
    job_titles_path.write_text(
        json.dumps(
            {
                "Computer Vision Engineer": {
                    "aliases": [
                        "CV Engineer",
                        "Computer Vision Developer",
                        "Computer Vision Specialist",
                    ],
                    "job_group": "AI Engineer",
                    "specialization": "Computer Vision",
                    "tech_domain": "AI",
                },
                "Machine Learning Engineer": {
                    "aliases": ["ML Engineer"],
                    "job_group": "AI Engineer",
                    "specialization": "Machine Learning",
                    "tech_domain": "AI",
                },
                "NLP Engineer": {
                    "aliases": [],
                    "job_group": "AI Engineer",
                    "specialization": "Natural Language Processing",
                    "tech_domain": "AI",
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    seniority_path.write_text(
        json.dumps(
            {
                "Senior": "Senior",
                "Lead": "Lead",
                "Junior": "Junior",
                "Intern": "Intern",
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return job_titles_path, seniority_path
