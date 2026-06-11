from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from core.knowledge_packages.base_package import BasePackage
from core.knowledge_packages.knowledge_package_engine import KnowledgePackageEngine
from core.knowledge_packages.package_registry import PackageRegistry
from core.knowledge_packages.skill_package import SkillPackage
from core.semantic_resolver.semantic_column_resolver import SemanticColumnResolver
from models.semantic_column_registry import SemanticColumnRegistry


class ContextEchoPackage(BasePackage):
    package_id = "context_echo"
    name = "Context Echo"
    required_columns = ("source",)
    produced_columns = ("context_seen",)

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        output = dataframe.copy(deep=True)
        output["context_seen"] = bool((runtime_context or {}).get("semantic_columns"))
        return output


def test_skill_package_uses_semantic_columns_for_dataset_a(tmp_path: Path) -> None:
    dataframe = pd.DataFrame(
        {
            "requirements": ["Python tốt. TensorFlow."],
            "description": ["Python. AWS. Docker."],
        }
    )

    result = _run_skill_package(
        dataframe=dataframe,
        registry=SemanticColumnRegistry(
            columns_by_semantic_type={
                "JOB_REQUIREMENTS": ["requirements"],
                "JOB_DESCRIPTION": ["description"],
            }
        ),
    )

    assert result.dataframe.loc[0, "skills"] == [
        "AWS",
        "Docker",
        "Python",
        "TensorFlow",
    ]


def test_skill_package_uses_semantic_columns_for_dataset_b(tmp_path: Path) -> None:
    dataframe = pd.DataFrame(
        {
            "must_have": ["Python tốt. TensorFlow."],
            "qualifications": ["Python. AWS. Docker."],
        }
    )

    result = _run_skill_package(
        dataframe=dataframe,
        registry=SemanticColumnRegistry(
            columns_by_semantic_type={
                "JOB_REQUIREMENTS": ["must_have", "qualifications"],
            }
        ),
    )

    assert result.dataframe.loc[0, "skills"] == [
        "AWS",
        "Docker",
        "Python",
        "TensorFlow",
    ]


def test_skill_package_uses_semantic_columns_for_dataset_c(tmp_path: Path) -> None:
    dataframe = pd.DataFrame(
        {
            "Yêu cầu": ["Python tốt. TensorFlow."],
            "Mô tả công việc": ["Python. AWS. Docker."],
        }
    )

    result = _run_skill_package(
        dataframe=dataframe,
        registry=SemanticColumnRegistry(
            columns_by_semantic_type={
                "JOB_REQUIREMENTS": ["Yêu cầu"],
                "JOB_DESCRIPTION": ["Mô tả công việc"],
            }
        ),
    )

    assert result.dataframe.loc[0, "skills"] == [
        "AWS",
        "Docker",
        "Python",
        "TensorFlow",
    ]


def test_skill_package_has_no_physical_column_assumptions() -> None:
    dataframe = pd.DataFrame(
        {
            "job_requirements": ["Python Docker"],
            "job_description": ["AWS TensorFlow"],
        }
    )

    result = _run_skill_package(
        dataframe=dataframe,
        registry=SemanticColumnRegistry(),
    )

    assert list(result.dataframe.columns) == ["job_requirements", "job_description"]
    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["skill"]
    assert result.report.warnings == ["Skill package skipped; no usable semantic columns."]


def test_skill_package_deduplicates_across_semantic_columns() -> None:
    dataframe = pd.DataFrame(
        {
            "must_have": ["Python. Python. TensorFlow."],
            "qualifications": ["Python. AWS. Docker. AWS."],
            "responsibilities": ["Docker. TensorFlow."],
        }
    )

    result = _run_skill_package(
        dataframe=dataframe,
        registry=SemanticColumnRegistry(
            columns_by_semantic_type={
                "JOB_REQUIREMENTS": ["must_have", "qualifications"],
                "JOB_DESCRIPTION": ["responsibilities"],
            }
        ),
    )

    assert result.dataframe.loc[0, "skills"] == [
        "AWS",
        "Docker",
        "Python",
        "TensorFlow",
    ]


def test_empty_semantic_columns_warn_only_from_skill_package(caplog) -> None:
    dataframe = pd.DataFrame({"source": ["Python Docker"]})
    resolver = SemanticColumnResolver(SemanticColumnRegistry())

    result = _run_skill_package(
        dataframe=dataframe,
        registry=resolver.registry,
    )

    assert resolver.get_columns(["JOB_REQUIREMENTS", "JOB_DESCRIPTION"]) == []
    assert result.report.applied_packages == []
    assert "Skill package skipped; no usable semantic columns." in result.report.warnings
    assert caplog.records == []


def test_skill_package_does_not_import_semantic_column_resolver() -> None:
    source = Path("core/knowledge_packages/skill_package.py").read_text(encoding="utf-8")

    assert "semantic_column_resolver" not in source
    assert "SemanticColumnResolver" not in source


def test_engine_runtime_context_injection_is_generic() -> None:
    dataframe = pd.DataFrame({"source": ["value"]})
    registry = PackageRegistry()
    registry.register(ContextEchoPackage())
    runtime_context = {"semantic_columns": {"ANY": ["source"]}}

    result = KnowledgePackageEngine(registry).apply_packages(
        dataframe,
        package_names=["context_echo"],
        package_configs={"context_echo": {}},
        runtime_context=runtime_context,
    )

    assert bool(result.dataframe.loc[0, "context_seen"]) is True
    assert result.report.applied_packages == ["context_echo"]


def test_engine_has_no_skill_specific_runtime_logic() -> None:
    source = Path("core/knowledge_packages/knowledge_package_engine.py").read_text(
        encoding="utf-8"
    )

    assert 'package_id == "skill"' not in source
    assert "package_id == 'skill'" not in source


def _run_skill_package(
    dataframe: pd.DataFrame,
    registry: SemanticColumnRegistry,
) -> Any:
    package_registry = PackageRegistry()
    package_registry.register(SkillPackage())
    resolver = SemanticColumnResolver(registry)
    runtime_context = {
        "semantic_columns": {
            semantic_type: resolver.get_columns([semantic_type])
            for semantic_type in resolver.available_semantic_types()
        }
    }
    return KnowledgePackageEngine(package_registry).apply_packages(
        dataframe,
        package_names=["skill"],
        package_configs={"skill": {"skills_file": "knowledge/jobs/skills.json"}},
        runtime_context=runtime_context,
    )
