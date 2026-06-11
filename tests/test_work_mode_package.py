from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from core.knowledge_packages.base_package import BasePackage
from core.knowledge_packages.knowledge_package_engine import KnowledgePackageEngine
from core.knowledge_packages.package_registry import PackageRegistry
from core.knowledge_packages.work_mode_package import WorkModePackage


class RetailProductPackage(BasePackage):
    package_id = "retail_product"
    name = "Retail Product Package"
    priority = 50
    required_columns: tuple[str, ...] = ()
    produced_columns = ("product_family",)

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        output = dataframe.copy(deep=True)
        output["product_family"] = "General"
        return output


def test_work_mode_package_appends_work_mode_from_semantic_columns(
    tmp_path: Path,
) -> None:
    work_modes_path = _write_work_mode_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "job_description": ["Hybrid working, 3 days onsite"],
            "benefits": ["Team lunch"],
        }
    )
    original = dataframe.copy(deep=True)

    result = _engine_with_work_mode_package().apply_packages(
        dataframe,
        package_names=["work_mode"],
        package_configs={"work_mode": {"work_modes_file": str(work_modes_path)}},
        runtime_context=_runtime_context(
            descriptions=["job_description"],
            benefits=["benefits"],
        ),
    )

    assert list(result.dataframe.columns) == [
        "job_description",
        "benefits",
        "work_mode",
    ]
    assert result.dataframe.loc[0, "work_mode"] == "Hybrid"
    assert result.report.applied_packages == ["work_mode"]
    assert result.report.produced_columns_by_package == {
        "work_mode": ["work_mode"]
    }
    assert result.report.unknown_values_by_package == {}
    assert result.report.warnings == []
    pd.testing.assert_frame_equal(dataframe, original)


def test_work_mode_package_does_not_overwrite_existing_work_mode(
    tmp_path: Path,
) -> None:
    work_modes_path = _write_work_mode_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "job_description": ["WFH full time"],
            "work_mode": ["Existing Mode"],
        }
    )

    result = _engine_with_work_mode_package().apply_packages(
        dataframe,
        package_configs={"work_mode": {"work_modes_file": str(work_modes_path)}},
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe.loc[0, "work_mode"] == "Existing Mode"
    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["work_mode"]
    assert any(
        "attempted to overwrite existing column: work_mode" in warning
        for warning in result.report.warnings
    )


def test_work_mode_package_skips_without_usable_semantic_columns() -> None:
    dataframe = pd.DataFrame({"job_description": ["WFH full time"]})

    result = _engine_with_work_mode_package().apply_packages(
        dataframe,
        package_configs={"work_mode": {}},
        runtime_context={"semantic_columns": {}},
    )

    assert list(result.dataframe.columns) == ["job_description"]
    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["work_mode"]
    assert result.report.warnings == [
        "Work mode package skipped; no usable semantic columns."
    ]


def test_work_mode_package_runtime_context_none_skips_gracefully() -> None:
    dataframe = pd.DataFrame({"job_description": ["WFH full time"]})

    result = _engine_with_work_mode_package().apply_packages(
        dataframe,
        package_names=["work_mode"],
        package_configs={"work_mode": {}},
        runtime_context=None,
    )

    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["work_mode"]
    assert result.report.warnings == [
        "Work mode package skipped; no usable semantic columns."
    ]


def test_work_mode_package_matches_case_insensitive_english_aliases(
    tmp_path: Path,
) -> None:
    work_modes_path = _write_work_mode_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "description": ["WFH FULL TIME"],
            "work_location": ["Work at office in Hanoi"],
        }
    )

    result = _engine_with_work_mode_package().apply_packages(
        dataframe,
        package_configs={"work_mode": {"work_modes_file": str(work_modes_path)}},
        runtime_context=_runtime_context(
            descriptions=["description"],
            work_locations=["work_location"],
        ),
    )

    assert result.dataframe.loc[0, "work_mode"] == "Remote"


def test_work_mode_package_matches_vietnamese_aliases(tmp_path: Path) -> None:
    work_modes_path = _write_work_mode_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "benefits": [
                "Có chính sách làm việc từ xa.",
                "Mô hình làm việc kết hợp theo tuần.",
                "Làm việc tại văn phòng ở Hà Nội.",
            ]
        }
    )

    result = _engine_with_work_mode_package().apply_packages(
        dataframe,
        package_configs={"work_mode": {"work_modes_file": str(work_modes_path)}},
        runtime_context=_runtime_context(benefits=["benefits"]),
    )

    assert result.dataframe["work_mode"].tolist() == ["Remote", "Hybrid", "Onsite"]


def test_work_mode_package_resolves_multiple_matches_by_knowledge_priority(
    tmp_path: Path,
) -> None:
    work_modes_path = _write_work_mode_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "job_description": [
                "Remote role with hybrid working option.",
                "Hybrid working, 3 days onsite.",
            ]
        }
    )

    result = _engine_with_work_mode_package().apply_packages(
        dataframe,
        package_configs={"work_mode": {"work_modes_file": str(work_modes_path)}},
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert result.dataframe["work_mode"].tolist() == ["Remote", "Hybrid"]


def test_work_mode_package_deduplicates_text_inputs_without_changing_output(
    tmp_path: Path,
) -> None:
    work_modes_path = _write_work_mode_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "job_description": ["WFH full time"],
            "benefits": ["WFH full time"],
            "working_time": ["WFH full time"],
        }
    )

    result = _engine_with_work_mode_package().apply_packages(
        dataframe,
        package_configs={"work_mode": {"work_modes_file": str(work_modes_path)}},
        runtime_context=_runtime_context(
            descriptions=["job_description"],
            benefits=["benefits"],
            working_times=["working_time"],
        ),
    )

    assert result.dataframe.loc[0, "work_mode"] == "Remote"
    assert result.report.warnings == []


def test_work_mode_package_unknown_text_remains_quiet(tmp_path: Path) -> None:
    work_modes_path = _write_work_mode_knowledge(tmp_path)
    dataframe = pd.DataFrame({"job_description": ["No work arrangement mentioned."]})

    result = _engine_with_work_mode_package().apply_packages(
        dataframe,
        package_configs={"work_mode": {"work_modes_file": str(work_modes_path)}},
        runtime_context=_runtime_context(descriptions=["job_description"]),
    )

    assert pd.isna(result.dataframe.loc[0, "work_mode"])
    assert result.report.warnings == []
    assert result.report.unknown_values_by_package == {}


def test_work_mode_package_output_is_deterministic(tmp_path: Path) -> None:
    work_modes_path = _write_work_mode_knowledge(tmp_path)
    dataframe = pd.DataFrame(
        {
            "job_description": ["Hybrid working, 3 days onsite"],
            "benefits": ["linh hoạt"],
        }
    )
    kwargs = {
        "package_configs": {"work_mode": {"work_modes_file": str(work_modes_path)}},
        "runtime_context": _runtime_context(
            descriptions=["job_description"],
            benefits=["benefits"],
        ),
    }

    first = _engine_with_work_mode_package().apply_packages(dataframe, **kwargs)
    second = _engine_with_work_mode_package().apply_packages(dataframe, **kwargs)

    pd.testing.assert_frame_equal(first.dataframe, second.dataframe)
    assert first.report == second.report


def test_work_mode_package_metadata_matches_phase_10_4a_contract() -> None:
    metadata = WorkModePackage().metadata

    assert metadata.package_id == "work_mode"
    assert metadata.name == "Work Mode Package"
    assert metadata.priority == 500
    assert metadata.required_columns == []
    assert metadata.produced_columns == ["work_mode"]


def test_work_mode_package_does_not_import_semantic_resolver() -> None:
    source = Path("core/knowledge_packages/work_mode_package.py").read_text(
        encoding="utf-8"
    )

    assert "semantic_column_resolver" not in source
    assert "SemanticColumnResolver" not in source


def test_retail_package_can_be_added_without_work_mode_or_engine_changes() -> None:
    dataframe = pd.DataFrame({"product_name": ["Laptop"]})
    registry = PackageRegistry()
    registry.register(RetailProductPackage())

    result = KnowledgePackageEngine(registry).apply_packages(
        dataframe,
        package_names=["retail_product"],
        package_configs={"retail_product": {}},
    )

    assert result.report.applied_packages == ["retail_product"]
    assert result.report.produced_columns_by_package == {
        "retail_product": ["product_family"]
    }
    assert result.dataframe.loc[0, "product_family"] == "General"


def _engine_with_work_mode_package() -> KnowledgePackageEngine:
    registry = PackageRegistry()
    registry.register(WorkModePackage())
    return KnowledgePackageEngine(registry)


def _runtime_context(
    descriptions: list[str] | None = None,
    benefits: list[str] | None = None,
    working_times: list[str] | None = None,
    work_locations: list[str] | None = None,
) -> dict[str, dict[str, list[str]]]:
    return {
        "semantic_columns": {
            "JOB_DESCRIPTION": descriptions or [],
            "BENEFITS": benefits or [],
            "WORKING_TIME": working_times or [],
            "WORK_LOCATION": work_locations or [],
        }
    }


def _write_work_mode_knowledge(tmp_path: Path) -> Path:
    work_modes_path = tmp_path / "work_modes.json"
    work_modes_path.write_text(
        json.dumps(
            {
                "Remote": {
                    "priority": 100,
                    "aliases": [
                        "fully remote",
                        "làm việc từ xa",
                        "remote",
                        "wfh",
                        "work from home",
                    ],
                },
                "Hybrid": {
                    "priority": 200,
                    "aliases": [
                        "2 days remote",
                        "3 days onsite",
                        "hybrid",
                        "hybrid working",
                        "linh hoạt",
                        "làm việc kết hợp",
                    ],
                },
                "Onsite": {
                    "priority": 300,
                    "aliases": [
                        "full office",
                        "làm việc tại văn phòng",
                        "onsite",
                        "work at office",
                    ],
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return work_modes_path
