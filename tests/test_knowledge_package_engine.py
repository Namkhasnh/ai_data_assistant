from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from core.knowledge_packages.base_package import BasePackage
from core.knowledge_packages.knowledge_package_engine import KnowledgePackageEngine
from core.knowledge_packages.package_registry import PackageRegistry
from models.knowledge_package import KnowledgePackageResult


class SemanticAppendPackage(BasePackage):
    package_id = "semantic_append"
    name = "Semantic Append"
    description = "Adds deterministic semantic labels for tests."
    required_columns = ("source",)
    produced_columns = ("semantic_label",)

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
    ) -> pd.DataFrame:
        output = dataframe.copy(deep=True)
        output["semantic_label"] = output["source"].map({"A": "Alpha"})
        return output


class EarlyPackage(SemanticAppendPackage):
    package_id = "early"
    priority = 10
    produced_columns = ("early_column",)

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
    ) -> pd.DataFrame:
        output = dataframe.copy(deep=True)
        output["early_column"] = "early"
        return output


class TieBreakPackage(SemanticAppendPackage):
    package_id = "alpha"
    priority = 10
    produced_columns = ("alpha_column",)

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
    ) -> pd.DataFrame:
        output = dataframe.copy(deep=True)
        output["alpha_column"] = "alpha"
        return output


class LatePackage(SemanticAppendPackage):
    package_id = "late"
    priority = 200
    produced_columns = ("late_column",)

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
    ) -> pd.DataFrame:
        output = dataframe.copy(deep=True)
        output["late_column"] = "late"
        return output


class MissingRequiredColumnPackage(SemanticAppendPackage):
    package_id = "missing_required"
    required_columns = ("missing_source",)


class OverwriteAttemptPackage(SemanticAppendPackage):
    package_id = "overwrite_attempt"
    produced_columns = ("source", "new_semantic")

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
    ) -> pd.DataFrame:
        output = dataframe.copy(deep=True)
        output["source"] = "changed"
        output["new_semantic"] = "safe"
        return output


class FailurePackage(SemanticAppendPackage):
    package_id = "failure"

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
    ) -> pd.DataFrame:
        raise RuntimeError("package exploded")


class RetailProductPackage(SemanticAppendPackage):
    package_id = "retail_product"
    name = "Retail Product"
    priority = 25
    produced_columns = ("product_group",)

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
    ) -> pd.DataFrame:
        output = dataframe.copy(deep=True)
        output["product_group"] = output["source"].map({"A": "Retail Group"})
        return output


class RuntimeContextPackage(SemanticAppendPackage):
    package_id = "runtime_context"
    name = "Runtime Context"
    produced_columns = ("context_value",)

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        output = dataframe.copy(deep=True)
        output["context_value"] = (runtime_context or {}).get("value")
        return output


class LocationPackage(SemanticAppendPackage):
    package_id = "location"
    name = "Location"
    priority = 30
    produced_columns = ("region",)

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
    ) -> pd.DataFrame:
        output = dataframe.copy(deep=True)
        output["region"] = output["source"].map({"A": "North"})
        return output


def test_engine_applies_packages_additively_without_mutating_source_dataframe() -> None:
    dataframe = pd.DataFrame({"source": ["A", "Unknown"], "existing": [1, 2]})
    original = dataframe.copy(deep=True)
    registry = PackageRegistry()
    registry.register(SemanticAppendPackage())

    result = KnowledgePackageEngine(registry).apply_packages(
        dataframe,
        package_configs={"semantic_append": {}},
    )

    assert isinstance(result, KnowledgePackageResult)
    assert list(result.dataframe.columns) == ["source", "existing", "semantic_label"]
    assert result.dataframe.loc[0, "semantic_label"] == "Alpha"
    assert pd.isna(result.dataframe.loc[1, "semantic_label"])
    assert result.report.applied_packages == ["semantic_append"]
    assert result.report.skipped_packages == []
    assert result.report.produced_columns == ["semantic_label"]
    assert result.report.produced_columns_by_package == {
        "semantic_append": ["semantic_label"]
    }
    assert result.report.unknown_values_by_package == {}
    pd.testing.assert_frame_equal(dataframe, original)


def test_engine_orders_packages_by_priority_then_package_id() -> None:
    dataframe = pd.DataFrame({"source": ["A"]})
    registry = PackageRegistry()
    registry.register(LatePackage())
    registry.register(EarlyPackage())
    registry.register(TieBreakPackage())

    result = KnowledgePackageEngine(registry).apply_packages(
        dataframe,
        package_names=["late", "early", "alpha"],
        package_configs={"late": {}, "early": {}, "alpha": {}},
    )

    assert result.report.applied_packages == ["alpha", "early", "late"]
    assert list(result.dataframe.columns) == [
        "source",
        "alpha_column",
        "early_column",
        "late_column",
    ]


def test_engine_warns_for_unknown_package_and_missing_required_columns() -> None:
    dataframe = pd.DataFrame({"source": ["A"]})
    registry = PackageRegistry()
    registry.register(MissingRequiredColumnPackage())

    result = KnowledgePackageEngine(registry).apply_packages(
        dataframe,
        package_names=["unknown_package", "missing_required"],
        package_configs={"missing_required": {}},
    )

    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["unknown_package", "missing_required"]
    assert any("Unknown knowledge package skipped" in warning for warning in result.report.warnings)
    assert any("missing required columns" in warning for warning in result.report.warnings)
    assert list(result.dataframe.columns) == ["source"]


def test_engine_warns_for_missing_config_and_missing_knowledge_files(tmp_path: Path) -> None:
    dataframe = pd.DataFrame({"source": ["A"]})
    registry = PackageRegistry()
    registry.register(SemanticAppendPackage())
    registry.register(LatePackage())
    missing_file = tmp_path / "missing_knowledge.json"

    result = KnowledgePackageEngine(registry).apply_packages(
        dataframe,
        package_names=["semantic_append", "late"],
        package_configs={"late": {"knowledge_file": str(missing_file)}},
    )

    assert result.report.applied_packages == ["semantic_append"]
    assert result.report.skipped_packages == ["late"]
    assert any("has no package configuration" in warning for warning in result.report.warnings)
    assert any("missing knowledge files" in warning for warning in result.report.warnings)


def test_engine_prevents_source_mutation_and_existing_column_overwrite() -> None:
    dataframe = pd.DataFrame({"source": ["A"]})
    registry = PackageRegistry()
    registry.register(OverwriteAttemptPackage())

    result = KnowledgePackageEngine(registry).apply_packages(
        dataframe,
        package_configs={"overwrite_attempt": {}},
    )

    assert result.dataframe.loc[0, "source"] == "A"
    assert result.dataframe.loc[0, "new_semantic"] == "safe"
    assert result.report.produced_columns == ["new_semantic"]
    assert any("attempted to modify source column" in warning for warning in result.report.warnings)
    assert any("attempted to overwrite existing column" in warning for warning in result.report.warnings)


def test_engine_warns_for_package_failures_without_raising() -> None:
    dataframe = pd.DataFrame({"source": ["A"]})
    registry = PackageRegistry()
    registry.register(FailurePackage())

    result = KnowledgePackageEngine(registry).apply_packages(
        dataframe,
        package_configs={"failure": {}},
    )

    assert result.report.applied_packages == []
    assert result.report.skipped_packages == ["failure"]
    assert any("failed" in warning for warning in result.report.warnings)
    assert list(result.dataframe.columns) == ["source"]


def test_retail_product_package_can_be_added_without_engine_or_registry_changes() -> None:
    dataframe = pd.DataFrame({"source": ["A"]})
    registry = PackageRegistry()
    registry.register(RetailProductPackage())

    result = KnowledgePackageEngine(registry).apply_packages(
        dataframe,
        package_names=["retail_product"],
        package_configs={"retail_product": {}},
    )

    assert result.report.applied_packages == ["retail_product"]
    assert result.report.produced_columns_by_package == {
        "retail_product": ["product_group"]
    }
    assert result.dataframe.loc[0, "product_group"] == "Retail Group"


def test_location_package_can_be_added_without_existing_package_or_engine_changes() -> None:
    dataframe = pd.DataFrame({"source": ["A"]})
    registry = PackageRegistry()
    registry.register(LocationPackage())

    result = KnowledgePackageEngine(registry).apply_packages(
        dataframe,
        package_names=["location"],
        package_configs={"location": {}},
    )

    assert result.report.applied_packages == ["location"]
    assert result.report.produced_columns_by_package == {"location": ["region"]}
    assert result.dataframe.loc[0, "region"] == "North"


def test_engine_dispatches_old_and_runtime_context_packages_generically() -> None:
    dataframe = pd.DataFrame({"source": ["A"]})
    registry = PackageRegistry()
    registry.register(SemanticAppendPackage())
    registry.register(RuntimeContextPackage())

    result = KnowledgePackageEngine(registry).apply_packages(
        dataframe,
        package_names=["semantic_append", "runtime_context"],
        package_configs={"semantic_append": {}, "runtime_context": {}},
        runtime_context={"value": "received"},
    )

    assert result.report.applied_packages == ["runtime_context", "semantic_append"]
    assert result.dataframe.loc[0, "semantic_label"] == "Alpha"
    assert result.dataframe.loc[0, "context_value"] == "received"


def test_engine_has_no_package_specific_branches() -> None:
    source = Path("core/knowledge_packages/knowledge_package_engine.py").read_text(
        encoding="utf-8"
    )

    assert 'package_id == "job_title"' not in source
    assert 'package_id == "location"' not in source
    assert 'package_id == "salary"' not in source
    assert 'package_id == "skill"' not in source
    assert "if package_id ==" not in source
