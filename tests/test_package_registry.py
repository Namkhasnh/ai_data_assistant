from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd
import pytest

from core.knowledge_packages.base_package import BasePackage
from core.knowledge_packages.package_registry import PackageRegistry


class DemoPackage(BasePackage):
    package_id = "demo"
    name = "Demo Package"
    description = "Test package metadata."
    version = "2.0"
    priority = 50
    required_columns = ("source",)
    produced_columns = ("semantic",)

    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
    ) -> pd.DataFrame:
        return dataframe.copy(deep=True)


class LaterPackage(DemoPackage):
    package_id = "later"
    priority = 200


class EarlierPackage(DemoPackage):
    package_id = "earlier"
    priority = 10


def test_registry_registers_instances_and_returns_metadata() -> None:
    registry = PackageRegistry()
    package = DemoPackage()

    registry.register(package)

    assert registry.get("demo") is package
    assert registry.get("missing") is None
    metadata = package.metadata
    assert metadata.package_id == "demo"
    assert metadata.name == "Demo Package"
    assert metadata.description == "Test package metadata."
    assert metadata.version == "2.0"
    assert metadata.enabled is True
    assert metadata.priority == 50
    assert metadata.required_columns == ["source"]
    assert metadata.produced_columns == ["semantic"]


def test_registry_lists_packages_by_priority_then_package_id() -> None:
    registry = PackageRegistry()
    registry.register(LaterPackage())
    registry.register(DemoPackage())
    registry.register(EarlierPackage())

    assert [package.package_id for package in registry.list_packages()] == [
        "earlier",
        "demo",
        "later",
    ]


def test_registry_rejects_packages_without_package_id() -> None:
    class InvalidPackage(DemoPackage):
        package_id = ""

    registry = PackageRegistry()

    with pytest.raises(ValueError, match="package_id"):
        registry.register(InvalidPackage())
