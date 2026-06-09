from __future__ import annotations

import pytest

from core.standardization.base_standardizer import StandardizationError
from core.standardization.mapping_standardizer import MappingStandardizer
from core.standardization.standardizer_registry import StandardizerRegistry


def test_standardizer_registry_dispatches_registered_standardizers() -> None:
    registry = StandardizerRegistry()
    registry.register("mapping", MappingStandardizer)

    standardizer = registry.create("mapping")

    assert isinstance(standardizer, MappingStandardizer)
    assert registry.available_standardizers() == ["mapping"]


def test_default_standardizer_registry_exposes_core_standardizers() -> None:
    registry = StandardizerRegistry.default()

    assert registry.available_standardizers() == [
        "categorical",
        "mapping",
        "numeric_bucket",
    ]


def test_standardizer_registry_rejects_unknown_standardizer() -> None:
    registry = StandardizerRegistry.default()

    with pytest.raises(StandardizationError, match="Unsupported standardizer"):
        registry.create("currency")
