from __future__ import annotations

from core.standardization.base_standardizer import BaseStandardizer, StandardizationError
from core.standardization.categorical_standardizer import CategoricalStandardizer
from core.standardization.mapping_standardizer import MappingStandardizer
from core.standardization.numeric_standardizer import NumericStandardizer


class StandardizerRegistry:
    """Plugin registry for deterministic standardizers."""

    def __init__(self) -> None:
        self._standardizers: dict[str, type[BaseStandardizer]] = {}

    def register(self, name: str, standardizer_class: type[BaseStandardizer]) -> None:
        self._standardizers[name.strip().lower()] = standardizer_class

    def create(self, name: str) -> BaseStandardizer:
        standardizer_class = self._standardizers.get(name.strip().lower())
        if standardizer_class is None:
            raise StandardizationError(f"Unsupported standardizer: {name}")
        return standardizer_class()

    def available_standardizers(self) -> list[str]:
        return sorted(self._standardizers)

    @classmethod
    def default(cls) -> StandardizerRegistry:
        registry = cls()
        registry.register("mapping", MappingStandardizer)
        registry.register("categorical", CategoricalStandardizer)
        registry.register("numeric_bucket", NumericStandardizer)
        return registry
