from __future__ import annotations

from core.enrichment.base_enricher import BaseEnricher, EnrichmentError
from core.enrichment.kb_enricher import KBEnricher


class EnricherRegistry:
    """Plugin registry for deterministic enrichment implementations."""

    def __init__(self) -> None:
        self._enrichers: dict[str, type[BaseEnricher]] = {}

    def register(self, name: str, enricher_class: type[BaseEnricher]) -> None:
        self._enrichers[name.strip().lower()] = enricher_class

    def create(self, name: str) -> BaseEnricher:
        enricher_class = self._enrichers.get(name.strip().lower())
        if enricher_class is None:
            raise EnrichmentError(f"Unsupported enricher: {name}")
        return enricher_class()

    def available_enrichers(self) -> list[str]:
        return sorted(self._enrichers)

    @classmethod
    def default(cls) -> EnricherRegistry:
        registry = cls()
        registry.register("knowledge", KBEnricher)
        return registry
