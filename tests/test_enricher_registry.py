from __future__ import annotations

import pytest

from core.enrichment.base_enricher import EnrichmentError
from core.enrichment.enricher_registry import EnricherRegistry
from core.enrichment.kb_enricher import KBEnricher


def test_enricher_registry_dispatches_registered_enrichers() -> None:
    registry = EnricherRegistry()
    registry.register("knowledge", KBEnricher)

    enricher = registry.create("knowledge")

    assert isinstance(enricher, KBEnricher)
    assert registry.available_enrichers() == ["knowledge"]


def test_default_enricher_registry_exposes_knowledge_only() -> None:
    registry = EnricherRegistry.default()

    assert registry.available_enrichers() == ["knowledge"]


def test_enricher_registry_rejects_unknown_enricher() -> None:
    registry = EnricherRegistry.default()

    with pytest.raises(EnrichmentError, match="Unsupported enricher"):
        registry.create("llm")
