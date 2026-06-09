from __future__ import annotations

from abc import ABC, abstractmethod


class BaseProvider(ABC):
    """Interface for swappable local or remote text generation providers."""

    name: str = "base"
    model: str | None = None

    @abstractmethod
    def health_check(self) -> bool:
        """Return whether the provider is available for generation."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate raw text from a prompt."""
