from __future__ import annotations

from core.knowledge_packages.base_package import BasePackage


class PackageRegistry:
    """Registry of optional knowledge package instances."""

    def __init__(self) -> None:
        self._packages: dict[str, BasePackage] = {}

    def register(self, package: BasePackage) -> None:
        """Register or replace a package instance by package_id."""

        if not package.package_id:
            raise ValueError("Knowledge package package_id must not be empty")
        self._packages[package.package_id] = package

    def get(self, package_id: str) -> BasePackage | None:
        """Return a registered package instance, or None when unknown."""

        return self._packages.get(package_id)

    def list_packages(self) -> list[BasePackage]:
        """Return packages in deterministic execution order."""

        return sorted(
            self._packages.values(),
            key=lambda package: (package.priority, package.package_id),
        )
