from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
import inspect
from typing import Any

import pandas as pd

from models.knowledge_package import KnowledgePackageMetadata


class BasePackage(ABC):
    """Base interface for deterministic, additive knowledge packages."""

    package_id: str = ""
    name: str = ""
    description: str = ""
    version: str = "1.0"
    enabled: bool = True
    priority: int = 100
    required_columns: tuple[str, ...] = ()
    produced_columns: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.warnings: list[str] = []
        self.unknown_values: list[str] = []
        self.skip_requested: bool = False

    def reset_run_state(self) -> None:
        """Clear package-scoped transient state before one engine run."""

        self.warnings = []
        self.unknown_values = []
        self.skip_requested = False

    def request_skip(self, warning: str | None = None) -> None:
        """Request a graceful package skip without engine-level failure warnings."""

        self.skip_requested = True
        if warning is not None:
            self.warnings.append(warning)

    def record_unknown_value(self, value: object) -> None:
        """Record one package-level unknown value for generic report aggregation."""

        if value is None or pd.isna(value):
            return
        value_text = str(value).strip()
        if value_text:
            self.unknown_values.append(value_text)

    def apply_with_context(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Apply a package with optional runtime dependencies when supported."""

        signature = inspect.signature(self.apply)
        parameters = signature.parameters
        if "runtime_context" in parameters:
            return self.apply(
                dataframe,
                knowledge_config,
                runtime_context=runtime_context,
            )
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
            return self.apply(
                dataframe,
                knowledge_config,
                runtime_context=runtime_context,
            )
        positional_parameters = [
            parameter
            for parameter in parameters.values()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if len(positional_parameters) >= 3:
            return self.apply(dataframe, knowledge_config, runtime_context)
        return self.apply(dataframe, knowledge_config)

    @property
    def metadata(self) -> KnowledgePackageMetadata:
        """Return package metadata from the package object itself."""

        return KnowledgePackageMetadata(
            package_id=self.package_id,
            name=self.name or self.package_id,
            description=self.description,
            version=self.version,
            enabled=self.enabled,
            priority=self.priority,
            warnings=list(self.warnings),
            required_columns=list(self.required_columns),
            produced_columns=list(self.produced_columns),
        )

    @abstractmethod
    def apply(
        self,
        dataframe: pd.DataFrame,
        knowledge_config: Mapping[str, Any] | None,
        runtime_context: Mapping[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Return a dataframe with package-produced columns appended."""
