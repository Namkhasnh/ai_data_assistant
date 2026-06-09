from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models.column_profile import ColumnProfile
from models.dataset import DatasetMetadata
from models.semantic_tag import SemanticDetectionReport, SemanticTag


class PromptBuilder:
    """Build compact prompts from profiling and semantic artifacts."""

    def __init__(
        self,
        template_path: str | Path = "core/ai/prompts/rule_generation.txt",
        max_sample_values: int = 5,
        max_top_values: int = 5,
        max_value_length: int = 120,
    ) -> None:
        self.template_path = Path(template_path)
        self.max_sample_values = max_sample_values
        self.max_top_values = max_top_values
        self.max_value_length = max_value_length

    def build_rule_generation_prompt(
        self,
        metadata: DatasetMetadata,
        semantic_report: SemanticDetectionReport,
    ) -> str:
        """Return a compact JSON-only rule suggestion prompt."""

        context = self._build_context(metadata, semantic_report)
        context_json = json.dumps(context, ensure_ascii=False, indent=2)
        template = self._load_template()
        return template.replace("{{CONTEXT_JSON}}", context_json).strip()

    def _build_context(
        self,
        metadata: DatasetMetadata,
        semantic_report: SemanticDetectionReport,
    ) -> dict[str, Any]:
        semantic_by_column = {
            tag.column_name: tag
            for tag in semantic_report.columns
        }
        return {
            "dataset": {
                "source_file": metadata.source_file,
                "file_format": metadata.file_format,
                "row_count": metadata.row_count,
                "column_count": metadata.column_count,
                "duplicate_count": metadata.duplicate_count,
            },
            "columns": [
                self._compact_column(column, semantic_by_column.get(column.name))
                for column in metadata.columns
            ],
        }

    def _compact_column(
        self,
        column: ColumnProfile,
        semantic_tag: SemanticTag | None,
    ) -> dict[str, Any]:
        return {
            "name": column.name,
            "data_type": column.data_type,
            "null_percentage": column.null_percentage,
            "unique_value_count": column.unique_value_count,
            "semantic_type": semantic_tag.semantic_type if semantic_tag else "UNKNOWN",
            "semantic_confidence": semantic_tag.confidence if semantic_tag else 0.0,
            "top_values": [
                {
                    "value": self._compact_value(top_value.value),
                    "count": top_value.count,
                }
                for top_value in column.top_values[: self.max_top_values]
            ],
            "sample_values": [
                self._compact_value(value)
                for value in column.sample_values[: self.max_sample_values]
            ],
        }

    def _compact_value(self, value: Any) -> Any:
        if isinstance(value, str) and len(value) > self.max_value_length:
            return value[: self.max_value_length].rstrip() + "..."
        return value

    def _load_template(self) -> str:
        try:
            return self.template_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Prompt template not found: {self.template_path}") from exc
