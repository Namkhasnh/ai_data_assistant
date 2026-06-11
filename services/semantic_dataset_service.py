from __future__ import annotations

from datetime import datetime

from models.knowledge_package import KnowledgePackageResult
from models.semantic_dataset import SemanticDataset, SemanticDatasetReport
from storage.artifact_store import ArtifactStore


class SemanticDatasetService:
    """Persist semantic enrichment results as first-class dataset artifacts."""

    def __init__(self, artifact_store: ArtifactStore | None = None) -> None:
        self.artifact_store = artifact_store or ArtifactStore()

    def generate(
        self,
        knowledge_package_result: KnowledgePackageResult,
        generated_at: datetime | None = None,
        dataset_filename: str = "semantic_dataset.csv",
        report_filename: str = "semantic_dataset_report.json",
    ) -> SemanticDataset:
        """Persist a semantic dataset from an existing KnowledgePackageResult."""

        if generated_at is None:
            generated_at = datetime.utcnow()

        source_dataframe = knowledge_package_result.dataframe.copy(deep=True)
        semantic_columns = self._semantic_columns(knowledge_package_result)
        source_columns = [
            column for column in source_dataframe.columns if column not in semantic_columns
        ]
        ordered_columns = [*source_columns, *semantic_columns]
        semantic_dataframe = source_dataframe.loc[:, ordered_columns].copy(deep=True)

        report = SemanticDatasetReport(
            generated_at=generated_at,
            source_column_count=len(source_columns),
            semantic_column_count=len(semantic_columns),
            total_columns=len(ordered_columns),
            source_columns=[str(column) for column in source_columns],
            semantic_columns=[str(column) for column in semantic_columns],
            warnings=list(knowledge_package_result.report.warnings),
        )
        self.artifact_store.write_dataframe_csv(dataset_filename, semantic_dataframe)
        self.artifact_store.write_json(report_filename, report)
        return SemanticDataset(dataframe=semantic_dataframe, report=report)

    def _semantic_columns(self, knowledge_package_result: KnowledgePackageResult) -> list[str]:
        dataframe_columns = set(knowledge_package_result.dataframe.columns)
        seen: set[str] = set()
        semantic_columns: list[str] = []
        for column in knowledge_package_result.report.produced_columns:
            if column in dataframe_columns and column not in seen:
                semantic_columns.append(column)
                seen.add(column)
        return semantic_columns
