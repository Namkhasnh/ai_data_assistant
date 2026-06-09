from __future__ import annotations

from pathlib import Path

from core.profiling.dataset_profiler import DatasetProfiler
from models.dataset import DatasetMetadata
from storage.artifact_store import ArtifactStore


class ProfilingService:
    """Thin service layer for dataset profiling workflows."""

    def __init__(
        self,
        profiler: DatasetProfiler | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self.profiler = profiler or DatasetProfiler()
        self.artifact_store = artifact_store or ArtifactStore()

    def profile_dataset(
        self,
        file_path: str | Path,
        output_path: str | Path | None = None,
        write_output: bool = True,
    ) -> DatasetMetadata:
        metadata = self.profiler.profile(file_path=file_path)
        if write_output:
            if output_path is not None:
                self.artifact_store.write_json_to_path(output_path, metadata)
            else:
                self.artifact_store.write_json("metadata.json", metadata)
        return metadata
