from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from models.column_profile import ColumnProfile


class DatasetMetadata(BaseModel):
    """Dataset-level profiling metadata exported to metadata.json."""

    model_config = ConfigDict(extra="forbid")

    source_file: str = Field(min_length=1)
    file_format: str = Field(min_length=1)
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    duplicate_count: int = Field(ge=0)
    columns: list[ColumnProfile] = Field(default_factory=list)
