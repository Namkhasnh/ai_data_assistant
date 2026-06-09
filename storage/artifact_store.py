from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class ArtifactStore:
    """Filesystem store for generated project artifacts."""

    def __init__(self, artifact_dir: str | Path = "storage/artifacts") -> None:
        self.artifact_dir = Path(artifact_dir)

    def artifact_path(self, filename: str) -> Path:
        return self.artifact_dir / filename

    def write_json(self, filename: str, payload: BaseModel | dict[str, Any]) -> Path:
        output_path = self.artifact_path(filename)
        return self.write_json_to_path(output_path=output_path, payload=payload)

    def write_json_to_path(
        self,
        output_path: str | Path,
        payload: BaseModel | dict[str, Any],
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(payload, BaseModel):
            serializable_payload: dict[str, Any] = payload.model_dump(mode="json")
        else:
            serializable_payload = payload

        output_path.write_text(
            json.dumps(serializable_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote artifact to %s", output_path)
        return output_path

    def write_dataframe_csv(self, filename: str, dataframe: pd.DataFrame) -> Path:
        output_path = self.artifact_path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_path, index=False)
        logger.info("Wrote dataframe artifact to %s", output_path)
        return output_path
