from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from core.rules.rule_engine import RuleEngine
from models.rule import ExecutionReport, RuleSet
from storage.artifact_store import ArtifactStore


class RuleService:
    """Service layer for loading rules, executing them, and writing artifacts."""

    def __init__(
        self,
        engine: RuleEngine | None = None,
        artifact_store: ArtifactStore | None = None,
    ) -> None:
        self.engine = engine or RuleEngine()
        self.artifact_store = artifact_store or ArtifactStore()

    def load_rules(self, rules_path: str | Path) -> RuleSet:
        path = Path(rules_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Rules file not found: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Rules file is invalid JSON: {path}") from exc
        return RuleSet.model_validate(payload)

    def save_rules(self, rule_set: RuleSet, rules_path: str | Path) -> Path:
        """Persist a reviewed rule set to rules.json."""

        output_path = Path(rules_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(rule_set.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return output_path

    def execute_rules(
        self,
        df: pd.DataFrame,
        rules_path: str | Path,
        cleaned_filename: str = "cleaned_dataset.csv",
        report_filename: str = "rule_execution_report.json",
    ) -> tuple[pd.DataFrame, ExecutionReport]:
        rule_set = self.load_rules(rules_path)
        cleaned_dataframe, report = self.engine.execute(df, rule_set)
        self.artifact_store.write_dataframe_csv(cleaned_filename, cleaned_dataframe)
        self.artifact_store.write_json(report_filename, report)
        return cleaned_dataframe, report

    def execute_rules_for_csv(
        self,
        input_csv_path: str | Path,
        rules_path: str | Path,
        cleaned_filename: str = "cleaned_dataset.csv",
        report_filename: str = "rule_execution_report.json",
    ) -> tuple[pd.DataFrame, ExecutionReport]:
        dataframe = pd.read_csv(input_csv_path)
        return self.execute_rules(
            df=dataframe,
            rules_path=rules_path,
            cleaned_filename=cleaned_filename,
            report_filename=report_filename,
        )
