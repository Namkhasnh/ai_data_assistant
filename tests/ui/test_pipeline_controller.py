from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from app.controllers.pipeline_controller import PipelineController
from app.controllers.session_controller import SessionController
from app.controllers.workspace_controller import WorkspaceController
from models.dataset import DatasetMetadata
from models.rule import Rule, RuleSet
from models.rule_generation import RuleGenerationReport
from models.semantic_tag import SemanticDetectionReport


def test_pipeline_controller_orchestrates_existing_services(tmp_path) -> None:
    sample_path = tmp_path / "sample.csv"
    sample_path.write_text("title\nData Engineer\n", encoding="utf-8")
    session = SessionController({})
    session.set("uploaded_file_path", str(sample_path))
    workspace = WorkspaceController(session=session, workspace_root=tmp_path / "runs")
    profiling = FakeProfilingService()
    semantic = FakeSemanticService()
    rule_generation = FakeRuleGenerationService()

    controller = PipelineController(
        session=session,
        workspace=workspace,
        profiling_service=profiling,
        semantic_service=semantic,
        rule_generation_service=rule_generation,
    )

    metadata = controller.run_profile()
    semantic_report = controller.run_semantic()
    rule_set = controller.run_rule_generation()

    assert profiling.called is True
    assert semantic.called is True
    assert rule_generation.called is True
    assert session.get("metadata") == metadata
    assert session.get("semantic_report") == semantic_report
    assert session.get("rules") == rule_set
    assert workspace.rules_path.exists()


def test_pipeline_controller_saves_draft_rules_only_on_explicit_call(tmp_path) -> None:
    session = SessionController({})
    workspace = WorkspaceController(session=session, workspace_root=tmp_path)
    rule_service = FakeRuleService()
    controller = PipelineController(
        session=session,
        workspace=workspace,
        rule_service=rule_service,
    )
    rule_set = RuleSet(
        rules=[
            Rule(
                id="rule_001",
                type="transformation",
                column="title",
                parameters={"operation": "trim"},
                enabled=False,
            )
        ]
    )

    session.set_draft_rules(rule_set)
    assert not workspace.rules_path.exists()

    saved = controller.save_draft_rules()

    assert saved == rule_set
    assert workspace.rules_path.exists()
    assert rule_service.saved_path == workspace.rules_path
    assert session.get("rules_dirty") is False


def test_pages_do_not_import_backend_services_directly() -> None:
    for page_path in Path("app/pages").glob("*.py"):
        source = page_path.read_text(encoding="utf-8")
        assert "from services." not in source
        assert "import services." not in source


class FakeProfilingService:
    def __init__(self) -> None:
        self.called = False

    def profile_dataset(self, file_path, output_path):
        self.called = True
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")
        return DatasetMetadata(
            source_file=str(file_path),
            file_format="csv",
            row_count=1,
            column_count=1,
            duplicate_count=0,
            columns=[],
        )


class FakeSemanticService:
    def __init__(self) -> None:
        self.called = False

    def detect_columns(self, metadata, output_path):
        self.called = True
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("{}", encoding="utf-8")
        return SemanticDetectionReport(
            source_file=metadata.source_file,
            column_count=0,
            columns=[],
        )


class FakeRuleGenerationService:
    def __init__(self) -> None:
        self.called = False

    def generate_from_artifacts(
        self,
        metadata_path,
        semantic_report_path,
        rules_output_path,
        report_filename,
    ):
        self.called = True
        rule_set = RuleSet(
            rules=[
                Rule(
                    id="rule_001",
                    type="transformation",
                    column="title",
                    parameters={"operation": "trim"},
                )
            ]
        )
        rules_output_path.parent.mkdir(parents=True, exist_ok=True)
        rules_output_path.write_text(rule_set.model_dump_json(), encoding="utf-8")
        return rule_set, RuleGenerationReport(
            total_generated_rules=1,
            generated_by_generator={"fake": 1},
            warnings=[],
        )


class FakeRuleService:
    def __init__(self) -> None:
        self.saved_path: Path | None = None

    def save_rules(self, rule_set: RuleSet, rules_path: Path) -> Path:
        self.saved_path = rules_path
        rules_path.parent.mkdir(parents=True, exist_ok=True)
        rules_path.write_text(rule_set.model_dump_json(), encoding="utf-8")
        return rules_path
