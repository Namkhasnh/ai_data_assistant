from __future__ import annotations

from pathlib import Path

import pandas as pd

from audit.audit_report import AuditReportStore
from audit.change_log import ArtifactSpec, ChangeLogStore
from audit.rule_history import RuleHistoryStore
from app.controllers.session_controller import SessionController
from app.controllers.workspace_controller import WorkspaceController
from models.rule import RuleSet
from services.ai_rule_generation_service import AIRuleGenerationService
from services.enrichment_service import EnrichmentService
from services.export_service import ExportService
from services.profiling_service import ProfilingService
from services.report_service import ReportService
from services.rule_generation_service import RuleGenerationService
from services.rule_service import RuleService
from services.semantic_service import SemanticService
from services.standardization_service import StandardizationService
from storage.artifact_store import ArtifactStore


class PipelineController:
    """Coordinate existing backend services for Streamlit pages."""

    def __init__(
        self,
        session: SessionController,
        workspace: WorkspaceController,
        profiling_service: ProfilingService | None = None,
        semantic_service: SemanticService | None = None,
        rule_generation_service: RuleGenerationService | None = None,
        rule_service: RuleService | None = None,
        standardization_service: StandardizationService | None = None,
        enrichment_service: EnrichmentService | None = None,
        ai_rule_generation_service: AIRuleGenerationService | None = None,
    ) -> None:
        self.session = session
        self.workspace = workspace
        self.artifact_store = ArtifactStore(artifact_dir=self.workspace.artifact_dir)
        self.profiling_service = profiling_service or ProfilingService(
            artifact_store=self.artifact_store
        )
        self.semantic_service = semantic_service or SemanticService(artifact_store=self.artifact_store)
        self.rule_generation_service = rule_generation_service or RuleGenerationService(
            artifact_store=self.artifact_store
        )
        self.rule_service = rule_service or RuleService(artifact_store=self.artifact_store)
        self.standardization_service = standardization_service or StandardizationService(
            artifact_store=self.artifact_store
        )
        self.enrichment_service = enrichment_service or EnrichmentService(
            artifact_store=self.artifact_store
        )
        self.ai_rule_generation_service = ai_rule_generation_service or AIRuleGenerationService(
            artifact_store=self.artifact_store
        )

    def run_profile(self) -> object:
        uploaded_path = self._uploaded_path()
        metadata = self.profiling_service.profile_dataset(
            file_path=uploaded_path,
            output_path=self.workspace.artifact_dir / "metadata.json",
        )
        self.session.set_metadata(metadata)
        return metadata

    def run_semantic(self) -> object:
        metadata = self.session.get("metadata")
        if metadata is None:
            metadata = self.run_profile()
        semantic_report = self.semantic_service.detect_columns(
            metadata=metadata,
            output_path=self.workspace.artifact_dir / "semantic_columns.json",
        )
        self.session.set_semantic_report(semantic_report)
        return semantic_report

    def run_rule_generation(self) -> RuleSet:
        if self.session.get("semantic_report") is None:
            self.run_semantic()
        rule_set, report = self.rule_generation_service.generate_from_artifacts(
            metadata_path=self.workspace.artifact_dir / "metadata.json",
            semantic_report_path=self.workspace.artifact_dir / "semantic_columns.json",
            rules_output_path=self.workspace.rules_path,
            report_filename="rule_generation_report.json",
        )
        self.session.set_rules(rule_set)
        if report.warnings:
            for warning in report.warnings:
                self.session.append_warning(warning)
        return rule_set

    def save_draft_rules(self) -> RuleSet:
        draft_rules = self.session.get("draft_rules")
        if draft_rules is None:
            raise ValueError("No draft rules available to save.")
        rule_set = draft_rules if isinstance(draft_rules, RuleSet) else RuleSet.model_validate(draft_rules)
        self.rule_service.save_rules(rule_set, self.workspace.rules_path)
        self.session.mark_rules_saved()
        return rule_set

    def run_rule_execution(self) -> tuple[pd.DataFrame, object]:
        if not self.workspace.rules_path.exists():
            self.run_rule_generation()
        dataframe = self.workspace.load_dataframe(self._uploaded_path())
        cleaned_dataframe, report = self.rule_service.execute_rules(
            df=dataframe,
            rules_path=self.workspace.rules_path,
            cleaned_filename="cleaned_dataset.csv",
            report_filename="rule_execution_report.json",
        )
        return cleaned_dataframe, report

    def run_standardization(self) -> tuple[pd.DataFrame, object]:
        cleaned_path = self.workspace.artifact_dir / "cleaned_dataset.csv"
        if not cleaned_path.exists():
            self.run_rule_execution()
        standardized, report = self.standardization_service.standardize_csv(
            input_csv_path=cleaned_path,
            semantic_report_path=self.workspace.artifact_dir / "semantic_columns.json",
            config_path=Path("knowledge/domains/jobs/standardization_rules.json"),
            standardized_filename="standardized_dataset.csv",
            report_filename="standardization_report.json",
        )
        self.session.set_standardized_dataframe(standardized)
        for warning in report.warnings:
            self.session.append_warning(warning)
        return standardized, report

    def run_enrichment(self) -> tuple[pd.DataFrame, object]:
        standardized_path = self.workspace.artifact_dir / "standardized_dataset.csv"
        if not standardized_path.exists():
            self.run_standardization()
        enriched, report = self.enrichment_service.enrich_csv(
            input_csv_path=standardized_path,
            semantic_report_path=self.workspace.artifact_dir / "semantic_columns.json",
            config_path=Path("knowledge/domains/jobs/enrichment_rules.json"),
            enriched_filename="enriched_dataset.csv",
            report_filename="enrichment_report.json",
        )
        self.session.set_enriched_dataframe(enriched)
        for warning in report.warnings:
            self.session.append_warning(warning)
        return enriched, report

    def run_audit(self) -> object:
        artifact_specs = self._artifact_specs()
        rule_history_path = self.workspace.audit_dir / "rule_history.json"
        change_log_path = self.workspace.audit_dir / "change_log.json"
        RuleHistoryStore(audit_path=rule_history_path).build_from_files(
            rules_path=self.workspace.rules_path,
            execution_report_path=self.workspace.artifact_dir / "rule_execution_report.json",
            merge_existing=False,
        )
        ChangeLogStore(audit_path=change_log_path).build_from_artifacts(
            artifact_specs=artifact_specs
        )
        audit_report, _path = AuditReportStore(
            audit_path=self.workspace.audit_dir / "audit_report.json",
            change_log_path=change_log_path,
            rule_history_path=rule_history_path,
        ).persist(artifact_specs=artifact_specs)
        for warning in audit_report.warnings:
            self.session.append_warning(warning)
        return audit_report

    def run_report(self) -> object:
        report, _path = ReportService(
            artifact_dir=self.workspace.artifact_dir,
            audit_dir=self.workspace.audit_dir,
            rules_path=self.workspace.rules_path,
            report_dir=self.workspace.report_dir,
        ).generate_report()
        for warning in report.warnings:
            self.session.append_warning(warning)
        return report

    def run_export(self) -> object:
        export_report = ExportService(
            artifact_dir=self.workspace.artifact_dir,
            audit_dir=self.workspace.audit_dir,
            report_dir=self.workspace.report_dir,
            export_dir=self.workspace.export_dir,
        ).export_all()
        for warning in export_report.warnings:
            self.session.append_warning(warning)
        return export_report

    def run_pipeline(self) -> object:
        self.run_profile()
        self.run_semantic()
        self.run_rule_generation()
        self.run_rule_execution()
        self.run_standardization()
        self.run_enrichment()
        self.run_audit()
        self.run_report()
        return self.run_export()

    def generate_ai_suggestions(self) -> object:
        report = self.ai_rule_generation_service.generate_suggestions(
            metadata_path=self.workspace.artifact_dir / "metadata.json",
            semantic_report_path=self.workspace.artifact_dir / "semantic_columns.json",
            output_filename="ai_suggestions.json",
        )
        for warning in report.warnings:
            self.session.append_warning(warning)
        return report

    def _uploaded_path(self) -> Path:
        uploaded_path = self.session.get("uploaded_file_path")
        if not uploaded_path:
            raise ValueError("No uploaded dataset is available.")
        return Path(uploaded_path)

    def _artifact_specs(self) -> tuple[ArtifactSpec, ...]:
        artifact_dir = self.workspace.artifact_dir
        return (
            ArtifactSpec("profiling", "metadata.json", artifact_dir / "metadata.json"),
            ArtifactSpec("semantic", "semantic_columns.json", artifact_dir / "semantic_columns.json"),
            ArtifactSpec("rule_generation", "rules.json", self.workspace.rules_path),
            ArtifactSpec("rule_generation", "rule_generation_report.json", artifact_dir / "rule_generation_report.json"),
            ArtifactSpec("rule_execution", "rule_execution_report.json", artifact_dir / "rule_execution_report.json"),
            ArtifactSpec("rule_execution", "cleaned_dataset.csv", artifact_dir / "cleaned_dataset.csv"),
            ArtifactSpec("standardization", "standardization_report.json", artifact_dir / "standardization_report.json"),
            ArtifactSpec("standardization", "standardized_dataset.csv", artifact_dir / "standardized_dataset.csv"),
            ArtifactSpec("enrichment", "enrichment_report.json", artifact_dir / "enrichment_report.json"),
            ArtifactSpec("enrichment", "enriched_dataset.csv", artifact_dir / "enriched_dataset.csv"),
            ArtifactSpec("ai_suggestions", "ai_suggestions.json", artifact_dir / "ai_suggestions.json"),
        )
