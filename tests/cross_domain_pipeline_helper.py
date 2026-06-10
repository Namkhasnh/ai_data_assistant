from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from audit.audit_report import AuditReportStore
from audit.change_log import ArtifactSpec, ChangeLogStore
from audit.rule_history import RuleHistoryStore
from models.cross_domain_validation import CrossDomainValidationRecord, ValidationStatus
from models.enrichment import EnrichmentConfig
from models.standardization import StandardizationReport
from services.enrichment_service import EnrichmentService
from services.export_service import ExportService
from services.profiling_service import ProfilingService
from services.report_service import ReportService
from services.rule_generation_service import RuleGenerationService
from services.rule_service import RuleService
from services.semantic_service import SemanticService
from services.standardization_service import StandardizationService
from storage.artifact_store import ArtifactStore


JOB_COLUMNS = {"job_family", "job_domain"}
EXPECTED_ENRICHMENT_COLUMNS = {"job_family", "job_domain", "region", "country"}


def run_cross_domain_pipeline(
    domain: str,
    sample_path: Path,
    run_root: Path,
    standardization_config_path: Path | None = None,
    enrichment_config_path: Path | None = None,
) -> CrossDomainValidationRecord:
    """Run the existing pipeline for one domain in isolated directories."""

    domain_root = run_root / domain
    artifact_dir = domain_root / "artifacts"
    audit_dir = domain_root / "audit"
    rules_dir = domain_root / "rules"
    report_dir = domain_root / "reports"
    export_dir = domain_root / "exports"
    warnings: list[str] = []
    missing_knowledge: list[str] = []
    generated_artifacts: list[str] = []

    artifact_store = ArtifactStore(artifact_dir=artifact_dir)
    metadata_path = artifact_dir / "metadata.json"
    semantic_path = artifact_dir / "semantic_columns.json"
    rules_path = rules_dir / "rules.json"

    try:
        metadata = ProfilingService(artifact_store=artifact_store).profile_dataset(
            file_path=sample_path,
            output_path=metadata_path,
        )
        generated_artifacts.append(str(metadata_path))

        semantic_report = SemanticService(artifact_store=artifact_store).detect_columns(
            metadata=metadata,
            output_path=semantic_path,
        )
        generated_artifacts.append(str(semantic_path))
        unknown_columns = [
            tag.column_name
            for tag in semantic_report.columns
            if tag.semantic_type == "UNKNOWN"
        ]
        if unknown_columns:
            warnings.append(f"UNKNOWN semantic columns: {unknown_columns}")

        rule_generation_store = ArtifactStore(artifact_dir=artifact_dir)
        _rule_set, _rule_generation_report = RuleGenerationService(
            artifact_store=rule_generation_store
        ).generate_from_artifacts(
            metadata_path=metadata_path,
            semantic_report_path=semantic_path,
            rules_output_path=rules_path,
            report_filename="rule_generation_report.json",
        )
        generated_artifacts.extend(
            [
                str(rules_path),
                str(artifact_dir / "rule_generation_report.json"),
            ]
        )

        raw_dataframe = pd.read_csv(sample_path)
        cleaned_dataframe, _execution_report = RuleService(
            artifact_store=artifact_store
        ).execute_rules(
            df=raw_dataframe,
            rules_path=rules_path,
            cleaned_filename="cleaned_dataset.csv",
            report_filename="rule_execution_report.json",
        )
        generated_artifacts.extend(
            [
                str(artifact_dir / "cleaned_dataset.csv"),
                str(artifact_dir / "rule_execution_report.json"),
            ]
        )

        standardized_dataframe, _standardization_report = _standardize_domain(
            dataframe=cleaned_dataframe,
            semantic_path=semantic_path,
            config_path=standardization_config_path,
            artifact_store=artifact_store,
            warnings=warnings,
            missing_knowledge=missing_knowledge,
        )
        generated_artifacts.extend(
            [
                str(artifact_dir / "standardized_dataset.csv"),
                str(artifact_dir / "standardization_report.json"),
            ]
        )

        enriched_dataframe, _enrichment_report = _enrich_domain(
            dataframe=standardized_dataframe,
            semantic_path=semantic_path,
            config_path=enrichment_config_path,
            artifact_store=artifact_store,
            warnings=warnings,
            missing_knowledge=missing_knowledge,
        )
        generated_artifacts.extend(
            [
                str(artifact_dir / "enriched_dataset.csv"),
                str(artifact_dir / "enrichment_report.json"),
            ]
        )

        rule_history_path = audit_dir / "rule_history.json"
        change_log_path = audit_dir / "change_log.json"
        RuleHistoryStore(audit_path=rule_history_path).build_from_files(
            rules_path=rules_path,
            execution_report_path=artifact_dir / "rule_execution_report.json",
            merge_existing=False,
        )
        artifact_specs = _artifact_specs(
            artifact_dir=artifact_dir,
            rules_path=rules_path,
        )
        ChangeLogStore(audit_path=change_log_path).build_from_artifacts(
            artifact_specs=artifact_specs,
        )
        audit_report, audit_report_path = AuditReportStore(
            audit_path=audit_dir / "audit_report.json",
            change_log_path=change_log_path,
            rule_history_path=rule_history_path,
        ).persist(artifact_specs=artifact_specs)
        generated_artifacts.extend(
            [
                str(rule_history_path),
                str(change_log_path),
                str(audit_report_path),
            ]
        )
        warnings.extend(audit_report.warnings)

        _report, report_path = ReportService(
            artifact_dir=artifact_dir,
            audit_dir=audit_dir,
            rules_path=rules_path,
            report_dir=report_dir,
        ).generate_report()
        generated_artifacts.append(str(report_path))

        export_report = ExportService(
            artifact_dir=artifact_dir,
            audit_dir=audit_dir,
            report_dir=report_dir,
            export_dir=export_dir,
        ).export_all()
        generated_artifacts.extend(
            [
                str(export_dir / "csv" / "export_dataset.csv"),
                str(export_dir / "xlsx" / "export_dataset.xlsx"),
                str(export_dir / "json" / "export_dataset.json"),
                str(export_dir / "export_report.json"),
            ]
        )
        warnings.extend(export_report.warnings)

        exported_dataframe = pd.read_csv(export_dir / "csv" / "export_dataset.csv")
        unexpected_column_loss = (
            _has_column_loss(raw_dataframe, cleaned_dataframe)
            or _has_column_loss(cleaned_dataframe, standardized_dataframe)
            or _has_column_loss(standardized_dataframe, enriched_dataframe)
            or list(exported_dataframe.columns) != list(enriched_dataframe.columns)
        )
        column_leakage_detected = domain != "jobs" and bool(JOB_COLUMNS & set(enriched_dataframe.columns))
        if domain == "jobs" and not JOB_COLUMNS.issubset(enriched_dataframe.columns):
            warnings.append("Jobs enrichment columns missing: ['job_family', 'job_domain']")
            column_leakage_detected = True

        status = _status(
            warnings=warnings,
            column_leakage_detected=column_leakage_detected,
            unexpected_column_loss=unexpected_column_loss,
        )
        return CrossDomainValidationRecord(
            domain=domain,
            status=status,
            warnings=_deduplicate(warnings),
            missing_knowledge=_deduplicate(missing_knowledge),
            generated_artifacts=sorted(path for path in generated_artifacts if Path(path).exists()),
            column_leakage_detected=column_leakage_detected,
            unexpected_column_loss=unexpected_column_loss,
        )
    except Exception as exc:  # noqa: BLE001 - validation should report failures.
        warnings.append(f"Pipeline failed: {exc}")
        return CrossDomainValidationRecord(
            domain=domain,
            status="FAIL",
            warnings=_deduplicate(warnings),
            missing_knowledge=_deduplicate(missing_knowledge),
            generated_artifacts=sorted(path for path in generated_artifacts if Path(path).exists()),
            column_leakage_detected=False,
            unexpected_column_loss=True,
        )


def assert_cross_domain_outputs(
    domain: str,
    run_root: Path,
    expect_job_columns: bool,
) -> None:
    """Assert acceptance criteria for one completed domain validation run."""

    domain_root = run_root / domain
    artifact_dir = domain_root / "artifacts"
    audit_dir = domain_root / "audit"
    report_dir = domain_root / "reports"
    export_dir = domain_root / "exports"

    assert (export_dir / "csv" / "export_dataset.csv").exists()
    assert (report_dir / "report.html").exists()
    assert (audit_dir / "audit_report.json").exists()

    cleaned = pd.read_csv(artifact_dir / "cleaned_dataset.csv")
    standardized = pd.read_csv(artifact_dir / "standardized_dataset.csv")
    enriched = pd.read_csv(artifact_dir / "enriched_dataset.csv")
    exported = pd.read_csv(export_dir / "csv" / "export_dataset.csv")

    assert not _has_column_loss(cleaned, standardized)
    assert not _has_column_loss(standardized, enriched)
    assert list(exported.columns) == list(enriched.columns)

    if expect_job_columns:
        assert JOB_COLUMNS.issubset(enriched.columns)
    else:
        assert JOB_COLUMNS.isdisjoint(enriched.columns)


def _standardize_domain(
    dataframe: pd.DataFrame,
    semantic_path: Path,
    config_path: Path | None,
    artifact_store: ArtifactStore,
    warnings: list[str],
    missing_knowledge: list[str],
) -> tuple[pd.DataFrame, StandardizationReport]:
    if config_path is None or not config_path.exists():
        warnings.append(f"Missing standardization config: {config_path}")
        if config_path is not None:
            missing_knowledge.append(str(config_path))
        report = StandardizationReport(
            total_standardized_columns=0,
            warnings=[f"Missing standardization config: {config_path}"],
            skipped_columns=list(dataframe.columns),
        )
        artifact_store.write_dataframe_csv("standardized_dataset.csv", dataframe.copy(deep=True))
        artifact_store.write_json("standardization_report.json", report)
        return dataframe.copy(deep=True), report
    if _is_empty_standardization_config(config_path):
        warning = f"No standardization rules configured: {config_path}"
        warnings.append(warning)
        missing_knowledge.append(str(config_path))

    try:
        standardized, report = StandardizationService(
            artifact_store=artifact_store
        ).standardize_dataframe(
            dataframe=dataframe,
            semantic_report_path=semantic_path,
            config_path=config_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        warnings.append(str(exc))
        missing_knowledge.append(str(config_path))
        report = StandardizationReport(
            total_standardized_columns=0,
            warnings=[str(exc)],
            skipped_columns=list(dataframe.columns),
        )
        standardized = dataframe.copy(deep=True)
        artifact_store.write_dataframe_csv("standardized_dataset.csv", standardized)
        artifact_store.write_json("standardization_report.json", report)
    else:
        warnings.extend(report.warnings)
        missing_knowledge.extend(_knowledge_warnings(report.warnings))
    return standardized, report


def _enrich_domain(
    dataframe: pd.DataFrame,
    semantic_path: Path,
    config_path: Path | None,
    artifact_store: ArtifactStore,
    warnings: list[str],
    missing_knowledge: list[str],
) -> tuple[pd.DataFrame, Any]:
    if config_path is None or not config_path.exists():
        warnings.append(f"Missing enrichment config: {config_path}")
        if config_path is not None:
            missing_knowledge.append(str(config_path))
        config = EnrichmentConfig()
        config_path = artifact_store.artifact_dir / "empty_enrichment_rules.json"
        artifact_store.write_json_to_path(config_path, config)
    elif _is_empty_enrichment_config(config_path):
        warning = f"No enrichment rules configured: {config_path}"
        warnings.append(warning)
        missing_knowledge.append(str(config_path))

    try:
        enriched, report = EnrichmentService(
            artifact_store=artifact_store
        ).enrich_dataframe(
            dataframe=dataframe,
            semantic_report_path=semantic_path,
            config_path=config_path,
        )
    except (FileNotFoundError, ValueError) as exc:
        warnings.append(str(exc))
        if config_path is not None:
            missing_knowledge.append(str(config_path))
        config_path = artifact_store.artifact_dir / "empty_enrichment_rules.json"
        artifact_store.write_json_to_path(config_path, EnrichmentConfig())
        enriched, report = EnrichmentService(
            artifact_store=artifact_store
        ).enrich_dataframe(
            dataframe=dataframe,
            semantic_report_path=semantic_path,
            config_path=config_path,
        )
    warnings.extend(report.warnings)
    missing_knowledge.extend(_knowledge_warnings(report.warnings))
    return enriched, report


def _artifact_specs(artifact_dir: Path, rules_path: Path) -> tuple[ArtifactSpec, ...]:
    return (
        ArtifactSpec("profiling", "metadata.json", artifact_dir / "metadata.json"),
        ArtifactSpec("semantic", "semantic_columns.json", artifact_dir / "semantic_columns.json"),
        ArtifactSpec("rule_generation", "rules.json", rules_path),
        ArtifactSpec("rule_generation", "rule_generation_report.json", artifact_dir / "rule_generation_report.json"),
        ArtifactSpec("rule_execution", "rule_execution_report.json", artifact_dir / "rule_execution_report.json"),
        ArtifactSpec("rule_execution", "cleaned_dataset.csv", artifact_dir / "cleaned_dataset.csv"),
        ArtifactSpec("standardization", "standardization_report.json", artifact_dir / "standardization_report.json"),
        ArtifactSpec("standardization", "standardized_dataset.csv", artifact_dir / "standardized_dataset.csv"),
        ArtifactSpec("enrichment", "enrichment_report.json", artifact_dir / "enrichment_report.json"),
        ArtifactSpec("enrichment", "enriched_dataset.csv", artifact_dir / "enriched_dataset.csv"),
    )


def _has_column_loss(before: pd.DataFrame, after: pd.DataFrame) -> bool:
    return any(column not in after.columns for column in before.columns)


def _status(
    warnings: list[str],
    column_leakage_detected: bool,
    unexpected_column_loss: bool,
) -> ValidationStatus:
    if column_leakage_detected or unexpected_column_loss:
        return "FAIL"
    if warnings:
        return "PASS_WITH_WARNINGS"
    return "PASS"


def _is_empty_standardization_config(config_path: Path) -> bool:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return payload == {}


def _is_empty_enrichment_config(config_path: Path) -> bool:
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return payload == {"enrichments": []}


def _knowledge_warnings(warnings: list[str]) -> list[str]:
    return [
        warning
        for warning in warnings
        if "Knowledge file not found" in warning
        or "Missing standardization config" in warning
        or "Missing enrichment config" in warning
        or "No standardization rules configured" in warning
        or "No enrichment rules configured" in warning
        or "Standardization input not found" in warning
        or "Enrichment input not found" in warning
    ]


def _deduplicate(values: list[str]) -> list[str]:
    return sorted(dict.fromkeys(values))
