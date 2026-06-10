from __future__ import annotations

from pathlib import Path

from tests.cross_domain_pipeline_helper import (
    assert_cross_domain_outputs,
    run_cross_domain_pipeline,
)


def test_crm_pipeline_cross_domain_validation(tmp_path: Path) -> None:
    run_root = tmp_path / "validation_runs"

    record = run_cross_domain_pipeline(
        domain="crm",
        sample_path=Path("data/samples/crm/crm_sample.csv"),
        run_root=run_root,
        standardization_config_path=Path("knowledge/domains/crm/standardization_rules.json"),
        enrichment_config_path=Path("knowledge/domains/crm/enrichment_rules.json"),
    )

    assert record.status == "PASS_WITH_WARNINGS"
    assert record.column_leakage_detected is False
    assert record.unexpected_column_loss is False
    assert "knowledge/domains/crm/standardization_rules.json" in record.missing_knowledge
    assert "knowledge/domains/crm/enrichment_rules.json" in record.missing_knowledge
    assert_cross_domain_outputs("crm", run_root, expect_job_columns=False)
