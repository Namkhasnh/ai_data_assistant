from __future__ import annotations

from pathlib import Path

from tests.cross_domain_pipeline_helper import (
    assert_cross_domain_outputs,
    run_cross_domain_pipeline,
)


def test_jobs_pipeline_cross_domain_validation(tmp_path: Path) -> None:
    run_root = tmp_path / "validation_runs"

    record = run_cross_domain_pipeline(
        domain="jobs",
        sample_path=Path("data/samples/jobs/jobs_sample.csv"),
        run_root=run_root,
        standardization_config_path=Path("knowledge/domains/jobs/standardization_rules.json"),
        enrichment_config_path=Path("knowledge/domains/jobs/enrichment_rules.json"),
    )

    assert record.status in {"PASS", "PASS_WITH_WARNINGS"}
    assert record.column_leakage_detected is False
    assert record.unexpected_column_loss is False
    assert_cross_domain_outputs("jobs", run_root, expect_job_columns=True)
