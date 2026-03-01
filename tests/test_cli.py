"""Tests for CLI output formatting."""

import json

from src.cli import _print_batch_summary, _print_summary
from src.models import (
    BatchReport,
    BoundingBox,
    FileError,
    PHITagFinding,
    PixelPHIFinding,
    ScanReport,
    Severity,
)


def _make_report(**overrides) -> ScanReport:
    defaults = dict(
        filepath="/tmp/test.dcm",
        tag_findings=[],
        pixel_findings=[],
        burned_in_annotation_tag_present=True,
        burned_in_annotation_value="NO",
        total_phi_count=0,
        risk_level=Severity.LOW,
        recommendations=["No PHI detected — file appears safe for sharing"],
    )
    defaults.update(overrides)
    return ScanReport(**defaults)


def _make_batch_report(**overrides) -> BatchReport:
    defaults = dict(
        directory="/tmp/dicoms",
        total_files=0,
        files_with_phi=0,
        files_clean=0,
        files_errored=0,
        risk_breakdown={"high": 0, "medium": 0, "low": 0},
        reports=[],
        errors=[],
    )
    defaults.update(overrides)
    return BatchReport(**defaults)


# --- Single-file summary tests (existing) ---


def test_print_summary_clean(capsys):
    report = _make_report()
    _print_summary(report)
    output = capsys.readouterr().out
    assert "No PHI detected" in output
    assert "LOW" in output


def test_print_summary_with_tag_findings(capsys):
    findings = [
        PHITagFinding(
            tag="(0010,0010)",
            tag_name="PatientName",
            value="DOE^JANE",
            severity=Severity.HIGH,
            hipaa_category="Name",
        ),
    ]
    report = _make_report(
        tag_findings=findings,
        total_phi_count=1,
        risk_level=Severity.HIGH,
        recommendations=["Remove or redact PHI from DICOM header tags before sharing"],
    )
    _print_summary(report)
    output = capsys.readouterr().out
    assert "HIGH" in output
    assert "PatientName" in output
    assert "DOE^JANE" in output


def test_print_summary_with_pixel_findings(capsys):
    findings = [
        PixelPHIFinding(
            text="SMITH, JOHN",
            bbox=BoundingBox(x=10, y=20, width=100, height=15),
            phi_type="patient_name",
            confidence=0.95,
            severity=Severity.HIGH,
        ),
    ]
    report = _make_report(
        pixel_findings=findings,
        total_phi_count=1,
        risk_level=Severity.HIGH,
        recommendations=["Redact burned-in PHI text"],
    )
    _print_summary(report)
    output = capsys.readouterr().out
    assert "SMITH, JOHN" in output
    assert "patient_name" in output
    assert "100x15" in output


# --- Batch summary tests ---


def test_print_batch_summary_clean(capsys):
    reports = [_make_report(filepath=f"/tmp/{i}.dcm") for i in range(3)]
    batch = _make_batch_report(
        total_files=3,
        files_with_phi=0,
        files_clean=3,
        risk_breakdown={"high": 0, "medium": 0, "low": 3},
        reports=reports,
    )
    _print_batch_summary(batch)
    output = capsys.readouterr().out
    assert "Files with PHI: 0" in output
    assert "Files clean: 3" in output
    assert "Total files scanned: 3" in output


def test_print_batch_summary_mixed(capsys):
    high_report = _make_report(
        filepath="/tmp/a.dcm",
        risk_level=Severity.HIGH,
        total_phi_count=3,
        tag_findings=[
            PHITagFinding(
                tag="(0010,0010)",
                tag_name="PatientName",
                value="DOE^JANE",
                severity=Severity.HIGH,
                hipaa_category="Name",
            ),
        ],
    )
    medium_report = _make_report(
        filepath="/tmp/b.dcm",
        risk_level=Severity.MEDIUM,
        total_phi_count=1,
    )
    low_report = _make_report(filepath="/tmp/c.dcm")
    batch = _make_batch_report(
        total_files=3,
        files_with_phi=2,
        files_clean=1,
        risk_breakdown={"high": 1, "medium": 1, "low": 1},
        reports=[high_report, medium_report, low_report],
    )
    _print_batch_summary(batch)
    output = capsys.readouterr().out
    assert "Files with PHI: 2" in output
    assert "HIGH:   1" in output
    assert "MEDIUM: 1" in output
    assert "LOW:    1" in output


def test_print_batch_summary_with_errors(capsys):
    batch = _make_batch_report(
        total_files=2,
        files_errored=1,
        files_clean=1,
        reports=[_make_report()],
        errors=[FileError(filepath="/tmp/bad.dcm", error="Invalid DICOM file")],
    )
    _print_batch_summary(batch)
    output = capsys.readouterr().out
    assert "Files errored: 1" in output
    assert "/tmp/bad.dcm" in output
    assert "Invalid DICOM file" in output


def test_batch_json_output():
    batch = _make_batch_report(
        total_files=1,
        files_clean=1,
        risk_breakdown={"high": 0, "medium": 0, "low": 1},
        reports=[_make_report()],
    )
    data = json.loads(batch.model_dump_json())
    assert data["directory"] == "/tmp/dicoms"
    assert data["total_files"] == 1
    assert data["files_clean"] == 1
    assert "reports" in data
    assert "errors" in data
    assert data["risk_breakdown"] == {"high": 0, "medium": 0, "low": 1}
