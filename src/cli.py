"""CLI entry point for DICOM PHI scanning."""

import argparse
import logging
import sys
from pathlib import Path

from .agent import run_agent, run_direct_scan
from .models import BatchReport, FileError, ScanReport


def main():
    parser = argparse.ArgumentParser(
        description="DICOM PHI Screening Agent — scan DICOM files for protected health information"
    )
    parser.add_argument("filepath", nargs="?", default=None, help="Path to a single DICOM file")
    parser.add_argument("--dir", dest="directory", help="Recursively scan directory for .dcm files")
    parser.add_argument(
        "--mode",
        choices=["agent", "direct"],
        default="agent",
        help="Scan mode: 'agent' uses Claude orchestration, 'direct' runs scans sequentially (default: agent)",
    )
    parser.add_argument(
        "--output",
        choices=["json", "summary"],
        default="summary",
        help="Output format (default: summary)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.filepath and args.directory:
        print("Error: Cannot specify both a file path and --dir", file=sys.stderr)
        sys.exit(1)
    if not args.filepath and not args.directory:
        print("Error: Must specify either a file path or --dir", file=sys.stderr)
        sys.exit(1)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.directory:
        batch = _run_batch(args.directory, args.mode)
        if args.output == "json":
            print(batch.model_dump_json(indent=2))
        else:
            _print_batch_summary(batch)
    else:
        try:
            if args.mode == "agent":
                report = run_agent(args.filepath)
            else:
                report = run_direct_scan(args.filepath)
        except FileNotFoundError:
            print(f"Error: File not found: {args.filepath}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        if args.output == "json":
            print(report.model_dump_json(indent=2))
        else:
            _print_summary(report)


def _discover_dcm_files(directory: str) -> list[str]:
    """Recursively find all .dcm files in a directory."""
    dirpath = Path(directory)
    if not dirpath.is_dir():
        print(f"Error: Not a directory: {directory}", file=sys.stderr)
        sys.exit(1)

    files = sorted(
        str(p) for p in dirpath.rglob("*") if p.suffix.lower() == ".dcm" and p.is_file()
    )

    if not files:
        print(f"Error: No .dcm files found in {directory}", file=sys.stderr)
        sys.exit(1)

    return files


def _run_batch(directory: str, mode: str) -> BatchReport:
    """Scan all .dcm files in a directory and return an aggregate report."""
    files = _discover_dcm_files(directory)
    total = len(files)

    reports: list[ScanReport] = []
    errors: list[FileError] = []

    import anthropic

    client = anthropic.Anthropic()

    for i, filepath in enumerate(files, 1):
        try:
            if mode == "agent":
                report = run_agent(filepath, client=client)
            else:
                report = run_direct_scan(filepath, client=client)
            reports.append(report)
            risk = report.risk_level.value.upper()
            print(f"[{i}/{total}] Scanning {filepath} ... {risk} ({report.total_phi_count} PHI findings)")
        except Exception as e:
            errors.append(FileError(filepath=filepath, error=str(e)))
            print(f"[{i}/{total}] Scanning {filepath} ... ERROR: {e}")

    files_with_phi = sum(1 for r in reports if r.has_phi)
    risk_breakdown = {"high": 0, "medium": 0, "low": 0}
    for r in reports:
        risk_breakdown[r.risk_level.value] += 1

    return BatchReport(
        directory=directory,
        total_files=total,
        files_with_phi=files_with_phi,
        files_clean=len(reports) - files_with_phi,
        files_errored=len(errors),
        risk_breakdown=risk_breakdown,
        reports=reports,
        errors=errors,
    )


def _print_batch_summary(batch: BatchReport):
    """Print a human-readable summary of a batch scan."""
    print(f"\n{'='*60}")
    print("DICOM PHI Batch Scan Summary")
    print(f"{'='*60}")
    print(f"Directory: {batch.directory}")
    print(f"Total files scanned: {batch.total_files}")
    print(f"Files with PHI: {batch.files_with_phi}")
    print(f"Files clean: {batch.files_clean}")
    print(f"Files errored: {batch.files_errored}")
    print()
    print("Risk Breakdown:")
    print(f"  HIGH:   {batch.risk_breakdown.get('high', 0)}")
    print(f"  MEDIUM: {batch.risk_breakdown.get('medium', 0)}")
    print(f"  LOW:    {batch.risk_breakdown.get('low', 0)}")

    if batch.errors:
        print()
        print("Errors:")
        for err in batch.errors:
            print(f"  {err.filepath}: {err.error}")

    print(f"{'='*60}\n")


def _print_summary(report):
    """Print a human-readable summary of the scan report."""
    risk_colors = {"high": "RED", "medium": "YELLOW", "low": "GREEN"}
    risk = report.risk_level.value

    print(f"\n{'='*60}")
    print("DICOM PHI Scan Report")
    print(f"{'='*60}")
    print(f"File: {report.filepath}")
    print(f"Risk Level: [{risk_colors[risk]}] {risk.upper()}")
    print(f"Total PHI Findings: {report.total_phi_count}")
    print()

    if report.tag_findings:
        print(f"--- Header Tag Findings ({len(report.tag_findings)}) ---")
        for f in report.tag_findings:
            print(f"  [{f.severity.value.upper()}] {f.tag} {f.tag_name}: {f.value}")
        print()

    if report.pixel_findings:
        print(f"--- Pixel PHI Findings ({len(report.pixel_findings)}) ---")
        for f in report.pixel_findings:
            print(
                f"  [{f.severity.value.upper()}] \"{f.text}\" ({f.phi_type})"
                f" at ({f.bbox.x},{f.bbox.y}) {f.bbox.width}x{f.bbox.height}"
            )
        print()

    bia = report.burned_in_annotation_value or "NOT PRESENT"
    print(f"BurnedInAnnotation (0028,0301): {bia}")
    print()

    print("Recommendations:")
    for r in report.recommendations:
        print(f"  - {r}")
    print(f"{'='*60}\n")
