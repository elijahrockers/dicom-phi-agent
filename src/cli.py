"""CLI entry point for DICOM PHI scanning."""

import argparse
import gc
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import pydicom

from .scanner import scan_file
from .models import BatchReport, ScanReport


def main():
    parser = argparse.ArgumentParser(
        description="DICOM PHI Scanner — scan DICOM files for protected health information",
        epilog=(
            "examples:\n"
            "  dicom-phi-scan image.dcm -o report.json\n"
            "  dicom-phi-scan --dir ./dataset -L -o results.jsonl\n"
            "  dicom-phi-scan --dir ./dataset -L -o results.jsonl --limit 50\n"
            "\n"
            "query the JSONL report:\n"
            "  jq 'select(.risk_level == \"high\") | .filepath' results.jsonl\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("filepath", nargs="?", default=None,
                        help="Path to a single DICOM file")
    parser.add_argument("--dir", dest="directory",
                        help="Recursively scan directory for .dcm files")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Max number of files to scan in batch mode (default: all)",
    )
    parser.add_argument(
        "-o", "--output", dest="output_file", required=True,
        help="Write JSON report to file (single file: pretty JSON, batch: JSONL)",
    )
    parser.add_argument("-L", "--follow-symlinks", dest="follow_symlinks",
                        action="store_true",
                        help="Follow symbolic links when scanning directories")
    parser.add_argument("-v", "--verbose",
                        action="store_true", help="Verbose logging")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if args.filepath and args.directory:
        print("Error: Cannot specify both a file path and --dir", file=sys.stderr)
        sys.exit(1)
    if not args.filepath and not args.directory:
        parser.print_help()
        sys.exit(0)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if args.directory:
        _run_batch(args.directory, args.limit, args.follow_symlinks, args.output_file)
    else:
        try:
            report = scan_file(args.filepath)
        except FileNotFoundError:
            print(f"Error: File not found: {args.filepath}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        _print_summary(report)
        if args.output_file:
            with open(args.output_file, "w") as f:
                f.write(report.model_dump_json(indent=2) + "\n")
            print(f"Report written to {args.output_file}")


def _discover_dcm_files(
    directory: str, limit: int | None = None, follow_symlinks: bool = False,
) -> list[str]:
    """Recursively find all .dcm files in a directory."""
    dirpath = Path(directory)
    if not dirpath.is_dir():
        print(f"Error: Not a directory: {directory}", file=sys.stderr)
        sys.exit(1)

    files = []
    for root, _dirs, filenames in os.walk(dirpath, followlinks=follow_symlinks):
        for fn in filenames:
            if fn.lower().endswith(".dcm"):
                files.append(os.path.join(root, fn))
    files.sort()

    if not files:
        print(f"Error: No .dcm files found in {directory}", file=sys.stderr)
        sys.exit(1)

    if limit and limit < len(files):
        files = files[:limit]

    return files


def _read_modality(filepath: str) -> str:
    """Read modality from DICOM header without loading pixel data."""
    try:
        ds = pydicom.dcmread(filepath, stop_before_pixels=True)
        return getattr(ds, "Modality", "UNKNOWN")
    except Exception:
        return "UNKNOWN"


def _run_batch(
    directory: str,
    limit: int | None = None,
    follow_symlinks: bool = False,
    output_file: str | None = None,
) -> None:
    """Scan all .dcm files, print summary to screen, optionally write JSONL report."""
    files = _discover_dcm_files(directory, limit, follow_symlinks)
    total = len(files)

    print(f"\nScanning {total} files in {directory} ...")
    if output_file:
        print(f"Writing JSONL report to {output_file}")
    print("=" * 72)

    # Aggregate stats — no ScanReport objects retained
    files_with_phi = 0
    files_clean = 0
    files_errored = 0
    error_list: list[tuple[str, str]] = []
    risk_counts: defaultdict[str, int] = defaultdict(int)
    tag_name_counts: defaultdict[str, int] = defaultdict(int)
    pixel_text_counts: defaultdict[str, int] = defaultdict(int)
    total_tag_findings = 0
    total_pixel_findings = 0
    modality_counts: defaultdict[str, int] = defaultdict(int)
    modality_phi: defaultdict[str, int] = defaultdict(int)

    rf = open(output_file, "w") if output_file else None
    try:
        for i, filepath in enumerate(files, 1):
            modality = _read_modality(filepath)
            modality_counts[modality] += 1
            short_path = f"{Path(filepath).parent.name}/{Path(filepath).name}"

            try:
                report = scan_file(filepath)
            except Exception as e:
                files_errored += 1
                error_list.append((filepath, str(e)))
                print(f"\n[{i}/{total}] {short_path} -- ERROR: {e}")
                if rf:
                    rf.write(json.dumps({"filepath": filepath, "error": str(e)}) + "\n")
                gc.collect()
                continue

            # Print per-file findings
            _print_file_findings(report, i, total, short_path)

            # Stream to JSONL
            if rf:
                rf.write(report.model_dump_json() + "\n")

            # Accumulate stats
            risk_counts[report.risk_level.value] += 1
            if report.has_phi:
                files_with_phi += 1
                modality_phi[modality] += 1
            else:
                files_clean += 1

            total_tag_findings += len(report.tag_findings)
            total_pixel_findings += len(report.pixel_findings)
            for f in report.tag_findings:
                tag_name_counts[f.tag_name] += 1
            for f in report.pixel_findings:
                pixel_text_counts[f.text] += 1

            del report
            gc.collect()
    finally:
        if rf:
            rf.close()

    # Aggregate summary
    print("\n" + "=" * 72)
    print("BATCH SCAN SUMMARY")
    print("=" * 72)

    print(f"\nDirectory:      {directory}")
    print(f"Files scanned:  {total}")
    print(f"Files with PHI: {files_with_phi}")
    print(f"Files clean:    {files_clean}")
    print(f"Files errored:  {files_errored}")

    print("\nRisk breakdown:")
    for level in ["high", "medium", "low"]:
        print(f"  {level.upper():8s} {risk_counts.get(level, 0)}")

    print(f"\nFindings: {total_tag_findings} header tags, {total_pixel_findings} pixel texts")

    if modality_counts:
        print("\nBy modality:")
        for mod in sorted(modality_counts):
            phi = modality_phi.get(mod, 0)
            scanned = modality_counts[mod]
            pct = phi / scanned * 100 if scanned else 0
            print(f"  {mod:4s}  {scanned:4d} scanned, {phi:4d} with PHI ({pct:.0f}%)")

    if tag_name_counts:
        print("\nTop header PHI tags:")
        for tag_name, count in sorted(tag_name_counts.items(), key=lambda x: -x[1])[:15]:
            print(f"  {count:4d}x  {tag_name}")

    if pixel_text_counts:
        print("\nTop pixel text detections:")
        for text, count in sorted(pixel_text_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"  {count:4d}x  \"{text}\"")

    if error_list:
        print("\nErrors:")
        for fp, err in error_list:
            print(f"  {fp}: {err}")

    if output_file:
        print(f"\nReport: {output_file}")

    print("=" * 72)
    print()


def _print_batch_summary(batch: BatchReport) -> None:
    """Print a human-readable summary from a BatchReport model.

    Used by JSON-mode callers and tests. The streaming summary path
    (_run_batch_summary) prints richer output inline.
    """
    print(f"\n{'=' * 60}")
    print("DICOM PHI Batch Scan Summary")
    print(f"{'=' * 60}")
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

    print(f"{'=' * 60}\n")


def _print_file_findings(report: ScanReport, index: int, total: int, short_path: str) -> None:
    """Print condensed per-file findings during batch scan."""
    phi_count = report.total_phi_count
    risk = report.risk_level.value.upper()

    status = f"{risk} ({phi_count} findings)" if phi_count > 0 else "CLEAN"
    print(f"\n[{index}/{total}] {short_path} -- {status}")

    if report.tag_findings:
        print(f"  Header tags ({len(report.tag_findings)}):")
        for f in report.tag_findings:
            print(f"    [{f.severity.value.upper()}] {f.tag} {f.tag_name}: {f.value}")

    if report.pixel_findings:
        print(f"  Pixel text ({len(report.pixel_findings)}):")
        for f in report.pixel_findings:
            print(
                f"    [{f.severity.value.upper()}] \"{f.text}\""
                f" at ({f.bbox.x},{f.bbox.y}) {f.bbox.width}x{f.bbox.height}"
                f" conf={f.confidence:.0%}"
            )

    bia = report.burned_in_annotation_value or "MISSING"
    if bia != "NO":
        print(f"  BurnedInAnnotation: {bia}")


def _print_summary(report):
    """Print a human-readable summary of the scan report."""
    risk = report.risk_level.value

    print(f"\n{'='*60}")
    print("DICOM PHI Scan Report")
    print(f"{'='*60}")
    print(f"File: {report.filepath}")
    print(f"Risk Level: {risk.upper()}")
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
