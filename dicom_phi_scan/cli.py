"""CLI entry point for DICOM PHI scanning."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import RedactionResult, ScanReport

# Mirrors redactor.COMPRESS_CHOICES / ON_UNREDACTABLE_CHOICES; kept literal so argparse
# doesn't force the heavy pydicom/numpy import in redactor until after argument validation.
REDACT_COMPRESS_CHOICES = ("none", "rle", "jpeg-ls", "jpeg2000", "jpeg")
REDACT_ON_UNREDACTABLE_CHOICES = ("copy", "omit")


def main():
    parser = argparse.ArgumentParser(
        description="DICOM PHI Scanner — scan DICOM files for protected health information",
        epilog=(
            "examples:\n"
            "  dicom-phi-scan image.dcm -o report.json\n"
            "  dicom-phi-scan --dir ./dataset -L -o results.jsonl\n"
            "  dicom-phi-scan --dir ./dataset -L -o results.jsonl --limit 50\n"
            "\n"
            "redact the top banner (writes redacted copies; originals untouched;\n"
            "--on-unredactable and --manifest are required):\n"
            "  dicom-phi-scan --redact-banner --dir ./series --output-dir ./redacted \\\n"
            "      --on-unredactable omit --manifest run.jsonl\n"
            "  dicom-phi-scan --redact-banner image.dcm --output-dir ./redacted --banner-height 80 \\\n"
            "      --on-unredactable copy --manifest run.jsonl\n"
            "  dicom-phi-scan --redact-banner --dir ./series --output-dir ./redacted \\\n"
            "      --compress jpeg-ls --on-unredactable omit --manifest run.jsonl\n"
            "\n"
            "preview a single file (GUI window over X-forwarding, or PDF export):\n"
            "  dicom-phi-scan --view image.dcm\n"
            "  dicom-phi-scan --view cine.dcm --pdf sheet.pdf\n"
            "  dicom-phi-scan --view cine.dcm --frame 120 --pdf frame120.pdf\n"
            "\n"
            "query the JSONL report:\n"
            "  jq 'select(.risk_level == \"high\") | .filepath' results.jsonl\n"
            "\n"
            "exit codes (scan mode):\n"
            "  0  all files clean\n"
            "  1  PHI detected in one or more files\n"
            "  2  one or more files errored\n"
            "exit codes (redact mode):\n"
            "  0  every file redacted (or benignly copied/omitted with no pixel data)\n"
            "  2  one or more files not fully redacted — review the manifest\n"
            "exit codes (view mode):\n"
            "  0  window shown or PDF written\n"
            "  2  error (missing file, no pixel data, decode failure, no display)\n"
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
        "-o", "--output", dest="output_file", default=None,
        help="Write JSON report to file (single file: pretty JSON, batch: JSONL). "
             "Required in scan mode; not used with --redact-banner.",
    )
    parser.add_argument("-L", "--follow-symlinks", dest="follow_symlinks",
                        action="store_true",
                        help="Follow symbolic links when scanning directories")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU for OCR even if GPU is available")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted batch scan; skip files already in output JSONL")
    parser.add_argument("-v", "--verbose",
                        action="store_true", help="Verbose logging")

    redact = parser.add_argument_group("banner redaction (write redacted copies)")
    redact.add_argument(
        "--redact-banner", action="store_true",
        help="Redaction mode: black out the top banner and write redacted copies "
             "to --output-dir (originals are never modified)",
    )
    redact.add_argument(
        "--output-dir", dest="output_dir", default=None,
        help="Directory for redacted copies (required with --redact-banner)",
    )
    redact.add_argument(
        "--banner-height", dest="banner_height", type=int, default=None,
        help="Rows to black out from the top. Optional if (0018,6011) Sequence of "
             "Ultrasound Regions is present (banner height is derived from it).",
    )
    redact.add_argument(
        "--compress", choices=REDACT_COMPRESS_CHOICES, default="none",
        help="Output encoding for decoded frames (default: none/uncompressed). "
             "rle/jpeg-ls/jpeg2000 are lossless; jpeg is lossy. jpeg-ls/jpeg2000/jpeg "
             "need an encoder plugin. Recommended for size: jpeg-ls.",
    )
    redact.add_argument(
        "--to-rgb", dest="to_rgb", action="store_true",
        help="Convert YBR color frames to RGB on output (default: keep native color)",
    )
    redact.add_argument(
        "--on-unredactable", dest="on_unredactable",
        choices=REDACT_ON_UNREDACTABLE_CHOICES, default=None,
        help="Required with --redact-banner: what to do with a file that cannot be "
             "redacted (no pixels, no banner height, decode failure). 'copy' writes the "
             "original through unchanged (may ship un-redacted pixels); 'omit' writes "
             "nothing for it. Every outcome is recorded in --manifest.",
    )
    redact.add_argument(
        "--manifest", dest="manifest", default=None,
        help="Required with --redact-banner: path to a JSONL manifest with one line per "
             "input file recording its outcome (status, reason, paths).",
    )
    redact.add_argument(
        "--force", action="store_true",
        help="Overwrite existing files in --output-dir (default: refuse)",
    )

    view = parser.add_argument_group("preview / view a single file")
    view.add_argument(
        "--view", action="store_true",
        help="View mode: display a single .dcm file's pixels (2D or cineloop) in a "
             "GUI window (needs X-forwarding), or export to PDF with --pdf",
    )
    view.add_argument(
        "--frame", type=int, default=None,
        help="Frame to show first (cineloop); with --pdf, export just this frame full-page",
    )
    view.add_argument(
        "--pdf", dest="pdf", default=None,
        help="Render to this PDF instead of opening a window (works headless). "
             "A cineloop becomes a contact sheet unless --frame is given.",
    )
    view.add_argument(
        "--sheet-cols", dest="sheet_cols", type=int, default=5,
        help="Columns in the cineloop contact-sheet PDF (default: 5)",
    )

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

    if args.resume:
        if not args.directory:
            print("Error: --resume is only supported in batch mode (--dir)", file=sys.stderr)
            sys.exit(1)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # --- View mode: preview a single file's pixels, no OCR/scan report ---
    if args.view:
        if args.directory:
            print("Error: --view takes a single file, not --dir", file=sys.stderr)
            sys.exit(1)
        if not args.filepath:
            print("Error: --view requires a file path", file=sys.stderr)
            sys.exit(1)
        if args.output_file:
            print("Error: -o/--output is not used with --view (use --pdf to export)",
                  file=sys.stderr)
            sys.exit(1)
        sys.exit(_run_view(
            args.filepath, frame=args.frame, pdf=args.pdf, sheet_cols=args.sheet_cols,
        ))

    # --- Redaction mode: write redacted copies, no OCR/scan report ---
    if args.redact_banner:
        if not args.output_dir:
            print("Error: --redact-banner requires --output-dir", file=sys.stderr)
            sys.exit(1)
        if not args.on_unredactable:
            print("Error: --redact-banner requires --on-unredactable {copy,omit} "
                  "(choose what happens to files that cannot be redacted)", file=sys.stderr)
            sys.exit(1)
        if not args.manifest:
            print("Error: --redact-banner requires --manifest PATH "
                  "(a JSONL record of every file's outcome)", file=sys.stderr)
            sys.exit(1)
        if args.output_file:
            print("Error: -o/--output is not used with --redact-banner (use --output-dir)",
                  file=sys.stderr)
            sys.exit(1)
        if args.resume:
            print("Error: --resume is not supported with --redact-banner", file=sys.stderr)
            sys.exit(1)
        if args.directory:
            rc = _run_redact_batch(
                args.directory, args.output_dir, args.follow_symlinks,
                banner_height=args.banner_height, compress=args.compress,
                to_rgb=args.to_rgb, on_unredactable=args.on_unredactable,
                manifest=args.manifest, force=args.force, limit=args.limit,
            )
        else:
            rc = _run_redact_single(
                args.filepath, args.output_dir,
                banner_height=args.banner_height, compress=args.compress,
                to_rgb=args.to_rgb, on_unredactable=args.on_unredactable,
                manifest=args.manifest, force=args.force,
            )
        sys.exit(rc)

    # --- Scan mode ---
    if not args.output_file:
        print("Error: -o/--output is required in scan mode", file=sys.stderr)
        sys.exit(1)

    from .pixel_scanner import init_reader
    from .scanner import scan_file

    init_reader(gpu=False if args.cpu else None)

    done_paths: set[str] | None = None
    if args.resume:
        done_paths = _load_done_paths(args.output_file)
        logging.getLogger(__name__).info(
            "Loaded %d completed paths from %s", len(done_paths), args.output_file,
        )

    if args.directory:
        rc = _run_batch(
            args.directory, args.limit, args.follow_symlinks, args.output_file,
            resume=args.resume, done_paths=done_paths,
        )
        sys.exit(rc)
    else:
        try:
            report = scan_file(args.filepath)
        except FileNotFoundError:
            print(f"Error: File not found: {args.filepath}", file=sys.stderr)
            sys.exit(2)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(2)

        _print_summary(report)
        if args.output_file:
            with open(args.output_file, "w") as f:
                f.write(report.model_dump_json(indent=2) + "\n")
            print(f"Report written to {args.output_file}")
        sys.exit(1 if report.has_phi else 0)


def _load_done_paths(output_file: str) -> set[str]:
    """Read an existing JSONL output and return paths already scanned."""
    logger = logging.getLogger(__name__)
    done: set[str] = set()
    path = Path(output_file)
    if not path.exists():
        return done
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                raw_path = record.get("filepath", "")
                if raw_path:
                    done.add(raw_path)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping corrupted line %d in %s", line_no, output_file,
                )
    return done


def _discover_dcm_files(
    directory: str, limit: int | None = None, follow_symlinks: bool = False,
    done_paths: set[str] | None = None,
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

    if done_paths:
        before = len(files)
        files = [f for f in files if f not in done_paths]
        skipped = before - len(files)
        if skipped:
            logging.getLogger(__name__).info(
                "Resuming: skipped %d already-scanned files, %d remaining", skipped, len(files),
            )

    if limit and limit < len(files):
        files = files[:limit]

    return files


def _read_modality(filepath: str) -> str:
    """Read modality from DICOM header without loading pixel data."""
    import pydicom
    try:
        ds = pydicom.dcmread(filepath, stop_before_pixels=True)
        return getattr(ds, "Modality", "UNKNOWN")
    except Exception:
        return "UNKNOWN"


GC_INTERVAL = 500
FLUSH_INTERVAL = 100


def _run_batch(
    directory: str,
    limit: int | None = None,
    follow_symlinks: bool = False,
    output_file: str | None = None,
    resume: bool = False,
    done_paths: set[str] | None = None,
) -> int:
    """Scan all .dcm files, print summary to screen, optionally write JSONL report.

    Returns exit code: 0 = clean, 1 = PHI found, 2 = errors.
    """
    from .models import FileError
    from .scanner import scan_file

    files = _discover_dcm_files(directory, limit, follow_symlinks, done_paths)
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

    if not files:
        print("\nNothing left to scan.")
        return 0

    mode = "a" if resume else "w"
    rf = open(output_file, mode) if output_file else None
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
                    rf.write(FileError(filepath=filepath, error=str(e)).model_dump_json() + "\n")
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

            # Periodic maintenance
            if i % FLUSH_INTERVAL == 0 and rf:
                rf.flush()
            if i % GC_INTERVAL == 0:
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

    if files_errored:
        return 2
    if files_with_phi:
        return 1
    return 0


def _redact_out_path(input_path: str, base_dir: str | None, output_dir: str) -> str:
    """Map an input file to its path under output_dir, preserving structure.

    For batch (base_dir given) the input's path relative to base_dir is preserved;
    for a single file, just the basename is used.
    """
    if base_dir is not None:
        rel = os.path.relpath(input_path, base_dir)
    else:
        rel = os.path.basename(input_path)
    return os.path.join(output_dir, rel)


def _print_redaction(result: "RedactionResult", index: int | None, total: int | None) -> None:
    """Print a per-file redaction status line."""
    prefix = f"[{index}/{total}] " if index is not None else ""
    detail = ""
    if result.status == "redacted":
        detail = (f" -- REDACTED {result.frames}f, top {result.banner_height}px"
                  f", {result.photometric_in}->{result.photometric_out}")
    elif result.status == "copied":
        detail = f" -- COPIED ({result.message})"
    elif result.status == "omitted":
        detail = f" -- OMITTED ({result.message})"
    else:  # error
        detail = f" -- ERROR: {result.message}"
    print(f"{prefix}{Path(result.output_path).name}{detail}")


def _redaction_error_result(
    input_path: str, output_path: str, message: str
) -> "RedactionResult":
    """Build a RedactionResult for a failure handled in the CLI (before/around redact_file)."""
    from .models import RedactionResult

    return RedactionResult(
        input_path=input_path, output_path=output_path, status="error",
        frames=0, banner_height=0, photometric_in=None, photometric_out=None,
        transfer_syntax_out=None, message=message,
    )


def _redaction_needs_review(result: "RedactionResult") -> bool:
    """True if a file was not fully redacted in a PHI-relevant way (drives exit code 2).

    A ``copied``/``omitted`` file is benign only when it had no pixels to redact
    (``frames == 0``); otherwise un-redacted pixels were shipped or dropped.
    """
    if result.status == "redacted":
        return False
    if result.status in ("copied", "omitted"):
        return result.frames >= 1
    return True  # error


def _run_redact_single(
    filepath: str,
    output_dir: str,
    *,
    banner_height: int | None,
    compress: str,
    to_rgb: bool,
    on_unredactable: str,
    manifest: str,
    force: bool,
) -> int:
    """Redact a single file to output_dir; write a one-line manifest. Returns exit code."""
    from .redactor import redact_file

    out_path = _redact_out_path(filepath, None, output_dir)

    if not os.path.isfile(filepath):
        result = _redaction_error_result(filepath, out_path, f"file not found: {filepath}")
    elif os.path.exists(out_path) and not force:
        result = _redaction_error_result(filepath, out_path, "output exists (use --force)")
    else:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        try:
            result = redact_file(
                filepath, out_path, banner_height=banner_height,
                compress=compress, to_rgb=to_rgb, on_unredactable=on_unredactable,
            )
        except Exception as e:  # unexpected — record rather than crash
            result = _redaction_error_result(filepath, out_path, str(e))

    with open(manifest, "w") as mf:
        mf.write(result.model_dump_json() + "\n")

    _print_redaction(result, None, None)
    if result.status == "redacted":
        print(f"Redacted copy written to {out_path}")
    print(f"Manifest: {manifest}")
    return 2 if _redaction_needs_review(result) else 0


def _run_redact_batch(
    directory: str,
    output_dir: str,
    follow_symlinks: bool,
    *,
    banner_height: int | None,
    compress: str,
    to_rgb: bool,
    on_unredactable: str,
    manifest: str,
    force: bool,
    limit: int | None,
) -> int:
    """Redact every .dcm under directory into output_dir, preserving structure.

    Writes one JSONL manifest line per discovered file. Returns exit code: 0 = every
    file redacted or benignly copied/omitted, 2 = one or more files need review.
    """
    from .redactor import redact_file

    files = _discover_dcm_files(directory, limit, follow_symlinks)
    total = len(files)

    print(f"\nRedacting {total} files from {directory} -> {output_dir}")
    print("=" * 72)

    counts: defaultdict[str, int] = defaultdict(int)
    needs_review = 0
    with open(manifest, "w") as mf:
        for i, filepath in enumerate(files, 1):
            out_path = _redact_out_path(filepath, directory, output_dir)

            if os.path.exists(out_path) and not force:
                result = _redaction_error_result(
                    filepath, out_path, "output exists (use --force)"
                )
            else:
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                try:
                    result = redact_file(
                        filepath, out_path, banner_height=banner_height,
                        compress=compress, to_rgb=to_rgb, on_unredactable=on_unredactable,
                    )
                except Exception as e:  # unexpected — record and continue
                    result = _redaction_error_result(filepath, out_path, str(e))

            mf.write(result.model_dump_json() + "\n")
            counts[result.status] += 1
            if _redaction_needs_review(result):
                needs_review += 1
            _print_redaction(result, i, total)

            if i % GC_INTERVAL == 0:
                gc.collect()

    print("\n" + "=" * 72)
    print("REDACTION SUMMARY")
    print("=" * 72)
    print(f"\nInput:     {directory}")
    print(f"Output:    {output_dir}")
    print(f"Files:     {total}")
    print(f"Redacted:  {counts.get('redacted', 0)}")
    print(f"Copied:    {counts.get('copied', 0)}")
    print(f"Omitted:   {counts.get('omitted', 0)}")
    print(f"Errored:   {counts.get('error', 0)}")
    print(f"Manifest:  {manifest}")
    if needs_review:
        print(f"\n{needs_review} file(s) not fully redacted — review the manifest:")
        print(f"  jq 'select(.status != \"redacted\")' {manifest}")
    print("=" * 72 + "\n")

    return 2 if needs_review else 0


def _run_view(
    filepath: str, *, frame: int | None, pdf: str | None, sheet_cols: int
) -> int:
    """Preview a single file: GUI window, or PDF export with --pdf. Returns exit code."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("Error: view mode needs matplotlib — install it with: pip install '.[viz]'",
              file=sys.stderr)
        return 2

    import pydicom

    from .viewer import ViewerError, export_pdf, format_header, load_frames, show_interactive

    if not os.path.isfile(filepath):
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        return 2

    ds = pydicom.dcmread(filepath)
    if "PixelData" not in ds:
        print(f"Error: {Path(filepath).name} has no pixel data to view", file=sys.stderr)
        return 2

    print(f"\n{filepath}")
    print(format_header(ds))
    print()

    photometric = str(getattr(ds, "PhotometricInterpretation", ""))
    try:
        frames = load_frames(ds)
    except ViewerError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    title = Path(filepath).name

    if pdf:
        try:
            pages = export_pdf(
                frames, photometric, pdf, frame=frame, sheet_cols=sheet_cols, title=title,
            )
        except ViewerError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 2
        print(f"Wrote {pdf} ({pages} page{'s' if pages != 1 else ''})")
        return 0

    if not os.environ.get("DISPLAY"):
        print("Error: no X display detected — enable X-forwarding (ssh -X) or export "
              "with --pdf PATH", file=sys.stderr)
        return 2

    try:
        show_interactive(frames, photometric, start_frame=frame or 0, title=title)
    except ViewerError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    return 0


def _print_file_findings(report: "ScanReport", index: int, total: int, short_path: str) -> None:
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
