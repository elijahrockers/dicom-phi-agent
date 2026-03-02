#!/usr/bin/env python3
"""Smoke test: run the DICOM PHI pipeline against the TCIA MIDI-B answer key.

Samples 1 random DICOM per modality from the TCIA Synthetic Validation dataset,
runs our scan_file() pipeline, then compares findings against the answer key DB
to report true positives, false negatives, and overall detection rates.

Usage:
    python scripts/smoke_test_midi.py
    python scripts/smoke_test_midi.py --seed 42   # reproducible sampling
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import pydicom

# Suppress noisy pydicom UID validation warnings from synthetic data
warnings.filterwarnings("ignore", message="Invalid value for VR UI")

# Add project root to path so we can import src.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scanner import scan_file  # noqa: E402
from src.tag_scanner import scan_tags, get_burned_in_annotation  # noqa: E402
from src.models import ScanReport, Severity  # noqa: E402

# --- Configuration -----------------------------------------------------------

ANSWER_KEY_DB = PROJECT_ROOT / "TCIA-answer-key" / "MIDI-B-Answer-Key-Validation.db"
TCIA_DIR = PROJECT_ROOT / "TCIA-MIDI-B-Synthetic-Validation_20250502"
MODALITIES = ["PT", "CT", "MR", "US", "DX", "MG", "CR", "SR"]


# --- Answer key helpers -------------------------------------------------------


def load_answer_key_index(conn: sqlite3.Connection) -> dict[str, list[dict]]:
    """Build {Modality: [{SOPInstanceUID, SeriesInstanceUID, AnswerData}, ...]}."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT Modality, SOPInstanceUID, SeriesInstanceUID, AnswerData FROM answer_data"
    )
    index: dict[str, list[dict]] = defaultdict(list)
    for modality, sop_uid, series_uid, answer_json in cursor:
        index[modality].append(
            {
                "SOPInstanceUID": sop_uid,
                "SeriesInstanceUID": series_uid,
                "AnswerData": answer_json,  # parse lazily
            }
        )
    return index


def parse_ground_truth(answer_json: str) -> tuple[list[dict], list[dict]]:
    """Extract header PHI (text_removed) and pixel PHI (pixels_hidden) from AnswerData.

    Returns:
        (header_entries, pixel_entries) where each entry is the raw answer dict.
    """
    data = json.loads(answer_json)
    header_phi = []
    pixel_phi = []
    for entry in data.values():
        action = entry.get("action", "")
        if action == "<text_removed>":
            header_phi.append(entry)
        elif action == "<pixels_hidden>":
            pixel_phi.append(entry)
    return header_phi, pixel_phi


def normalize_tag(raw_tag: str) -> str:
    """Strip angle brackets: '<(0008,0050)>' -> '(0008,0050)', uppercase."""
    tag = raw_tag.strip("<>").strip()
    return tag.upper()


# --- File discovery -----------------------------------------------------------


def find_dicom_file(series_uid: str, sop_uid: str) -> Path | None:
    """Locate a .dcm file by SeriesInstanceUID (folder) and SOPInstanceUID (header)."""
    series_dir = TCIA_DIR / series_uid
    if not series_dir.is_dir():
        return None

    dcm_files = sorted(series_dir.glob("*.dcm"))
    if not dcm_files:
        return None

    # Fast path: single-file series
    if len(dcm_files) == 1:
        return dcm_files[0]

    # Multi-file series: read headers until we find the matching SOP
    for dcm_path in dcm_files:
        try:
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, specific_tags=[(0x0008, 0x0018)])
            if str(ds.SOPInstanceUID) == sop_uid:
                return dcm_path
        except Exception:
            continue

    return None


# --- Sampling -----------------------------------------------------------------


def sample_one_per_modality(
    index: dict[str, list[dict]], rng: random.Random
) -> list[tuple[str, dict]]:
    """Pick 1 random entry per modality, preferring small series for speed."""
    samples = []
    for mod in MODALITIES:
        entries = index.get(mod, [])
        if not entries:
            print(f"  WARN: no entries for modality {mod}")
            continue

        # Group by series and prefer single-file series for faster file discovery
        by_series: dict[str, list[dict]] = defaultdict(list)
        for e in entries:
            by_series[e["SeriesInstanceUID"]].append(e)

        small_series = {k: v for k, v in by_series.items() if len(v) <= 4}
        if small_series:
            series_uid = rng.choice(list(small_series.keys()))
            entry = rng.choice(small_series[series_uid])
        else:
            entry = rng.choice(entries)

        samples.append((mod, entry))
    return samples


# --- Comparison ---------------------------------------------------------------


def compare_findings(
    report: ScanReport,
    header_truth: list[dict],
    pixel_truth: list[dict],
) -> dict:
    """Compare pipeline output against ground truth.

    Returns dict with TP/FN lists for header and pixel PHI.
    """
    # Build set of detected tag addresses from pipeline
    detected_tags = {f.tag.upper() for f in report.tag_findings}

    header_tp = []
    header_fn = []

    for entry in header_truth:
        raw_tag = entry.get("tag", "")
        if not raw_tag:
            continue
        tag = normalize_tag(raw_tag)
        tag_name = entry.get("tag_name", "").strip("<>")

        if tag in detected_tags:
            header_tp.append(tag_name)
        else:
            header_fn.append(tag_name)

    # Pixel PHI: did we detect *any* pixel findings?
    pixel_tp = min(len(report.pixel_findings), len(pixel_truth))
    pixel_fn = max(0, len(pixel_truth) - pixel_tp)

    return {
        "header_tp": header_tp,
        "header_fn": header_fn,
        "pixel_expected": len(pixel_truth),
        "pixel_detected": len(report.pixel_findings),
        "pixel_tp": pixel_tp,
        "pixel_fn": pixel_fn,
    }


# --- Main ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Smoke test pipeline vs TCIA MIDI-B answer key")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # Validate paths
    if not ANSWER_KEY_DB.exists():
        print(f"ERROR: Answer key DB not found: {ANSWER_KEY_DB}")
        sys.exit(1)
    if not TCIA_DIR.is_dir():
        print(f"ERROR: TCIA data directory not found: {TCIA_DIR}")
        sys.exit(1)

    print("Loading answer key index...")
    conn = sqlite3.connect(str(ANSWER_KEY_DB))
    index = load_answer_key_index(conn)
    conn.close()
    print(f"  {sum(len(v) for v in index.values())} total entries across {len(index)} modalities\n")

    # Sample 1 file per modality
    print("Sampling 1 file per modality...")
    samples = sample_one_per_modality(index, rng)
    print(f"  Selected {len(samples)} files\n")

    # Accumulators for overall stats
    total_header_tp = 0
    total_header_fn = 0
    total_pixel_tp = 0
    total_pixel_fn = 0
    all_detected_tags: set[str] = set()
    all_missed_tags: set[str] = set()
    errors = 0

    print("=" * 72)
    print("PER-FILE RESULTS")
    print("=" * 72)

    for modality, entry in samples:
        sop_uid = entry["SOPInstanceUID"]
        series_uid = entry["SeriesInstanceUID"]
        sop_short = sop_uid[:25] + "..."

        # Find the file on disk
        dcm_path = find_dicom_file(series_uid, sop_uid)
        if dcm_path is None:
            print(f"\n[{modality}] SOP={sop_short}  FILE NOT FOUND (series={series_uid})")
            errors += 1
            continue

        # Parse ground truth
        header_truth, pixel_truth = parse_ground_truth(entry["AnswerData"])

        # Run pipeline — try full scan first, fall back to header-only if OCR unavailable
        try:
            report = scan_file(str(dcm_path))
        except Exception as e:
            if "tesseract" in str(e).lower():
                # OCR not available — run header-only scan
                ds = pydicom.dcmread(str(dcm_path))
                tag_findings = scan_tags(ds)
                bia_present, bia_value = get_burned_in_annotation(ds)
                report = ScanReport(
                    filepath=str(dcm_path),
                    tag_findings=tag_findings,
                    pixel_findings=[],
                    burned_in_annotation_tag_present=bia_present,
                    burned_in_annotation_value=bia_value,
                    total_phi_count=len(tag_findings),
                    risk_level=Severity.HIGH if tag_findings else Severity.LOW,
                    recommendations=["(header-only scan — tesseract not installed)"],
                )
                ocr_skipped = True
            else:
                print(f"\n[{modality}] SOP={sop_short}  PIPELINE ERROR: {e}")
                errors += 1
                continue
        else:
            ocr_skipped = False

        # Compare
        result = compare_findings(report, header_truth, pixel_truth)

        tp_count = len(result["header_tp"])
        fn_count = len(result["header_fn"])
        total_header = tp_count + fn_count

        total_header_tp += tp_count
        total_header_fn += fn_count
        total_pixel_tp += result["pixel_tp"]
        total_pixel_fn += result["pixel_fn"]
        all_detected_tags.update(result["header_tp"])
        all_missed_tags.update(result["header_fn"])

        ocr_note = " [OCR skipped]" if ocr_skipped else ""
        print(f"\n[{modality}] SOP={sop_short}{ocr_note}")
        print(f"  Header PHI: {tp_count}/{total_header} detected", end="")
        if total_header > 0:
            print(f" ({tp_count/total_header*100:.0f}%)", end="")
        print(f", Pixel PHI: {result['pixel_detected']}/{result['pixel_expected']} detected")

        if result["header_tp"]:
            print(f"  DETECTED: {', '.join(sorted(result['header_tp']))}")
        if result["header_fn"]:
            print(f"  MISSED:   {', '.join(sorted(result['header_fn']))}")

    # --- Overall summary ---
    total_header_all = total_header_tp + total_header_fn
    total_pixel_all = total_pixel_tp + total_pixel_fn

    print("\n" + "=" * 72)
    print("OVERALL SUMMARY")
    print("=" * 72)

    if total_header_all > 0:
        pct = total_header_tp / total_header_all * 100
        print(f"Header PHI: {total_header_tp}/{total_header_all} detected ({pct:.1f}%)")
    else:
        print("Header PHI: no ground-truth entries found")

    if total_pixel_all > 0:
        pct = total_pixel_tp / total_pixel_all * 100
        print(f"Pixel PHI:  {total_pixel_tp}/{total_pixel_all} detected ({pct:.1f}%)")
    else:
        print("Pixel PHI:  no ground-truth pixel entries in sampled files")

    # Tags that were always detected vs never detected
    only_missed = all_missed_tags - all_detected_tags
    only_detected = all_detected_tags - all_missed_tags

    if only_detected:
        print(f"\nTags always detected:  {', '.join(sorted(only_detected))}")
    if only_missed:
        print(f"Tags never detected:   {', '.join(sorted(only_missed))}")

    print(f"\nPipeline errors: {errors}")
    print(f"Files sampled:   {len(samples)}")

    if args.seed is not None:
        print(f"Random seed:     {args.seed}")


if __name__ == "__main__":
    main()
