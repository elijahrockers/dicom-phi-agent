#!/usr/bin/env python3
"""Create a symlinked subset of the TCIA MIDI-B dataset for faster benchmarking.

Builds TCIA-subset/ (or custom dir) with relative symlinks to series folders,
targeting a configurable number of files per modality. Small modalities that
have fewer files than the target are included in full.

Usage:
    python scripts/create_tcia_subset.py                          # defaults
    python scripts/create_tcia_subset.py --target 200 --seed 42   # 200 files/modality
    python scripts/create_tcia_subset.py --force                  # overwrite existing
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

ANSWER_KEY_DB = PROJECT_ROOT / "TCIA-answer-key" / "MIDI-B-Answer-Key-Validation.db"
DEFAULT_TCIA_DIR = PROJECT_ROOT / "TCIA-MIDI-B-Synthetic-Validation_20250502"
DEFAULT_SUBSET_DIR = PROJECT_ROOT / "TCIA-subset"

MODALITIES = ["PT", "CT", "MR", "US", "DX", "MG", "CR", "SR"]


def query_series_counts(conn: sqlite3.Connection) -> dict[str, list[tuple[str, int]]]:
    """Return {Modality: [(SeriesInstanceUID, file_count), ...]}."""
    cursor = conn.cursor()
    cursor.execute(
        "SELECT Modality, SeriesInstanceUID, COUNT(*) as cnt "
        "FROM answer_data GROUP BY Modality, SeriesInstanceUID"
    )
    result: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for modality, series_uid, cnt in cursor:
        result[modality].append((series_uid, cnt))
    return result


def select_series(
    series_counts: list[tuple[str, int]],
    target: int,
    rng: random.Random,
) -> list[tuple[str, int]]:
    """Greedily select series to reach at least `target` files.

    Strategy: shuffle, then stable-sort ascending by file count so we pick
    small series first (faster scanning, broader coverage). Add series until
    the cumulative file count >= target.
    """
    total_files = sum(cnt for _, cnt in series_counts)
    if total_files <= target:
        return series_counts

    shuffled = list(series_counts)
    rng.shuffle(shuffled)
    # stable sort: small series first
    shuffled.sort(key=lambda x: x[1])

    selected = []
    accumulated = 0
    for series_uid, cnt in shuffled:
        selected.append((series_uid, cnt))
        accumulated += cnt
        if accumulated >= target:
            break
    return selected


def create_subset(
    tcia_dir: Path,
    subset_dir: Path,
    series_selections: dict[str, list[tuple[str, int]]],
    force: bool,
) -> dict[str, dict]:
    """Create symlinks and return per-modality stats."""
    if force and subset_dir.exists():
        shutil.rmtree(subset_dir)

    subset_dir.mkdir(exist_ok=True)

    stats: dict[str, dict] = {}
    for modality in MODALITIES:
        selected = series_selections.get(modality, [])
        linked = 0
        skipped_missing = 0
        skipped_exists = 0

        for series_uid, cnt in selected:
            source = tcia_dir / series_uid
            link = subset_dir / series_uid

            if not source.is_dir():
                skipped_missing += 1
                print(f"  WARN: series dir missing on disk: {series_uid} ({modality})")
                continue

            if link.exists() or link.is_symlink():
                skipped_exists += 1
                continue

            # Relative symlink: TCIA-subset/<uid> -> ../TCIA-MIDI-B-.../<uid>
            rel_target = os.path.relpath(source, subset_dir)
            link.symlink_to(rel_target)
            linked += 1

        total_files = sum(cnt for _, cnt in selected)
        stats[modality] = {
            "series": len(selected),
            "files": total_files,
            "linked": linked,
            "skipped_missing": skipped_missing,
            "skipped_exists": skipped_exists,
        }
    return stats


def print_summary(
    series_counts: dict[str, list[tuple[str, int]]],
    stats: dict[str, dict],
    target: int,
):
    """Print a summary table of the subset."""
    print(f"\n{'Modality':<10} {'Series':>7} {'Files':>7} {'Linked':>7} {'Skip(miss)':>11} "
          f"{'Skip(exist)':>12} {'Available':>10}")
    print("-" * 74)
    total_series = 0
    total_files = 0
    total_linked = 0
    for mod in MODALITIES:
        s = stats.get(mod, {})
        avail = sum(cnt for _, cnt in series_counts.get(mod, []))
        total_series += s.get("series", 0)
        total_files += s.get("files", 0)
        total_linked += s.get("linked", 0)
        print(
            f"{mod:<10} {s.get('series', 0):>7} {s.get('files', 0):>7} "
            f"{s.get('linked', 0):>7} {s.get('skipped_missing', 0):>11} "
            f"{s.get('skipped_exists', 0):>12} {avail:>10}"
        )
    print("-" * 74)
    total_avail = sum(sum(cnt for _, cnt in v) for v in series_counts.values())
    print(f"{'TOTAL':<10} {total_series:>7} {total_files:>7} {total_linked:>7} "
          f"{'':>11} {'':>12} {total_avail:>10}")
    print(f"\nTarget: {target} files/modality")


def main():
    parser = argparse.ArgumentParser(
        description="Create a symlinked TCIA MIDI-B subset for benchmarking"
    )
    parser.add_argument(
        "--target", type=int, default=200,
        help="Target number of files per modality (default: 200)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible selection (default: 42)",
    )
    parser.add_argument(
        "--subset-dir", type=Path, default=DEFAULT_SUBSET_DIR,
        help=f"Output subset directory (default: {DEFAULT_SUBSET_DIR.name})",
    )
    parser.add_argument(
        "--tcia-dir", type=Path, default=DEFAULT_TCIA_DIR,
        help="Path to the full TCIA dataset directory",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Remove existing subset directory before creating",
    )
    args = parser.parse_args()

    if args.target < 1:
        print("ERROR: --target must be >= 1")
        sys.exit(1)

    if not ANSWER_KEY_DB.exists():
        print(f"ERROR: Answer key DB not found: {ANSWER_KEY_DB}")
        sys.exit(1)

    tcia_dir = args.tcia_dir.resolve()
    if not tcia_dir.is_dir():
        print(f"ERROR: TCIA data directory not found: {tcia_dir}")
        sys.exit(1)

    rng = random.Random(args.seed)

    print("Loading answer key...")
    conn = sqlite3.connect(str(ANSWER_KEY_DB))
    series_counts = query_series_counts(conn)
    conn.close()

    total_entries = sum(sum(cnt for _, cnt in v) for v in series_counts.values())
    print(f"  {total_entries} files across {sum(len(v) for v in series_counts.values())} series")

    print(f"\nSelecting series (target={args.target} files/modality, seed={args.seed})...")
    selections: dict[str, list[tuple[str, int]]] = {}
    for mod in MODALITIES:
        counts = series_counts.get(mod, [])
        selections[mod] = select_series(counts, args.target, rng)

    subset_dir = args.subset_dir.resolve()
    print(f"Creating symlinks in {subset_dir}...")
    stats = create_subset(tcia_dir, subset_dir, selections, args.force)

    print_summary(series_counts, stats, args.target)
    print(f"\nDone. Subset directory: {subset_dir}")


if __name__ == "__main__":
    main()
