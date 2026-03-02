# CLAUDE.md — dicom-phi-agent

Two-layer DICOM PHI detection: header tag analysis + OCR pixel inspection with Claude classification.

## Quick Start

```bash
pip install -e .
export ANTHROPIC_API_KEY=...
python fixtures/create_test_fixtures.py   # Generate synthetic test DICOMs
dicom-phi-scan fixtures/test_phi_header.dcm              # CLI (agent mode)
dicom-phi-scan fixtures/test_phi_header.dcm --mode direct # CLI (no agent)
dicom-phi-scan fixtures/test_phi_header.dcm --output json # JSON output
uvicorn src.api:app --reload              # REST API at localhost:8000
pytest                                    # Unit tests
ruff check .                              # Lint
```

## Architecture

- `src/scanner.py` — `scan_file()` entry point: runs header scan, then conditional OCR pixel scan
- `src/tag_scanner.py` — Layer 1: checks ~40 DICOM tags against `PHI_TAGS` dict (HIPAA Safe Harbor 18 identifiers). Skips deidentified placeholders (ANONYMOUS, REMOVED, etc.) and NONE/UNKNOWN values
- `src/pixel_scanner.py` — Layer 2: pytesseract OCR + Claude classification. Runs only when BurnedInAnnotation is YES or missing
- `src/agent.py` — Agent mode: Claude tool use to orchestrate both layers
- `src/models.py` — Pydantic models: `PHITagFinding`, `PixelPHIFinding`, `ScanReport`, `BatchReport`
- `src/cli.py` — CLI entry point (`dicom-phi-scan`), supports `--dir` for batch scanning
- `src/api.py` — FastAPI REST API (POST /scan, GET /health)

## TCIA MIDI-B Synthetic Validation Dataset

The TCIA MIDI-B dataset has synthetic PHI injected into DICOM headers and pixels for benchmarking de-identification pipelines.

### Folder layout

```
TCIA-MIDI-B-Synthetic-Validation_20250502/
  <SeriesInstanceUID>/          # folder name = SeriesInstanceUID
    1-1.dcm                     # numbered .dcm files + LICENSE
    1-02.dcm
    ...
```

- 23,921 DICOMs across 280 studies, 8 modalities (PT, CT, MR, US, DX, MG, CR, SR)
- Folders are named by SeriesInstanceUID. File names are sequential (`1-1.dcm`, `1-02.dcm`, ...) — SOPInstanceUID must be read from the DICOM header to match a file to an answer key row.

### Answer key databases

Located in `TCIA-answer-key/`:

| File | Rows | Purpose |
|------|------|---------|
| `MIDI-B-Answer-Key-Validation.db` | 23,921 | Validation set (use this for benchmarking) |
| `MIDI-B-Answer-Key-Test.db` | 29,660 | Test set |

**Schema** — single table `answer_data`:

| Column | Type | Description |
|--------|------|-------------|
| `index` | INTEGER | Row index |
| `group` | INTEGER | Patient group |
| `Modality` | TEXT | PT, CT, MR, US, DX, MG, CR, SR |
| `SOPClassUID` | TEXT | DICOM SOP class |
| `PatientID` | INTEGER | Synthetic patient ID |
| `StudyInstanceUID` | TEXT | Study UID |
| `SeriesInstanceUID` | TEXT | Series UID (= folder name on disk) |
| `SOPInstanceUID` | TEXT | Instance UID (join key to DICOM files) |
| `Digest` | TEXT | File hash |
| `AnswerData` | TEXT | JSON blob with per-tag ground truth |

**AnswerData JSON structure** — keyed by sequential index (`"0"`, `"1"`, ...):

```json
{
  "0": {
    "scope": "<Instance>",
    "tag": "<(0008,0050)>",
    "tag_name": "<Accession Number>",
    "value": "<528B3426>",
    "action": "<text_removed>",
    "action_text": "<528B3426>",
    "answer_category": [...],
    "answer_category_v2": {...}
  }
}
```

**Key `action` values:**
- `<text_removed>` — header tag contains PHI that should be detected/removed
- `<pixels_hidden>` — burned-in pixel text PHI (tag/tag_name/value are null; `action_text` has JSON with `text`, `top_left`, `bottom_right`, optional `font`)
- `<text_retained>`, `<tag_retained>`, `<text_notnull>` — not PHI, retained as-is

**Tag format quirk:** tags in the answer key are wrapped in angle brackets: `<(0008,0050)>`. Strip outer `<>` to get the standard `(XXXX,XXXX)` format used by our pipeline.

### Known coverage gap

Our `PHI_TAGS` dict covers ~22 of 62 answer-key PHI tags. Smoke test (`scripts/smoke_test_midi.py --seed 42`) shows **~70% header detection rate**. Tags we miss:
- Clinical Trial fields: `(0012,*)` — Protocol ID/Name, Site ID, Subject ID, TimePoint ID/Description
- Descriptors: SeriesDescription, StudyDescription, ProtocolName
- Private vendor tags: GE GEMS (`[Slop_int_*]`, `[Service id]`), Siemens
- Other: `(0040,*)` procedure tags, Text Value, Verifying Observer Name

### Smoke test

```bash
python scripts/smoke_test_midi.py            # random sample, 1 file per modality
python scripts/smoke_test_midi.py --seed 42  # reproducible
```

Falls back to header-only scan if tesseract is not installed. Reports per-file and overall TP/FN rates against the answer key.

## Conventions

- Python 3.10+, editable install with `pip install -e .`
- Ruff: line-length=100, target py310
- pytest for tests (`tests/`)
- Claude API calls use claude-sonnet-4-20250514 with tool use
- All test/fixture data is synthetic — never use real patient data
- Scripts go in `scripts/` (utilities, not part of the package)
- PHI and patient privacy are primary concerns in all work
