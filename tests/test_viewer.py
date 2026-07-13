"""Tests for the preview/view mode (dicom_phi_scan.viewer + CLI dispatch).

Synthetic in-memory datasets only. The interactive GUI (show_interactive) needs a
display and can't run headless, so we test the decoding/color logic, the Agg-backed
PDF export, and the CLI error paths.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from dicom_phi_scan.viewer import (
    ViewerError,
    export_pdf,
    load_frames,
    to_display,
)


def _make_ds(pixels: np.ndarray, photometric: str, samples: int, frames: int = 1) -> Dataset:
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    if samples > 1:
        ds.Rows, ds.Columns = pixels.shape[-3], pixels.shape[-2]
        ds.PlanarConfiguration = 0
    else:
        ds.Rows, ds.Columns = pixels.shape[-2], pixels.shape[-1]
    ds.SamplesPerPixel = samples
    ds.PhotometricInterpretation = photometric
    if frames > 1:
        ds.NumberOfFrames = frames
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = np.ascontiguousarray(pixels.astype(np.uint8)).tobytes()
    return ds


def _pdf_page_count(path) -> int:
    """Count PDF pages via pypdf if available, else fall back to a validity check."""
    try:
        from pypdf import PdfReader
    except ImportError:
        data = open(path, "rb").read()
        assert data.startswith(b"%PDF")  # at least a valid PDF
        return -1
    return len(PdfReader(str(path)).pages)


# --- load_frames: shape contract --------------------------------------------

def test_load_frames_single_rgb_gets_frame_axis():
    arr = np.zeros((30, 40, 3), dtype=np.uint8)
    frames = load_frames(_make_ds(arr, "RGB", 3))
    assert frames.shape == (1, 30, 40, 3)


def test_load_frames_multiframe_rgb_keeps_all_frames():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(5, 30, 40, 3), dtype=np.uint8)
    frames = load_frames(_make_ds(arr, "RGB", 3, frames=5))
    assert frames.shape == (5, 30, 40, 3)
    assert (frames == arr).all()  # every frame present, not just frame 0


def test_load_frames_single_mono_gets_frame_axis():
    arr = np.zeros((20, 20), dtype=np.uint8)
    frames = load_frames(_make_ds(arr, "MONOCHROME2", 1))
    assert frames.shape == (1, 20, 20)


# --- to_display: color + colormap -------------------------------------------

def test_to_display_rgb_passthrough():
    frame = np.full((4, 4, 3), 200, dtype=np.uint8)
    out, cmap = to_display(frame, "RGB")
    assert cmap is None
    assert (out == frame).all()


def test_to_display_ybr_full_converts_to_rgb():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[..., 0] = 76   # Y
    frame[..., 1] = 128  # Cb
    frame[..., 2] = 128  # Cr
    out, cmap = to_display(frame, "YBR_FULL")
    assert cmap is None
    # Y=76, neutral chroma -> gray ~76 in all channels; differs from stored YBR.
    assert abs(int(out[0, 0, 0]) - 76) <= 2
    assert abs(int(out[0, 0, 1]) - 76) <= 2


def test_to_display_ybr_422_treated_as_full():
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    frame[..., 0] = 76
    frame[..., 1] = 128
    frame[..., 2] = 128
    out, cmap = to_display(frame, "YBR_FULL_422")
    assert cmap is None
    assert abs(int(out[0, 0, 0]) - 76) <= 2


def test_to_display_monochrome_colormaps():
    frame = np.zeros((4, 4), dtype=np.uint8)
    assert to_display(frame, "MONOCHROME2")[1] == "gray"
    assert to_display(frame, "MONOCHROME1")[1] == "gray_r"  # inverted for display


def test_to_display_palette_color_raises():
    with pytest.raises(ViewerError):
        to_display(np.zeros((4, 4), dtype=np.uint8), "PALETTE COLOR")


# --- export_pdf (Agg backend) -----------------------------------------------

def test_export_pdf_2d_single_page(tmp_path):
    arr = np.full((30, 40, 3), 180, dtype=np.uint8)
    frames = load_frames(_make_ds(arr, "RGB", 3))
    out = tmp_path / "one.pdf"

    pages = export_pdf(frames, "RGB", str(out))

    assert pages == 1
    assert out.stat().st_size > 0
    n = _pdf_page_count(out)
    assert n in (1, -1)


def test_export_pdf_cineloop_contact_sheet_pagination(tmp_path):
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 255, size=(32, 20, 20, 3), dtype=np.uint8)
    frames = load_frames(_make_ds(arr, "RGB", 3, frames=32))
    out = tmp_path / "sheet.pdf"

    # 3 cols x 6 rows = 18 per page -> ceil(32/18) = 2 pages
    pages = export_pdf(frames, "RGB", str(out), sheet_cols=3, sheet_rows=6)

    assert pages == 2
    n = _pdf_page_count(out)
    assert n in (2, -1)


def test_export_pdf_cineloop_single_frame_when_frame_given(tmp_path):
    rng = np.random.default_rng(2)
    arr = rng.integers(0, 255, size=(32, 20, 20, 3), dtype=np.uint8)
    frames = load_frames(_make_ds(arr, "RGB", 3, frames=32))
    out = tmp_path / "one_frame.pdf"

    pages = export_pdf(frames, "RGB", str(out), frame=10)

    assert pages == 1


# --- CLI dispatch error paths -----------------------------------------------

def _write(ds: Dataset, path) -> str:
    ds.save_as(str(path), enforce_file_format=True)
    return str(path)


def test_run_view_missing_file_returns_2(tmp_path):
    from dicom_phi_scan.cli import _run_view

    rc = _run_view(str(tmp_path / "nope.dcm"), frame=None, pdf=None, sheet_cols=5)
    assert rc == 2


def test_run_view_no_pixel_data_returns_2(tmp_path):
    from dicom_phi_scan.cli import _run_view

    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.PatientName = "TEST"
    src = _write(ds, tmp_path / "nopix.dcm")

    rc = _run_view(src, frame=None, pdf=None, sheet_cols=5)
    assert rc == 2


def test_run_view_interactive_without_display_returns_2(tmp_path, monkeypatch):
    from dicom_phi_scan.cli import _run_view

    arr = np.full((20, 20, 3), 120, dtype=np.uint8)
    src = _write(_make_ds(arr, "RGB", 3), tmp_path / "img.dcm")
    monkeypatch.delenv("DISPLAY", raising=False)

    rc = _run_view(src, frame=None, pdf=None, sheet_cols=5)
    assert rc == 2  # no display, no --pdf


def test_run_view_pdf_export_works_headless(tmp_path, monkeypatch):
    from dicom_phi_scan.cli import _run_view

    arr = np.full((20, 20, 3), 120, dtype=np.uint8)
    src = _write(_make_ds(arr, "RGB", 3), tmp_path / "img.dcm")
    out = tmp_path / "img.pdf"
    monkeypatch.delenv("DISPLAY", raising=False)  # PDF path must not need a display

    rc = _run_view(src, frame=None, pdf=str(out), sheet_cols=5)
    assert rc == 0
    assert out.stat().st_size > 0
