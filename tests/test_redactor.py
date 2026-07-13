"""Tests for banner redaction (dicom_phi_scan.redactor).

Synthetic in-memory DICOM datasets only — no external data. The compressed
YBR_FULL_422 cine path (JPEG) requires an encoder/decoder plugin and real data,
so it is verified manually on production per the plan; here we cover the
uncompressed, RLE, color-space, and edge-case logic.
"""

from __future__ import annotations

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, RLELossless, generate_uid

from dicom_phi_scan.redactor import (
    RedactionError,
    _fill_value,
    _output_photometric,
    redact_dataset,
    redact_file,
    resolve_banner_height,
)


def _make_ds(pixels: np.ndarray, photometric: str, samples: int, frames: int = 1) -> Dataset:
    """Build a minimal valid uncompressed DICOM dataset around a pixel array."""
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


def _write(ds: Dataset, path) -> str:
    ds.save_as(str(path), enforce_file_format=True)
    return str(path)


def _read_native(path: str) -> np.ndarray:
    """Read pixels in native color space (pydicom 3.x pixel_array auto-converts YBR)."""
    ds = pydicom.dcmread(path)
    from pydicom.pixels import pixel_array
    return pixel_array(ds, raw=True)


# --- pure helpers -----------------------------------------------------------

def test_fill_value_by_photometric():
    assert _fill_value("RGB", 8) == (0, 0, 0)
    assert _fill_value("YBR_FULL", 8) == (0, 128, 128)
    assert _fill_value("YBR_FULL_422", 8) == (0, 128, 128)
    assert _fill_value("MONOCHROME2", 8) == 0
    assert _fill_value("MONOCHROME1", 8) == 255
    # chroma center scales with bit depth
    assert _fill_value("YBR_FULL", 10) == (0, 512, 512)


def test_output_photometric_relabel_matrix():
    assert _output_photometric("RGB", False) == "RGB"
    assert _output_photometric("YBR_FULL", False) == "YBR_FULL"
    # 422 is invalid uncompressed -> relabel to full
    assert _output_photometric("YBR_FULL_422", False) == "YBR_FULL"
    # --to-rgb maps any YBR to RGB
    assert _output_photometric("YBR_FULL_422", True) == "RGB"
    assert _output_photometric("YBR_FULL", True) == "RGB"
    assert _output_photometric("RGB", True) == "RGB"


# --- banner height resolution ----------------------------------------------

def test_resolve_banner_height_override_wins():
    ds = _make_ds(np.zeros((10, 10, 3)), "RGB", 3)
    reg = Dataset()
    reg.RegionLocationMinY0 = 42
    ds.SequenceOfUltrasoundRegions = [reg]
    assert resolve_banner_height(ds, 7) == 7


def test_resolve_banner_height_from_regions_uses_min():
    ds = _make_ds(np.zeros((10, 10, 3)), "RGB", 3)
    r1, r2 = Dataset(), Dataset()
    r1.RegionLocationMinY0 = 77
    r2.RegionLocationMinY0 = 42
    ds.SequenceOfUltrasoundRegions = [r1, r2]
    assert resolve_banner_height(ds, None) == 42


def test_resolve_banner_height_missing_raises():
    ds = _make_ds(np.zeros((10, 10, 3)), "RGB", 3)
    with pytest.raises(RedactionError):
        resolve_banner_height(ds, None)


# --- redaction: uncompressed color -----------------------------------------

def test_redact_rgb_single_frame(tmp_path):
    rng = np.random.default_rng(0)
    arr = rng.integers(1, 255, size=(60, 80, 3), dtype=np.uint8)
    src = _write(_make_ds(arr, "RGB", 3), tmp_path / "in.dcm")
    out = str(tmp_path / "out.dcm")

    result = redact_file(src, out, banner_height=10)

    assert result.status == "redacted"
    assert result.photometric_in == "RGB" and result.photometric_out == "RGB"
    o = pydicom.dcmread(out).pixel_array
    assert (o[:10] == 0).all()          # banner blacked
    assert (o[10:] == arr[10:]).all()   # rest untouched


def test_redact_rgb_multiframe_all_frames(tmp_path):
    rng = np.random.default_rng(1)
    arr = rng.integers(1, 255, size=(4, 60, 80, 3), dtype=np.uint8)
    src = _write(_make_ds(arr, "RGB", 3, frames=4), tmp_path / "in.dcm")
    out = str(tmp_path / "out.dcm")

    result = redact_file(src, out, banner_height=10)

    assert result.status == "redacted" and result.frames == 4
    o = pydicom.dcmread(out).pixel_array
    assert (o[:, :10] == 0).all()             # every frame's banner blacked
    assert (o[:, 10:] == arr[:, 10:]).all()   # rest untouched


def test_redact_ybr_full_keeps_color_and_fills_chroma_center(tmp_path):
    # YBR_FULL 4:4:4 uncompressed round-trips cleanly (unlike 422).
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    arr[..., 0] = 100  # Y
    arr[..., 1] = 128
    arr[..., 2] = 128
    src = _write(_make_ds(arr, "YBR_FULL", 3), tmp_path / "in.dcm")
    out = str(tmp_path / "out.dcm")

    result = redact_file(src, out, banner_height=8)

    assert result.photometric_out == "YBR_FULL"
    o = _read_native(out)                # native YBR values, no auto-conversion
    assert (o[:8, :, 0] == 0).all()      # luma black
    assert (o[:8, :, 1] == 128).all()    # chroma centered
    assert (o[:8, :, 2] == 128).all()
    assert (o[8:, :, 0] == 100).all()    # below unchanged (Y)
    assert (o[8:, :, 1] == 128).all()    # below unchanged (Cb)
    assert (o[8:, :, 2] == 128).all()    # below unchanged (Cr)


def test_redact_ybr_to_rgb_conversion(tmp_path):
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    arr[..., 0] = 76  # arbitrary Y
    arr[..., 1] = 128
    arr[..., 2] = 128
    src = _write(_make_ds(arr, "YBR_FULL", 3), tmp_path / "in.dcm")
    out = str(tmp_path / "out.dcm")

    result = redact_file(src, out, banner_height=8, to_rgb=True)

    assert result.photometric_out == "RGB"
    o = pydicom.dcmread(out)
    assert o.PhotometricInterpretation == "RGB"
    px = o.pixel_array  # already RGB; no conversion applied
    assert (px[:8] == 0).all()          # banner is black in RGB too
    # below: YBR (76,128,128) -> neutral gray ~76 in RGB, unchanged & non-black
    assert (px[8:] > 0).all()
    assert abs(int(px[8, 0, 0]) - 76) <= 2


# --- redaction: grayscale ---------------------------------------------------

def test_redact_monochrome2(tmp_path):
    rng = np.random.default_rng(2)
    arr = rng.integers(1, 255, size=(50, 50), dtype=np.uint8)
    src = _write(_make_ds(arr, "MONOCHROME2", 1), tmp_path / "in.dcm")
    out = str(tmp_path / "out.dcm")

    redact_file(src, out, banner_height=12)

    o = pydicom.dcmread(out).pixel_array
    assert (o[:12] == 0).all()
    assert (o[12:] == arr[12:]).all()


# --- compressed input decode branch (RLE, no plugin needed) -----------------

def test_redact_compressed_input_to_uncompressed(tmp_path):
    rng = np.random.default_rng(3)
    arr = rng.integers(1, 255, size=(40, 60, 3), dtype=np.uint8)
    ds = _make_ds(arr, "RGB", 3)
    ds.compress(RLELossless)
    src = _write(ds, tmp_path / "in.dcm")
    assert pydicom.dcmread(src).file_meta.TransferSyntaxUID.is_compressed
    out = str(tmp_path / "out.dcm")

    result = redact_file(src, out, banner_height=10, compress="none")

    assert result.status == "redacted"
    assert result.transfer_syntax_out == str(ExplicitVRLittleEndian)
    o = pydicom.dcmread(out).pixel_array
    assert (o[:10] == 0).all()
    assert (o[10:] == arr[10:]).all()


def test_redact_with_rle_output(tmp_path):
    rng = np.random.default_rng(4)
    arr = rng.integers(1, 255, size=(40, 60, 3), dtype=np.uint8)
    src = _write(_make_ds(arr, "RGB", 3), tmp_path / "in.dcm")
    out = str(tmp_path / "out.dcm")

    result = redact_file(src, out, banner_height=10, compress="rle")

    assert result.transfer_syntax_out == str(RLELossless)
    o = pydicom.dcmread(out).pixel_array
    assert (o[:10] == 0).all()
    assert (o[10:] == arr[10:]).all()


# --- edge cases -------------------------------------------------------------

def test_original_file_never_modified(tmp_path):
    rng = np.random.default_rng(5)
    arr = rng.integers(1, 255, size=(60, 80, 3), dtype=np.uint8)
    src = _write(_make_ds(arr, "RGB", 3), tmp_path / "in.dcm")
    before = (tmp_path / "in.dcm").read_bytes()
    mtime = (tmp_path / "in.dcm").stat().st_mtime_ns

    redact_file(src, str(tmp_path / "out.dcm"), banner_height=10)

    assert (tmp_path / "in.dcm").read_bytes() == before
    assert (tmp_path / "in.dcm").stat().st_mtime_ns == mtime


def test_banner_height_clamped_to_image(tmp_path):
    arr = np.full((30, 40, 3), 200, dtype=np.uint8)
    src = _write(_make_ds(arr, "RGB", 3), tmp_path / "in.dcm")
    out = str(tmp_path / "out.dcm")

    result = redact_file(src, out, banner_height=999)

    assert result.banner_height == 30  # clamped to Rows
    assert (pydicom.dcmread(out).pixel_array == 0).all()


def test_banner_height_non_positive_is_error(tmp_path):
    arr = np.zeros((30, 40, 3), dtype=np.uint8)
    ds = _make_ds(arr, "RGB", 3)
    with pytest.raises(RedactionError):
        redact_dataset(ds, 0)


def _make_no_pixel_ds() -> Dataset:
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.PatientName = "TEST"
    return ds


def test_no_pixel_data_copied_through(tmp_path):
    src = _write(_make_no_pixel_ds(), tmp_path / "in.dcm")
    before = (tmp_path / "in.dcm").read_bytes()
    out = str(tmp_path / "out.dcm")

    result = redact_file(src, out, banner_height=10, on_unredactable="copy")

    assert result.status == "copied"
    assert result.frames == 0  # benign: nothing to redact
    assert open(out, "rb").read() == before  # copied unchanged


def test_no_pixel_data_omitted(tmp_path):
    src = _write(_make_no_pixel_ds(), tmp_path / "in.dcm")
    out = str(tmp_path / "out.dcm")

    result = redact_file(src, out, banner_height=10, on_unredactable="omit")

    assert result.status == "omitted"
    assert result.frames == 0
    assert not (tmp_path / "out.dcm").exists()  # nothing written


def _make_palette_ds() -> Dataset:
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta.MediaStorageSOPClassUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SOPClassUID = ds.file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns, ds.SamplesPerPixel = 4, 4, 1
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PhotometricInterpretation = "PALETTE COLOR"
    ds.PixelData = bytes(16)
    return ds


def test_palette_color_omitted(tmp_path):
    src = _write(_make_palette_ds(), tmp_path / "in.dcm")
    out = str(tmp_path / "out.dcm")

    result = redact_file(src, out, banner_height=2, on_unredactable="omit")

    assert result.status == "omitted"
    assert "PALETTE COLOR" in (result.message or "")
    assert not (tmp_path / "out.dcm").exists()  # no bad file written


def test_palette_color_copied_through(tmp_path):
    src = _write(_make_palette_ds(), tmp_path / "in.dcm")
    before = (tmp_path / "in.dcm").read_bytes()
    out = str(tmp_path / "out.dcm")

    result = redact_file(src, out, banner_height=2, on_unredactable="copy")

    assert result.status == "copied"
    assert result.frames >= 1  # had pixels -> PHI-relevant copy
    assert open(out, "rb").read() == before  # original bytes preserved


# --- undeterminable banner height -> on_unredactable governs outcome --------

def test_undetermined_banner_height_copy(tmp_path):
    # RGB with pixels but no (0018,6011) and no override -> can't redact.
    rng = np.random.default_rng(6)
    arr = rng.integers(1, 255, size=(40, 40, 3), dtype=np.uint8)
    src = _write(_make_ds(arr, "RGB", 3), tmp_path / "in.dcm")
    before = (tmp_path / "in.dcm").read_bytes()
    out = str(tmp_path / "out.dcm")

    result = redact_file(src, out, banner_height=None, on_unredactable="copy")

    assert result.status == "copied"
    assert result.frames >= 1
    assert "0018,6011" in (result.message or "")
    assert open(out, "rb").read() == before  # copied through un-redacted


def test_undetermined_banner_height_omit(tmp_path):
    rng = np.random.default_rng(7)
    arr = rng.integers(1, 255, size=(40, 40, 3), dtype=np.uint8)
    src = _write(_make_ds(arr, "RGB", 3), tmp_path / "in.dcm")
    out = str(tmp_path / "out.dcm")

    result = redact_file(src, out, banner_height=None, on_unredactable="omit")

    assert result.status == "omitted"
    assert result.frames >= 1
    assert not (tmp_path / "out.dcm").exists()  # nothing written


def test_unknown_on_unredactable_rejected(tmp_path):
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    src = _write(_make_ds(arr, "RGB", 3), tmp_path / "in.dcm")
    with pytest.raises(RedactionError):
        redact_file(src, str(tmp_path / "out.dcm"), banner_height=2, on_unredactable="bogus")


# --- CLI batch: manifest has one line per input file ------------------------

def test_batch_manifest_one_line_per_file(tmp_path):
    import json

    from dicom_phi_scan.cli import _run_redact_batch

    in_dir = tmp_path / "in"
    in_dir.mkdir()
    out_dir = tmp_path / "out"
    manifest = tmp_path / "run.jsonl"

    # (1) redactable RGB with an ultrasound region
    a = _make_ds(np.full((40, 40, 3), 200, dtype=np.uint8), "RGB", 3)
    reg = Dataset()
    reg.RegionLocationMinY0 = 6
    a.SequenceOfUltrasoundRegions = [reg]
    _write(a, in_dir / "a.dcm")
    # (2) RGB with pixels but no region and no --banner-height -> unredactable
    _write(_make_ds(np.full((40, 40, 3), 200, dtype=np.uint8), "RGB", 3), in_dir / "b.dcm")

    rc = _run_redact_batch(
        str(in_dir), str(out_dir), False,
        banner_height=None, compress="none", to_rgb=False,
        on_unredactable="omit", manifest=str(manifest), force=False, limit=None,
    )

    lines = [json.loads(ln) for ln in manifest.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2  # one line per input file
    statuses = {row["status"] for row in lines}
    assert statuses == {"redacted", "omitted"}
    assert rc == 2  # the omitted file (with pixels) needs review
    # redacted file was written; omitted file was not
    assert (out_dir / "a.dcm").exists()
    assert not (out_dir / "b.dcm").exists()
