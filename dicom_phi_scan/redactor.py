"""Banner redaction: black out the top header banner of ultrasound pixel data.

Writes redacted *copies* — originals are never opened for writing. Uses an
"auto method" per file:

* Uncompressed originals   -> re-serialize the (modified) pixel array; outside the
                              banner the bytes are identical, so no quality is lost.
* Compressed originals     -> decode -> black out banner -> re-encode per ``compress``.
                              JPEG decode is deterministic, so untouched pixels are
                              pixel-identical to the original.

See the module-level plan for the full write-back contract.
"""

from __future__ import annotations

import logging
import shutil

import numpy as np
import pydicom
from pydicom.dataset import Dataset

from .models import RedactionResult

logger = logging.getLogger(__name__)

# Decode failures raised by pydicom's pixel handlers (mirrors pixel_scanner).
_DECODE_ERRORS = (RuntimeError, NotImplementedError, ValueError)

_PYDICOM_MAJOR = int(pydicom.__version__.split(".")[0])

# --compress value -> transfer syntax UID (None = uncompressed Explicit VR LE).
COMPRESS_CHOICES = ("none", "rle", "jpeg-ls", "jpeg2000", "jpeg")
_COMPRESS_UID = {
    "rle": pydicom.uid.RLELossless,
    "jpeg-ls": pydicom.uid.JPEGLSLossless,
    "jpeg2000": pydicom.uid.JPEG2000Lossless,
    "jpeg": pydicom.uid.JPEGBaseline8Bit,  # lossy
}


class RedactionError(Exception):
    """Raised when a file cannot be redacted.

    ``redact_file`` catches this and routes the file through ``on_unredactable``
    (copy the original through, or omit it), recording the message as the reason.
    """


def _convert_ybr_to_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert a full-resolution YBR_FULL array to RGB (version-defensive import)."""
    try:
        from pydicom.pixels import convert_color_space
    except ImportError:  # pydicom < 3.0
        from pydicom.pixel_data_handlers.util import convert_color_space
    return convert_color_space(arr, "YBR_FULL", "RGB")


def _decode_pixels(ds: Dataset) -> np.ndarray:
    """Decode pixel data in the file's NATIVE color space (no color conversion).

    pydicom 3.0's ``ds.pixel_array`` auto-converts YBR->RGB, which would leave the
    decoded array out of sync with PhotometricInterpretation. We must keep them
    aligned so the banner fill and the output label are consistent, so we request
    the raw (un-converted) array. pydicom 2.x's ``pixel_array`` is already raw.
    """
    if _PYDICOM_MAJOR >= 3:
        from pydicom.pixels import pixel_array
        return pixel_array(ds, raw=True)
    return ds.pixel_array


def resolve_banner_height(ds: Dataset, override: int | None) -> int:
    """Return the number of top rows to black out.

    ``override`` (the --banner-height flag) wins if given. Otherwise derive it from
    the topmost ultrasound region: the smallest RegionLocationMinY0 across
    (0018,6011) Sequence of Ultrasound Regions — everything above the first image
    region is banner. Raises RedactionError if neither source is available.
    """
    if override is not None:
        return override

    regions = getattr(ds, "SequenceOfUltrasoundRegions", None)
    if regions:
        tops = [
            int(r.RegionLocationMinY0)
            for r in regions
            if getattr(r, "RegionLocationMinY0", None) is not None
        ]
        if tops:
            return min(tops)

    raise RedactionError(
        "cannot determine banner height: no (0018,6011) Sequence of Ultrasound "
        "Regions present — pass --banner-height N explicitly"
    )


def _output_photometric(photometric: str, to_rgb: bool) -> str:
    """PhotometricInterpretation to write, given the input PI and --to-rgb.

    YBR_FULL_422 is invalid for full-resolution uncompressed/native data, so it is
    always relabeled to YBR_FULL when color is kept; --to-rgb maps any YBR to RGB.
    """
    pi = (photometric or "").upper()
    if to_rgb and pi.startswith("YBR"):
        return "RGB"
    if pi == "YBR_FULL_422":
        return "YBR_FULL"
    return photometric


def _fill_value(photometric: str, bits_stored: int) -> int | tuple[int, int, int]:
    """Return the pixel value that renders as black for a photometric interpretation."""
    pi = (photometric or "").upper()
    center = 1 << (bits_stored - 1)  # 128 for 8-bit chroma
    if pi.startswith("YBR"):
        return (0, center, center)  # Y=0 (black), chroma centered
    if pi == "RGB":
        return (0, 0, 0)
    if pi == "MONOCHROME1":
        return (1 << bits_stored) - 1  # inverted LUT: max = black
    # MONOCHROME2 and anything else grayscale-ish
    return 0


def redact_dataset(ds: Dataset, banner_height: int, *, to_rgb: bool = False) -> int:
    """Black out rows ``[0:banner_height]`` across ALL frames, in memory.

    Also applies the write-back metadata contract so the dataset can be saved
    uncompressed or handed to ``ds.compress``. Returns the effective banner height
    actually applied (may be clamped to the image height).
    """
    photometric = str(getattr(ds, "PhotometricInterpretation", "") or "")
    if photometric.upper() == "PALETTE COLOR":
        raise RedactionError(
            "PALETTE COLOR is not supported (black is LUT-dependent)"
        )
    if banner_height <= 0:
        raise RedactionError(f"banner_height must be positive, got {banner_height}")

    try:
        arr = _decode_pixels(ds)
    except _DECODE_ERRORS as e:
        raise RedactionError(f"could not decode pixel data: {e}") from e

    rows = int(ds.Rows)
    bh = banner_height
    if bh >= rows:
        logger.warning(
            "banner_height %d >= image height %d — blacking out the entire frame",
            bh, rows,
        )
        bh = rows

    samples = int(getattr(ds, "SamplesPerPixel", 1))
    bits_stored = int(getattr(ds, "BitsStored", 8))
    fill = _fill_value(photometric, bits_stored)

    # Row axis: color arrays are (..., rows, cols, samples); mono are (..., rows, cols).
    row_axis = arr.ndim - 3 if samples > 1 else arr.ndim - 2
    banner_slice = [slice(None)] * arr.ndim
    banner_slice[row_axis] = slice(0, bh)
    arr[tuple(banner_slice)] = fill

    # --- write-back contract ---
    photometric_out = _output_photometric(photometric, to_rgb)
    if photometric_out == "RGB" and photometric.upper().startswith("YBR"):
        arr = _convert_ybr_to_rgb(arr)

    arr = np.ascontiguousarray(arr)
    ds.PixelData = arr.tobytes()
    ds["PixelData"].is_undefined_length = False  # drop any encapsulation flag
    ds.PhotometricInterpretation = photometric_out
    if samples > 1:
        ds.PlanarConfiguration = 0  # pixel_array is always interleaved

    return bh


def _apply_transfer_syntax(ds: Dataset, compress: str, was_compressed: bool) -> str:
    """Set the output transfer syntax / re-encode. Returns the resulting TS UID."""
    if compress == "none":
        if was_compressed:
            # We decompressed; pixels are now native little-endian.
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        # Uncompressed input keeps its original (already-native) transfer syntax.
        return str(ds.file_meta.TransferSyntaxUID)

    # Normalize to native first so pydicom treats current PixelData as raw input,
    # then let ds.compress own the resulting (encapsulated) transfer syntax.
    ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    uid = _COMPRESS_UID[compress]
    try:
        ds.compress(uid)
    except _DECODE_ERRORS as e:
        raise RedactionError(
            f"re-compression as {compress} failed (is the encoder plugin installed?): {e}"
        ) from e
    return str(ds.file_meta.TransferSyntaxUID)


def _save(ds: Dataset, output_path: str) -> None:
    """Write the dataset, defensively across pydicom 2.x/3.x save signatures."""
    if _PYDICOM_MAJOR >= 3:
        ds.save_as(output_path, enforce_file_format=True)
        return
    ts = ds.file_meta.TransferSyntaxUID
    ds.is_little_endian = ts.is_little_endian
    ds.is_implicit_VR = ts.is_implicit_VR
    ds.save_as(output_path, write_like_original=False)


ON_UNREDACTABLE_CHOICES = ("copy", "omit")


def redact_file(
    input_path: str,
    output_path: str,
    *,
    banner_height: int | None = None,
    compress: str = "none",
    to_rgb: bool = False,
    on_unredactable: str = "omit",
) -> RedactionResult:
    """Read the original (read-only) and write a redacted copy to ``output_path``.

    The input file is never opened for writing. Any file that cannot be redacted for
    any reason — no PixelData, PALETTE COLOR, an undeterminable banner height, or a
    decode/encode failure — is handled per ``on_unredactable``:

    * ``"copy"`` -> the original bytes are copied through to ``output_path`` (the output
      series stays complete, but may ship un-redacted pixels); status ``"copied"``.
    * ``"omit"`` -> nothing is written for that file; status ``"omitted"``.

    Either way the reason is recorded in ``message``. Never raises for per-file problems
    (only for programmer errors like an invalid option) — outcomes are returned as a
    RedactionResult.
    """
    if compress not in COMPRESS_CHOICES:
        raise RedactionError(f"unknown compress option {compress!r}")
    if on_unredactable not in ON_UNREDACTABLE_CHOICES:
        raise RedactionError(f"unknown on_unredactable option {on_unredactable!r}")

    ds = pydicom.dcmread(input_path)  # fresh in-memory object; disk is untouched
    photometric_in = getattr(ds, "PhotometricInterpretation", None)
    ts_in = str(getattr(ds.file_meta, "TransferSyntaxUID", None))

    def _result(status: str, **kw) -> RedactionResult:
        base = dict(
            input_path=input_path, output_path=output_path, status=status,
            frames=int(getattr(ds, "NumberOfFrames", 1)), banner_height=0,
            photometric_in=photometric_in, photometric_out=None,
            transfer_syntax_out=None,
        )
        base.update(kw)
        return RedactionResult(**base)

    def _unredactable(reason: str, **kw) -> RedactionResult:
        """Copy the original through or omit it entirely, per ``on_unredactable``."""
        if on_unredactable == "copy":
            shutil.copy2(input_path, output_path)
            return _result(
                "copied", photometric_out=photometric_in, transfer_syntax_out=ts_in,
                message=f"not redacted ({reason}) — copied unchanged", **kw,
            )
        return _result("omitted", message=f"not redacted ({reason}) — omitted", **kw)

    if "PixelData" not in ds:
        return _unredactable("no PixelData", frames=0)

    was_compressed = bool(ds.file_meta.TransferSyntaxUID.is_compressed)

    try:
        bh = resolve_banner_height(ds, banner_height)
        applied_bh = redact_dataset(ds, bh, to_rgb=to_rgb)
        photometric_out = str(ds.PhotometricInterpretation)
        ts_out = _apply_transfer_syntax(ds, compress, was_compressed)
        _save(ds, output_path)
    except RedactionError as e:
        return _unredactable(str(e))

    return _result(
        "redacted", banner_height=applied_bh, photometric_out=photometric_out,
        transfer_syntax_out=ts_out,
    )
