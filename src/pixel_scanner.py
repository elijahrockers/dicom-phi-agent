"""Layer 2: Pixel matrix PHI detection via OCR.

Extracts pixel data to images, runs EasyOCR for text with bounding boxes,
and flags all detected text as potential PHI (burned-in text is inherently suspicious).
"""

import logging

import easyocr
import numpy as np
from PIL import Image
from pydicom.dataset import Dataset

from .models import BoundingBox, PixelPHIFinding, Severity

logger = logging.getLogger(__name__)

# Minimum EasyOCR confidence (0.0-1.0) to accept an OCR result.
# Below this threshold, detections are treated as noise and discarded.
MIN_OCR_CONFIDENCE = 0.30

# Lazy singleton EasyOCR reader — avoids reloading ~100MB model per call.
_reader: easyocr.Reader | None = None


def _get_reader() -> easyocr.Reader:
    """Return a shared EasyOCR Reader, creating it on first use."""
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def extract_image(ds: Dataset) -> Image.Image | None:
    """Extract pixel data from a DICOM dataset as a PIL Image.

    Returns None if the dataset has no pixel data.
    """
    if not hasattr(ds, "PixelData"):
        return None

    try:
        pixel_array = ds.pixel_array
    except (RuntimeError, NotImplementedError, ValueError):
        logger.warning("Could not decompress pixel data — skipping pixel analysis")
        return None

    # Handle multi-frame (take first frame)
    num_frames = getattr(ds, "NumberOfFrames", 1)
    if int(num_frames) > 1 and len(pixel_array.shape) >= 3:
        pixel_array = pixel_array[0]

    # 2D grayscale and RGB/RGBA arrays are valid for PIL as-is.
    # Normalize to 8-bit for OCR
    pmin, pmax = pixel_array.min(), pixel_array.max()
    if pmax > 255:
        if pmax == pmin:
            pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
        else:
            pixel_array = ((pixel_array - pmin) / (pmax - pmin) * 255).astype(np.uint8)

    return Image.fromarray(pixel_array)


def run_ocr(image: Image.Image) -> list[dict]:
    """Run EasyOCR and return text with bounding boxes.

    Returns:
        List of dicts with keys: text, x, y, width, height, conf (conf 0-100)
    """
    reader = _get_reader()
    ocr_data = reader.readtext(np.array(image))

    results = []
    for bbox, text, conf in ocr_data:
        text = text.strip()
        if text and conf > MIN_OCR_CONFIDENCE:
            # bbox is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] — convert to {x, y, width, height}
            xs = [pt[0] for pt in bbox]
            ys = [pt[1] for pt in bbox]
            x = int(min(xs))
            y = int(min(ys))
            width = int(max(xs) - x)
            height = int(max(ys) - y)

            results.append(
                {
                    "text": text,
                    "x": x,
                    "y": y,
                    "width": width,
                    "height": height,
                    "conf": int(conf * 100),
                }
            )

    return results


def scan_pixels(ds: Dataset) -> list[PixelPHIFinding]:
    """Full pixel PHI scan: extract image → OCR → flag all text as PHI.

    Any burned-in text detected by OCR is flagged as potential PHI.

    Args:
        ds: A pydicom Dataset (must include pixel data).

    Returns:
        List of PixelPHIFinding for detected text.
    """
    image = extract_image(ds)
    if image is None:
        return []

    ocr_results = run_ocr(image)
    if not ocr_results:
        return []

    findings: list[PixelPHIFinding] = []
    for ocr_match in ocr_results:
        findings.append(
            PixelPHIFinding(
                text=ocr_match["text"],
                bbox=BoundingBox(
                    x=ocr_match["x"],
                    y=ocr_match["y"],
                    width=ocr_match["width"],
                    height=ocr_match["height"],
                ),
                phi_type="ocr_detected",
                confidence=ocr_match["conf"] / 100.0,
                severity=Severity.HIGH,
            )
        )

    return findings
