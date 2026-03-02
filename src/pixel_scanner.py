"""Layer 2: Pixel matrix PHI detection via OCR.

Extracts pixel data to images, runs pytesseract OCR for text with bounding boxes,
and flags all detected text as potential PHI (burned-in text is inherently suspicious).
"""

import logging

import numpy as np
import pytesseract
from PIL import Image
from pydicom.dataset import Dataset

from .models import BoundingBox, PixelPHIFinding, Severity

logger = logging.getLogger(__name__)


def extract_image(ds: Dataset) -> Image.Image | None:
    """Extract pixel data from a DICOM dataset as a PIL Image.

    Returns None if the dataset has no pixel data.
    """
    if not hasattr(ds, "PixelData"):
        return None

    try:
        pixel_array = ds.pixel_array
    except Exception:
        logger.warning("Could not decompress pixel data — skipping pixel analysis")
        return None

    # Handle multi-frame (take first frame)
    num_frames = getattr(ds, "NumberOfFrames", 1)
    if int(num_frames) > 1 and len(pixel_array.shape) >= 3:
        pixel_array = pixel_array[0]

    # Handle RGB/RGBA (keep as-is for PIL)
    # Only grayscale needs normalization
    if len(pixel_array.shape) == 2 or (
        len(pixel_array.shape) == 3 and pixel_array.shape[2] in (3, 4)
    ):
        pass  # 2D grayscale or RGB/RGBA — valid for PIL

    # Normalize to 8-bit for OCR
    pmin, pmax = pixel_array.min(), pixel_array.max()
    if pmax > 255:
        if pmax == pmin:
            pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
        else:
            pixel_array = ((pixel_array - pmin) / (pmax - pmin) * 255).astype(np.uint8)

    return Image.fromarray(pixel_array)


def run_ocr(image: Image.Image) -> list[dict]:
    """Run pytesseract OCR and return text with bounding boxes.

    Returns:
        List of dicts with keys: text, x, y, width, height, conf
    """
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    results = []
    n_boxes = len(ocr_data["text"])
    for i in range(n_boxes):
        text = ocr_data["text"][i].strip()
        conf = int(ocr_data["conf"][i])
        if text and conf > 30:  # Filter low-confidence noise
            results.append(
                {
                    "text": text,
                    "x": ocr_data["left"][i],
                    "y": ocr_data["top"][i],
                    "width": ocr_data["width"][i],
                    "height": ocr_data["height"][i],
                    "conf": conf,
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
                confidence=1.0,
                severity=Severity.HIGH,
            )
        )

    return findings
