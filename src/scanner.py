"""Direct DICOM PHI scanning pipeline.

Two-layer scan: header tag analysis followed by conditional OCR pixel
inspection.
"""

import logging

import pydicom

from .models import Severity, ScanReport
from .tag_scanner import scan_tags, get_burned_in_annotation
from .pixel_scanner import scan_pixels

logger = logging.getLogger(__name__)


def scan_file(filepath: str) -> ScanReport:
    """Scan a DICOM file for PHI in headers and pixel data.

    Runs header tag scan first, then conditionally runs pixel OCR scan
    if BurnedInAnnotation is YES or missing.

    Args:
        filepath: Path to the DICOM file to scan.

    Returns:
        ScanReport with all findings and recommendations.
    """
    ds = pydicom.dcmread(filepath)
    tag_findings = scan_tags(ds)
    bia_present, bia_value = get_burned_in_annotation(ds)

    pixel_findings = []
    if not bia_present or (bia_value and bia_value.upper() == "YES"):
        pixel_findings = scan_pixels(ds)
        # Free pixel data — no longer needed, findings are extracted
        try:
            del ds.PixelData
        except AttributeError:
            pass

    total = len(tag_findings) + len(pixel_findings)
    high_count = sum(1 for f in tag_findings if f.severity == Severity.HIGH) + sum(
        1 for f in pixel_findings if f.severity == Severity.HIGH
    )

    recommendations = []
    if tag_findings:
        recommendations.append("Remove or redact PHI from DICOM header tags before sharing")
    if pixel_findings:
        recommendations.append(
            "Redact burned-in PHI text from pixel data at identified bounding box regions"
        )
    if not bia_present:
        recommendations.append(
            "BurnedInAnnotation tag (0028,0301) is missing — add it for compliance"
        )
    if not tag_findings and not pixel_findings:
        recommendations.append("No PHI detected — file appears safe for sharing")

    if high_count > 0:
        risk_level = Severity.HIGH
    elif total > 0:
        risk_level = Severity.MEDIUM
    else:
        risk_level = Severity.LOW

    return ScanReport(
        filepath=filepath,
        tag_findings=tag_findings,
        pixel_findings=pixel_findings,
        burned_in_annotation_tag_present=bia_present,
        burned_in_annotation_value=bia_value,
        total_phi_count=total,
        risk_level=risk_level,
        recommendations=recommendations,
    )
