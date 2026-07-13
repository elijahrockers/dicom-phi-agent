"""Pydantic models for PHI detection results."""

from enum import Enum
from pydantic import BaseModel


class Severity(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PHITagFinding(BaseModel):
    """A PHI finding from DICOM header tag analysis."""

    tag: str
    tag_name: str
    value: str
    severity: Severity
    hipaa_category: str


class BoundingBox(BaseModel):
    """Pixel coordinates for detected text region."""

    x: int
    y: int
    width: int
    height: int


class PixelPHIFinding(BaseModel):
    """A PHI finding from burned-in pixel text."""

    text: str
    bbox: BoundingBox
    phi_type: str
    confidence: float
    severity: Severity


class ScanReport(BaseModel):
    """Complete PHI scan report for a DICOM file."""

    filepath: str
    tag_findings: list[PHITagFinding]
    pixel_findings: list[PixelPHIFinding]
    burned_in_annotation_tag_present: bool
    burned_in_annotation_value: str | None
    total_phi_count: int
    risk_level: Severity
    recommendations: list[str]

    @property
    def has_phi(self) -> bool:
        return self.total_phi_count > 0


class FileError(BaseModel):
    """A per-file error encountered during batch scanning."""

    filepath: str
    error: str


class RedactionResult(BaseModel):
    """Outcome of redacting (or copying) a single DICOM file."""

    input_path: str
    output_path: str
    status: str  # "redacted" | "copied" | "omitted" | "error"
    frames: int
    banner_height: int
    photometric_in: str | None
    photometric_out: str | None
    transfer_syntax_out: str | None
    message: str | None = None


