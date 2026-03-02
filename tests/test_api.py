"""Tests for FastAPI endpoints."""

import io

import numpy as np
import pydicom
import pytest
from pydicom.dataset import FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from fastapi.testclient import TestClient

from src.api import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_no_api_key(client):
    """Bug 6 regression: /health must work without an API key."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_scan_rejects_non_dicom(client):
    """Non-DICOM files should return 400."""
    file_content = b"not a dicom file"
    response = client.post(
        "/scan",
        files={"file": ("test.txt", io.BytesIO(file_content), "application/octet-stream")},
    )
    assert response.status_code == 400


def _make_dicom_bytes() -> bytes:
    """Create a minimal valid DICOM file in memory."""
    buf = io.BytesIO()
    file_meta = pydicom.dataset.FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

    ds = FileDataset(buf, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.PatientName = "TEST^PATIENT"
    ds.PatientID = "MRN-TEST"
    ds.BurnedInAnnotation = "NO"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.Rows = 8
    ds.Columns = 8
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = np.zeros((8, 8), dtype=np.uint8).tobytes()
    ds.save_as(buf)
    buf.seek(0)
    return buf.read()


def test_scan_returns_valid_report(client):
    """POST /scan with a valid DICOM should return a scan report."""
    dicom_bytes = _make_dicom_bytes()
    response = client.post(
        "/scan",
        files={"file": ("test.dcm", io.BytesIO(dicom_bytes), "application/dicom")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "filepath" in data
    assert "tag_findings" in data
    assert "risk_level" in data
