"""FastAPI server for DICOM PHI scanning."""

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException

from .scanner import scan_file
from .models import ScanReport

app = FastAPI(
    title="DICOM PHI Scanner",
    description="Two-layer pipeline for scanning DICOM datasets for PHI",
    version="0.1.0",
)


@app.post("/scan", response_model=ScanReport)
async def scan_dicom(
    file: UploadFile = File(...),
):
    """Upload and scan a DICOM file for PHI."""
    if not file.filename or not file.filename.lower().endswith((".dcm", ".dicom")):
        raise HTTPException(400, "File must be a DICOM file (.dcm)")

    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        report = scan_file(tmp_path)
        return report
    except Exception as e:
        raise HTTPException(500, f"Scan failed: {str(e)}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/health")
async def health():
    return {"status": "ok"}
