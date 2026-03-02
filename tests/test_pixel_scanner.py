"""Tests for pixel PHI scanner."""

from unittest.mock import PropertyMock, patch

import numpy as np
from pydicom.dataset import Dataset

from src.pixel_scanner import extract_image


# --- extract_image tests ---


def _ds_with_pixel_array(pixel_array: np.ndarray, **attrs) -> Dataset:
    """Create a Dataset that has PixelData and a mocked pixel_array property."""
    ds = Dataset()
    ds.PixelData = b"\x00"  # Must exist so hasattr check passes
    for k, v in attrs.items():
        setattr(ds, k, v)
    # Patch pixel_array to return our array directly
    ds._pixel_array_for_test = pixel_array
    return ds


def _extract_with_mock(ds: Dataset):
    """Call extract_image with pixel_array mocked on the dataset."""
    arr = ds._pixel_array_for_test
    with patch.object(type(ds), "pixel_array", new_callable=PropertyMock, return_value=arr):
        return extract_image(ds)


def test_extract_image_no_pixel_data():
    ds = Dataset()
    assert extract_image(ds) is None


def test_extract_image_basic_8bit():
    arr = np.full((64, 64), 128, dtype=np.uint8)
    ds = _ds_with_pixel_array(arr)
    img = _extract_with_mock(ds)
    assert img is not None
    assert img.size == (64, 64)


def test_extract_image_16bit_normalization():
    arr = np.array([[0, 1000], [2000, 4095]], dtype=np.uint16)
    ds = _ds_with_pixel_array(arr)
    img = _extract_with_mock(ds)
    assert img is not None
    pixel_arr = np.array(img)
    assert pixel_arr.max() == 255
    assert pixel_arr.min() == 0


def test_extract_image_uniform_no_div_by_zero():
    """Bug 2 regression: uniform images should not cause division by zero."""
    arr = np.full((64, 64), 1000, dtype=np.uint16)
    ds = _ds_with_pixel_array(arr)
    img = _extract_with_mock(ds)
    assert img is not None
    pixel_arr = np.array(img)
    assert pixel_arr.max() == 0  # All zeros for uniform image


def test_extract_image_multiframe():
    """Bug 1 regression: multi-frame images should extract first frame correctly."""
    frames = np.random.randint(0, 255, (3, 32, 32), dtype=np.uint8)
    ds = _ds_with_pixel_array(frames, NumberOfFrames=3)
    img = _extract_with_mock(ds)
    assert img is not None
    assert img.size == (32, 32)
