"""Preview a single DICOM file's pixel data — 2D or 3D (cineloop).

Opens a matplotlib GUI window at native resolution (zoom/pan preserve burned-in text
detail), or exports to PDF for a headless artifact. Decoding and color handling reuse
the redactor's native-color-space helpers so what you see matches the stored pixels:

* RGB                -> shown as-is
* YBR_FULL / _422    -> converted to RGB (422 is treated as full-res YBR_FULL)
* MONOCHROME2        -> grayscale
* MONOCHROME1        -> inverted grayscale (max = black), so polarity is correct

matplotlib is an optional dependency (``pip install '.[viz]'``); it is imported lazily
by the display/export functions so decoding logic stays importable without it.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from pydicom.dataset import Dataset

from .redactor import _DECODE_ERRORS, _convert_ybr_to_rgb, _decode_pixels

logger = logging.getLogger(__name__)


class ViewerError(Exception):
    """Raised when a file cannot be previewed."""


def load_frames(ds: Dataset) -> np.ndarray:
    """Decode all frames in native color space as ``(F, H, W[, C])`` with ``F >= 1``.

    A frame axis is prepended for single-frame files so 2D and cineloop share one shape
    contract. Raises ViewerError on decode failure (e.g. a compressed cine with no codec).
    """
    try:
        arr = _decode_pixels(ds)
    except _DECODE_ERRORS as e:
        raise ViewerError(
            f"could not decode pixel data: {e} — a compressed cine loop needs the codec "
            "plugins (pip install '.[codecs]')"
        ) from e

    samples = int(getattr(ds, "SamplesPerPixel", 1))
    # Color arrays are (H,W,C) / (F,H,W,C); mono are (H,W) / (F,H,W).
    single_ndim = 3 if samples > 1 else 2
    if arr.ndim == single_ndim:
        arr = arr[np.newaxis, ...]
    return arr


def to_display(frame: np.ndarray, photometric: str) -> tuple[np.ndarray, str | None]:
    """Return ``(array, cmap)`` for one frame, ready for ``imshow``.

    ``cmap`` is None for color (RGB) data; a grayscale colormap for monochrome. imshow
    autoscales vmin/vmax to the data range, preserving detail for >8-bit pixels.
    """
    pi = (photometric or "").upper()
    if pi == "PALETTE COLOR":
        raise ViewerError("PALETTE COLOR display is not supported (needs the color LUT)")
    if pi.startswith("YBR"):
        return _convert_ybr_to_rgb(frame), None
    if pi == "RGB":
        return frame, None
    if pi == "MONOCHROME1":
        return frame, "gray_r"  # inverted LUT: max = black
    return frame, "gray"  # MONOCHROME2 and other grayscale


def format_header(ds: Dataset) -> str:
    """One-block human summary of the file's pixel/geometry metadata."""
    ts = getattr(getattr(ds, "file_meta", None), "TransferSyntaxUID", "?")
    return (
        f"  Dimensions:  {getattr(ds, 'Rows', '?')} x {getattr(ds, 'Columns', '?')}\n"
        f"  Frames:      {int(getattr(ds, 'NumberOfFrames', 1))}\n"
        f"  Photometric: {getattr(ds, 'PhotometricInterpretation', '?')}\n"
        f"  Modality:    {getattr(ds, 'Modality', '?')}\n"
        f"  BitsStored:  {getattr(ds, 'BitsStored', '?')}\n"
        f"  Transfer:    {ts}"
    )


def _clamp_frame(i: int, n: int) -> int:
    if i < 0:
        logger.warning("frame %d < 0 — clamping to 0", i)
        return 0
    if i >= n:
        logger.warning("frame %d out of range [0, %d] — clamping", i, n - 1)
        return n - 1
    return i


def show_interactive(
    frames: np.ndarray, photometric: str, *, start_frame: int = 0, title: str = ""
) -> None:
    """Open a GUI window. 2D: a single zoomable image. Cineloop: a frame slider
    (left/right arrows step, space plays/pauses). Requires a GUI backend + display;
    the caller is responsible for checking ``$DISPLAY`` first.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    n = int(frames.shape[0])
    start = _clamp_frame(start_frame, n)

    fig, ax = plt.subplots()
    disp0, cmap = to_display(frames[start], photometric)
    im = ax.imshow(disp0, cmap=cmap)
    ax.set_axis_off()

    def render(i: int) -> None:
        disp, _ = to_display(frames[i], photometric)
        im.set_data(disp)
        ax.set_title(f"{title}  frame {i + 1}/{n}  [{photometric}]")
        fig.canvas.draw_idle()

    render(start)

    if n > 1:
        fig.subplots_adjust(bottom=0.16)
        slider_ax = fig.add_axes((0.15, 0.05, 0.70, 0.03))
        slider = Slider(slider_ax, "frame", 0, n - 1, valinit=start, valstep=1)
        slider.on_changed(lambda v: render(int(v)))

        state = {"playing": False}
        timer = fig.canvas.new_timer(interval=66)  # ~15 fps
        timer.add_callback(lambda: slider.set_val((int(slider.val) + 1) % n))

        def on_key(event) -> None:
            if event.key == "right":
                slider.set_val((int(slider.val) + 1) % n)
            elif event.key == "left":
                slider.set_val((int(slider.val) - 1) % n)
            elif event.key == " ":
                timer.stop() if state["playing"] else timer.start()
                state["playing"] = not state["playing"]

        fig.canvas.mpl_connect("key_press_event", on_key)
        # Keep widget/timer refs alive for the lifetime of the figure.
        fig._viewer_refs = (slider, timer)  # type: ignore[attr-defined]

    plt.show()


def export_pdf(
    frames: np.ndarray,
    photometric: str,
    path: str,
    *,
    frame: int | None = None,
    sheet_cols: int = 5,
    sheet_rows: int = 6,
    title: str = "",
) -> int:
    """Render to ``path`` (headless, Agg backend). Returns the number of pages written.

    A single-frame file — or any file when ``frame`` is given — is one full-page image.
    A cineloop is a contact sheet: ``sheet_cols * sheet_rows`` thumbnails per page,
    paginated over all frames, each captioned with its frame index.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    n = int(frames.shape[0])
    pages = 0
    with PdfPages(path) as pdf:
        if n == 1 or frame is not None:
            idx = 0 if n == 1 else _clamp_frame(frame, n)
            disp, cmap = to_display(frames[idx], photometric)
            fig, ax = plt.subplots()
            ax.imshow(disp, cmap=cmap)
            ax.set_axis_off()
            ax.set_title(f"{title}  frame {idx + 1}/{n}  [{photometric}]")
            pdf.savefig(fig)
            plt.close(fig)
            pages = 1
        else:
            per_page = sheet_cols * sheet_rows
            pages = math.ceil(n / per_page)
            for p in range(pages):
                fig, axes = plt.subplots(sheet_rows, sheet_cols)
                for k, ax in enumerate(np.atleast_1d(axes).ravel()):
                    ax.set_axis_off()
                    fi = p * per_page + k
                    if fi < n:
                        disp, cmap = to_display(frames[fi], photometric)
                        ax.imshow(disp, cmap=cmap)
                        ax.set_title(str(fi), fontsize=5)
                last = min((p + 1) * per_page, n) - 1
                fig.suptitle(
                    f"{title}  frames {p * per_page}-{last} of {n}  [{photometric}]",
                    fontsize=8,
                )
                pdf.savefig(fig)
                plt.close(fig)
    return pages
