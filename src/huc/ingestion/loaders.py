"""Loaders: turn a scanned PDF or a photo into page images ready for OCR.

Exams are handwritten, so PDFs are always rasterized (no digital-text branch).
Large images are downscaled to the size the OCR model actually uses.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from huc.domain.models import SourceKind

IMAGE_SUFFIXES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
SUPPORTED_SUFFIXES = {".pdf", *IMAGE_SUFFIXES}
# Default longest side, in pixels, of the image sent to the OCR model (HUC_OCR_MAX_SIDE_PX).
# Claude downscales anything larger than 1568 px, so bigger images only cost upload time.
MAX_SIDE_PX = 1568


@dataclass
class Page:
    number: int  # 1-based
    image_b64: str
    media_type: str = "image/png"

    def to_content_block(self) -> dict:
        """LangChain standard base64 image content block."""
        return {"type": "image", "base64": self.image_b64, "mime_type": self.media_type}


@dataclass
class LoadedDocument:
    kind: SourceKind
    path: Path
    pages: list[Page]


def student_id_from_path(path: Path) -> str:
    """Student ID from the file name: `Perez_Gomez_Ana__scan1.pdf` -> `Perez_Gomez_Ana`.

    The scans hold only answers, so the file name is the only place the student appears.
    """
    return path.stem.split("__")[0]


def detect_kind(path: Path) -> SourceKind:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return SourceKind.PDF
    if suffix in IMAGE_SUFFIXES:
        return SourceKind.IMAGE
    raise ValueError(f"Unsupported file format: {path.name} (expected PDF, PNG or JPG)")


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _render_png(page: pymupdf.Page, zoom: float) -> bytes:
    return page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)).tobytes("png")


def _fit_zoom(page: pymupdf.Page, max_side: int) -> float:
    """Zoom that makes the page's longest side `max_side` pixels."""
    return max_side / max(page.rect.width, page.rect.height)


def load_pdf(path: Path, dpi: int = 200, max_side: int = MAX_SIDE_PX) -> LoadedDocument:
    pages: list[Page] = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            zoom = min(dpi / 72, _fit_zoom(page, max_side))
            pages.append(Page(number=i, image_b64=_b64(_render_png(page, zoom))))
    return LoadedDocument(SourceKind.PDF, path, pages)


def load_image(path: Path, max_side: int = MAX_SIDE_PX) -> LoadedDocument:
    """Keep the original bytes unless the image is larger than `max_side`."""
    pix = pymupdf.Pixmap(str(path))
    if max(pix.width, pix.height) <= max_side:
        page = Page(number=1, image_b64=_b64(path.read_bytes()),
                    media_type=IMAGE_SUFFIXES[path.suffix.lower()])
    else:
        with pymupdf.open(path) as doc:
            page = Page(number=1, image_b64=_b64(_render_png(doc[0], _fit_zoom(doc[0], max_side))))
    return LoadedDocument(SourceKind.IMAGE, path, [page])


def load_document(path: Path, dpi: int = 200, max_side: int = MAX_SIDE_PX) -> LoadedDocument:
    """Load a PDF or an image into page images no larger than `max_side` pixels."""
    if detect_kind(path) is SourceKind.PDF:
        return load_pdf(path, dpi, max_side)
    return load_image(path, max_side)
