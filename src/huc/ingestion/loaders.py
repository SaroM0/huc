"""Loaders: turn a scanned PDF or a photo into page images ready for OCR.

Exams are handwritten, so PDFs are always rasterized (no digital-text branch).
Large images are downscaled so CPU OCR stays reasonably fast.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from huc.domain.models import SourceKind

IMAGE_SUFFIXES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
SUPPORTED_SUFFIXES = {".pdf", *IMAGE_SUFFIXES}
# Longest side, in pixels, of the image sent to the OCR model. Enough for handwriting;
# larger images only make CPU inference slower.
MAX_SIDE_PX = 2000


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


def _fit_zoom(page: pymupdf.Page) -> float:
    """Zoom that makes the page's longest side MAX_SIDE_PX pixels."""
    return MAX_SIDE_PX / max(page.rect.width, page.rect.height)


def load_pdf(path: Path, dpi: int = 200) -> LoadedDocument:
    pages: list[Page] = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            zoom = min(dpi / 72, _fit_zoom(page))
            pages.append(Page(number=i, image_b64=_b64(_render_png(page, zoom))))
    return LoadedDocument(SourceKind.PDF, path, pages)


def load_image(path: Path) -> LoadedDocument:
    """Keep the original bytes unless the image is larger than MAX_SIDE_PX."""
    pix = pymupdf.Pixmap(str(path))
    if max(pix.width, pix.height) <= MAX_SIDE_PX:
        page = Page(number=1, image_b64=_b64(path.read_bytes()),
                    media_type=IMAGE_SUFFIXES[path.suffix.lower()])
    else:
        with pymupdf.open(path) as doc:
            page = Page(number=1, image_b64=_b64(_render_png(doc[0], _fit_zoom(doc[0]))))
    return LoadedDocument(SourceKind.IMAGE, path, [page])


def load_document(path: Path, dpi: int = 200) -> LoadedDocument:
    """Load a PDF or an image into page images."""
    if detect_kind(path) is SourceKind.PDF:
        return load_pdf(path, dpi)
    return load_image(path)
