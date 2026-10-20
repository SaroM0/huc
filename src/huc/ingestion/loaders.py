"""Loaders: turn an input file into normalized pages.

Each page carries text (digital PDF) or a PNG/JPEG image (scanned PDF or photo).
The extraction node decides what to send to the model.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

from huc.domain.models import SourceKind

# A PDF page with less text than this is treated as scanned and rasterized.
MIN_CHARS_PER_PAGE = 40
IMAGE_SUFFIXES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
SUPPORTED_SUFFIXES = {".pdf", ".json", *IMAGE_SUFFIXES}


@dataclass
class Page:
    number: int  # 1-based
    text: str | None = None
    image_b64: str | None = None
    media_type: str = "image/png"

    def to_content_block(self) -> dict:
        """LangChain standard content block (text or base64 image)."""
        if self.image_b64:
            return {"type": "image", "base64": self.image_b64, "mime_type": self.media_type}
        return {"type": "text", "text": f'<page n="{self.number}">\n{self.text}\n</page>'}


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
    if suffix == ".json":
        return SourceKind.LMS_JSON
    raise ValueError(f"Unsupported file format: {path.name}")


def load_pdf(path: Path, dpi: int = 150) -> LoadedDocument:
    import pymupdf

    pages: list[Page] = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text().strip()
            if len(text) >= MIN_CHARS_PER_PAGE:
                pages.append(Page(number=i, text=text))
            else:
                png = page.get_pixmap(dpi=dpi).tobytes("png")
                pages.append(Page(number=i, image_b64=base64.b64encode(png).decode()))
    return LoadedDocument(SourceKind.PDF, path, pages)


def load_image(path: Path) -> LoadedDocument:
    data = base64.b64encode(path.read_bytes()).decode()
    media_type = IMAGE_SUFFIXES[path.suffix.lower()]
    return LoadedDocument(
        SourceKind.IMAGE, path, [Page(number=1, image_b64=data, media_type=media_type)]
    )


def load_document(path: Path) -> LoadedDocument:
    """Load a PDF or image into pages (LMS JSON is handled by `load_lms_json`)."""
    kind = detect_kind(path)
    if kind is SourceKind.PDF:
        return load_pdf(path)
    if kind is SourceKind.IMAGE:
        return load_image(path)
    raise ValueError(f"{path.name} is LMS JSON, use load_lms_json()")


def load_lms_json(path: Path) -> dict:
    """LMS export. Expected shape: see examples/submission_lms.json."""
    return json.loads(path.read_text(encoding="utf-8"))
