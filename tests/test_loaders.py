import base64

import pymupdf
import pytest

from huc.domain.models import SourceKind
from huc.ingestion import loaders


def _pdf(path, text: str | None):
    doc = pymupdf.open()
    page = doc.new_page()
    if text:
        page.insert_text((50, 72), text)
    else:
        page.draw_rect(pymupdf.Rect(50, 50, 200, 200))
    doc.save(path)


def test_digital_pdf_yields_text(tmp_path):
    path = tmp_path / "digital.pdf"
    _pdf(path, "Answer 2: mitosis produces two identical diploid cells.")
    doc = loaders.load_pdf(path)
    assert doc.kind is SourceKind.PDF
    [page] = doc.pages
    assert "mitosis" in page.text and page.image_b64 is None
    assert page.to_content_block()["type"] == "text"


def test_scanned_pdf_yields_png(tmp_path):
    path = tmp_path / "scanned.pdf"
    _pdf(path, None)
    [page] = loaders.load_pdf(path).pages
    assert page.text is None
    assert base64.b64decode(page.image_b64).startswith(b"\x89PNG")
    block = page.to_content_block()
    assert block == {"type": "image", "base64": page.image_b64, "mime_type": "image/png"}


def test_image_and_kind_detection(tmp_path):
    path = tmp_path / "photo.JPG"
    path.write_bytes(b"\xff\xd8fake")
    doc = loaders.load_document(path)
    assert doc.kind is SourceKind.IMAGE
    assert doc.pages[0].media_type == "image/jpeg"
    assert loaders.detect_kind(tmp_path / "x.json") is SourceKind.LMS_JSON
    with pytest.raises(ValueError):
        loaders.detect_kind(tmp_path / "x.docx")
