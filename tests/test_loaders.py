import base64

import pymupdf
import pytest

from huc.domain.models import SourceKind
from huc.ingestion import loaders


def _image(path, width: int, height: int):
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), False)
    pix.clear_with(255)
    pix.save(path)
    return path


def _png_size(b64: str) -> tuple[int, int]:
    pix = pymupdf.Pixmap(base64.b64decode(b64))
    return pix.width, pix.height


def test_pdf_pages_are_always_rasterized(tmp_path):
    path = tmp_path / "exam.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((50, 72), "Even a page with a text layer is OCR'd as an image.")
    doc.new_page()
    doc.save(path)

    loaded = loaders.load_document(path, dpi=100)
    assert loaded.kind is SourceKind.PDF
    assert [p.number for p in loaded.pages] == [1, 2]
    page = loaded.pages[0]
    assert base64.b64decode(page.image_b64).startswith(b"\x89PNG")
    assert page.to_content_block() == {
        "type": "image", "base64": page.image_b64, "mime_type": "image/png"
    }
    # A4 at 100 dpi
    assert _png_size(page.image_b64) == (827, 1170)


def test_high_dpi_pdf_is_capped(tmp_path):
    path = tmp_path / "exam.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(path)
    [page] = loaders.load_pdf(path, dpi=600).pages
    assert max(_png_size(page.image_b64)) == loaders.MAX_SIDE_PX


def test_small_image_is_kept_as_is(tmp_path):
    path = _image(tmp_path / "photo.JPG", 300, 400)
    loaded = loaders.load_document(path)
    assert loaded.kind is SourceKind.IMAGE
    [page] = loaded.pages
    assert page.media_type == "image/jpeg"
    assert base64.b64decode(page.image_b64) == path.read_bytes()


def test_large_image_is_downscaled(tmp_path):
    path = _image(tmp_path / "photo.png", 4000, 3000)
    [page] = loaders.load_document(path).pages
    assert page.media_type == "image/png"
    assert _png_size(page.image_b64) == (2000, 1500)


def test_unsupported_formats(tmp_path):
    for name in ("x.json", "x.docx"):
        with pytest.raises(ValueError, match="Unsupported file format"):
            loaders.detect_kind(tmp_path / name)
