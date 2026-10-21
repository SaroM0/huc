"""Local OCR helpers. No Ollama needed: errors are simulated with fakes."""

import httpx
import pytest
from conftest import FakeOCR
from ollama import ResponseError

from huc.config import Settings
from huc.ingestion.loaders import Page
from huc.ocr import OcrUnavailableError, get_ocr_model, transcribe_page
from huc.prompts import ocr as prompts

PAGE = Page(number=1, image_b64="aGk=")


class Raising:
    model, base_url = "qwen3-vl:8b-instruct", "http://ollama:11434"

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def ainvoke(self, messages, *args, **kwargs):
        raise self.exc


def test_get_ocr_model_uses_settings():
    settings = Settings(ocr_model="qwen2.5vl:7b", ollama_base_url="http://x:1", ocr_timeout_s=42)
    model = get_ocr_model(settings)  # does not connect
    assert model.model == "qwen2.5vl:7b"
    assert model.base_url == "http://x:1"
    assert model.temperature == 0
    assert model.client_kwargs == {"timeout": 42}


async def test_transcribe_page_sends_image_and_strips_text():
    ocr = FakeOCR(["  1. La mitosis...\n"])
    assert await transcribe_page(ocr, PAGE) == "1. La mitosis..."
    system, user = ocr.calls[0]
    assert system.content == prompts.OCR_SYSTEM
    assert user.content[0] == {"type": "image", "base64": "aGk=", "mime_type": "image/png"}


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ConnectionError("refused"), "ollama serve"),  # what the ollama client raises
        (httpx.ConnectError("refused"), "http://ollama:11434"),
        (ResponseError("model not found", 404), "ollama pull qwen3-vl:8b-instruct"),
        (httpx.ReadTimeout("slow"), "HUC_OCR_TIMEOUT_S"),
    ],
)
async def test_clear_errors(exc, expected):
    with pytest.raises(OcrUnavailableError, match=expected):
        await transcribe_page(Raising(exc), PAGE)


async def test_other_ollama_errors_propagate():
    with pytest.raises(ResponseError):
        await transcribe_page(Raising(ResponseError("boom", 500)), PAGE)
