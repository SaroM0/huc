"""Claude OCR adapter. No network, no API key: API errors are simulated with fakes."""

from types import SimpleNamespace

import anthropic
import httpx
import pytest
from conftest import FakeOCR
from pydantic import SecretStr

from huc.config import Settings
from huc.ingestion.loaders import Page
from huc.ocr import TRUNCATED_MARK, OcrUnavailableError, get_ocr_model, transcribe_page
from huc.prompts import ocr as prompts

PAGE = Page(number=1, image_b64="aGk=")
REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def _status_error(cls: type[anthropic.APIStatusError], status: int) -> anthropic.APIStatusError:
    return cls("error", response=httpx.Response(status, request=REQUEST), body=None)


class Raising:
    model = "claude-haiku-5-5"

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def ainvoke(self, messages, *args, **kwargs):
        raise self.exc


def test_get_ocr_model_uses_settings():
    settings = Settings(
        ocr_model="claude-haiku-5-5", ocr_timeout_s=42, anthropic_api_key=SecretStr("sk-test")
    )
    model = get_ocr_model(settings)  # does not connect
    assert model.model == "claude-haiku-5-5"
    assert model.default_request_timeout == 42
    assert model.anthropic_api_key.get_secret_value() == "sk-test"
    assert model.temperature is None  # current Claude models reject sampling parameters
    assert model.thinking is None


async def test_transcribe_page_sends_image_and_strips_text():
    ocr = FakeOCR(["  1. La mitosis...\n"])
    assert await transcribe_page(ocr, PAGE) == "1. La mitosis..."
    system, user = ocr.calls[0]
    assert system.content == prompts.OCR_SYSTEM
    assert user.content[0] == {"type": "image", "base64": "aGk=", "mime_type": "image/png"}


async def test_truncated_page_is_marked():
    class Truncating:
        async def ainvoke(self, messages, *args, **kwargs):
            return SimpleNamespace(
                content="1. Sea x", response_metadata={"stop_reason": "max_tokens"}
            )

    text = await transcribe_page(Truncating(), PAGE)
    assert text == f"1. Sea x\n{TRUNCATED_MARK}"


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (_status_error(anthropic.AuthenticationError, 401), "ANTHROPIC_API_KEY"),
        (_status_error(anthropic.PermissionDeniedError, 403), "cannot use the OCR model"),
        (_status_error(anthropic.NotFoundError, 404), "HUC_OCR_MODEL"),
        (_status_error(anthropic.RateLimitError, 429), "HTTP 429"),
        (_status_error(anthropic.InternalServerError, 529), "overloaded"),
        (anthropic.APITimeoutError(request=REQUEST), "HUC_OCR_TIMEOUT_S"),
        (anthropic.APIConnectionError(request=REQUEST), "network"),
    ],
)
async def test_clear_errors(exc, expected):
    with pytest.raises(OcrUnavailableError, match=expected):
        await transcribe_page(Raising(exc), PAGE)


async def test_bad_request_propagates():
    """A malformed request is a bug, not an outage: it must not be turned into a 503."""
    with pytest.raises(anthropic.BadRequestError):
        await transcribe_page(Raising(_status_error(anthropic.BadRequestError, 400)), PAGE)
