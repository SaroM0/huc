"""Local OCR: a vision model served by Ollama transcribes each handwritten page.

Page images never leave the machine; only the resulting text is sent to Claude.
`ChatOllama` is a LangChain chat model, so OCR calls are traced in Studio/LangSmith.
"""

from __future__ import annotations

from typing import Any

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from ollama import ResponseError

from huc.config import Settings
from huc.ingestion.loaders import Page
from huc.prompts import ocr as prompts

# Context window: one ~2000 px page image plus its transcription fit comfortably.
NUM_CTX = 8192
# Cap on generated tokens: stops the rare repetition loop of small vision models.
NUM_PREDICT = 4096


class OcrUnavailableError(RuntimeError):
    """Ollama is not reachable, the model is not pulled, or a page timed out."""


def get_ocr_model(settings: Settings) -> ChatOllama:
    """Build the Ollama chat model. Does not connect until the first call."""
    return ChatOllama(
        model=settings.ocr_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        num_ctx=NUM_CTX,
        num_predict=NUM_PREDICT,
        client_kwargs={"timeout": settings.ocr_timeout_s},
    )


def _describe(model: Any) -> tuple[str, str]:
    name = getattr(model, "model", None) or "<model>"
    url = getattr(model, "base_url", None) or "http://localhost:11434"
    return name, url


async def transcribe_page(model: Any, page: Page) -> str:
    """Transcribe one page image. `model` is any chat model with `ainvoke(messages)`."""
    messages = [
        SystemMessage(content=prompts.OCR_SYSTEM),
        HumanMessage(content=[page.to_content_block(), {"type": "text", "text": prompts.OCR_USER}]),
    ]
    name, url = _describe(model)
    try:
        reply = await model.ainvoke(messages)
    except (ConnectionError, httpx.ConnectError) as e:
        raise OcrUnavailableError(
            f"Cannot reach Ollama at {url}. Start it with `ollama serve` and make sure the "
            f"OCR model is available with `ollama pull {name}`."
        ) from e
    except httpx.TimeoutException as e:
        raise OcrUnavailableError(
            f"OCR of page {page.number} timed out. CPU inference is slow: raise "
            "HUC_OCR_TIMEOUT_S or use a smaller model (HUC_OCR_MODEL)."
        ) from e
    except ResponseError as e:
        if e.status_code == 404:
            raise OcrUnavailableError(
                f"OCR model {name!r} is not available in Ollama. Run `ollama pull {name}`."
            ) from e
        raise
    content = reply.content
    return (content if isinstance(content, str) else reply.text).strip()
