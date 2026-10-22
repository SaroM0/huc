"""OCR with Claude: a vision model (Claude Haiku) transcribes each handwritten page.

Page images are sent to the Anthropic API. The scans contain only answers (no student
names or IDs); the student is identified from the file name or `student_id`.
`ChatAnthropic` is a LangChain chat model, so OCR calls are traced in Studio/LangSmith.
"""

from __future__ import annotations

from typing import Any

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from huc.config import Settings
from huc.ingestion.loaders import Page
from huc.prompts import ocr as prompts

# A dense handwritten page transcribes to ~500-1500 tokens; this leaves ample room.
MAX_TOKENS = 8192
# The SDK retries rate limits (429), overload (529), 5xx and connection errors with backoff.
MAX_RETRIES = 4
# Appended when a page hits MAX_TOKENS, so graders see the answer may be incomplete.
TRUNCATED_MARK = "[transcription truncated]"


class OcrUnavailableError(RuntimeError):
    """The OCR model cannot be reached: missing/invalid API key, unknown model, or the API
    is down, overloaded or timing out after the SDK's retries."""


def get_ocr_model(settings: Settings) -> ChatAnthropic:
    """Build the Claude OCR model. Does not connect until the first call."""
    key = settings.anthropic_api_key
    return ChatAnthropic(
        model=settings.ocr_model,
        max_tokens=MAX_TOKENS,
        timeout=settings.ocr_timeout_s,
        max_retries=MAX_RETRIES,
        # Transcription needs no reasoning; thinking would only add latency and cost.
        # No `temperature`: current Claude models reject sampling parameters.
        **({"api_key": key.get_secret_value()} if key else {}),
    )


def _model_name(model: Any) -> str:
    return getattr(model, "model", None) or "<model>"


def _text(reply: Any) -> str:
    content = reply.content
    return (content if isinstance(content, str) else reply.text).strip()


def _truncated(reply: Any) -> bool:
    metadata = getattr(reply, "response_metadata", None) or {}
    return metadata.get("stop_reason") == "max_tokens"


async def transcribe_page(model: Any, page: Page) -> str:
    """Transcribe one page image. `model` is any chat model with `ainvoke(messages)`."""
    messages = [
        SystemMessage(content=prompts.OCR_SYSTEM),
        HumanMessage(content=[page.to_content_block(), {"type": "text", "text": prompts.OCR_USER}]),
    ]
    name = _model_name(model)
    try:
        reply = await model.ainvoke(messages)
    except anthropic.AuthenticationError as e:
        raise OcrUnavailableError(
            "The Anthropic API rejected the API key. Set a valid ANTHROPIC_API_KEY in .env "
            "or in the environment."
        ) from e
    except anthropic.PermissionDeniedError as e:
        raise OcrUnavailableError(f"This API key cannot use the OCR model {name!r}.") from e
    except anthropic.NotFoundError as e:
        raise OcrUnavailableError(
            f"OCR model {name!r} does not exist or is not available. Check HUC_OCR_MODEL."
        ) from e
    except anthropic.APITimeoutError as e:
        raise OcrUnavailableError(
            f"OCR of page {page.number} timed out after retries. Raise HUC_OCR_TIMEOUT_S "
            "or try again later."
        ) from e
    except anthropic.APIConnectionError as e:
        raise OcrUnavailableError(
            "Cannot reach the Anthropic API. Check the network connection."
        ) from e
    except (anthropic.RateLimitError, anthropic.InternalServerError) as e:
        raise OcrUnavailableError(
            f"The Anthropic API is rate limiting or overloaded (HTTP {e.status_code}) even "
            "after retries. Try again later or lower HUC_MAX_CONCURRENCY."
        ) from e
    text = _text(reply)
    if _truncated(reply):
        text = f"{text}\n{TRUNCATED_MARK}"
    return text
