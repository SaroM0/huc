"""OCR with Claude Haiku (vision): reads handwritten exam pages.

- `read_exam`: ONE call per submission with every page image and the exam questions;
  returns the answer to each question (transcription + segmentation in one step).
- `transcribe_page`: plain transcription of one page, for `huc ocr` and `scripts/ocr_eval.py`.

Page images are sent to the Anthropic API. The scans contain only answers (no student
names or IDs); the student is identified from the file name or `student_id`.
`ChatAnthropic` is a LangChain chat model, so OCR calls are traced in Studio/LangSmith.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Any

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from huc.config import Settings
from huc.domain.models import Exam
from huc.ingestion.loaders import Page
from huc.llm import build_messages, structured
from huc.prompts import ocr as prompts

# A whole exam (several dense pages) as structured answers fits well below this. Kept at
# 16k: above ~21k the SDK requires streaming for non-streaming-sized requests.
MAX_TOKENS = 16000
# The SDK retries rate limits (429), overload (529), 5xx and connection errors with backoff.
MAX_RETRIES = 4
# Appended when a page hits MAX_TOKENS, so graders see the answer may be incomplete.
TRUNCATED_MARK = "[transcription truncated]"


class OcrUnavailableError(RuntimeError):
    """The OCR model cannot be reached: missing/invalid API key, unknown model, or the API
    is down, overloaded or timing out after the SDK's retries."""


# Structured output of `read_exam`
class ReadAnswer(BaseModel):
    question_id: str
    text: str = Field(description="Verbatim transcription of the student's answer")
    confidence: float = Field(
        ge=0, le=1, description="Reliability of the transcription and of its question assignment"
    )
    source_pages: list[int] = Field(default_factory=list)


class ExamReading(BaseModel):
    answers: list[ReadAnswer]


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


async def _call(call: Awaitable[Any], model: Any, what: str) -> Any:
    """Await an OCR request, turning API outages into `OcrUnavailableError`."""
    name = _model_name(model)
    try:
        return await call
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
            f"OCR of {what} timed out after retries. Raise HUC_OCR_TIMEOUT_S "
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


async def read_exam(model: Any, pages: list[Page], exam: Exam) -> ExamReading:
    """Read every page of one submission and return the answer to each exam question.

    `model` is any chat model with `with_structured_output(schema).ainvoke(messages)`.
    The exam goes in the cached system prompt; the page images in the user message.
    """
    content: list[dict] = []
    for page in pages:
        content += [{"type": "text", "text": f"Page {page.number}:"}, page.to_content_block()]
    content.append({"type": "text", "text": prompts.READ_EXAM_USER})
    messages = build_messages(prompts.read_exam_system(exam), content)
    reader = structured(model, ExamReading)
    return await _call(reader.ainvoke(messages), model, f"{len(pages)} page(s)")


async def transcribe_page(model: Any, page: Page) -> str:
    """Transcribe one page image. `model` is any chat model with `ainvoke(messages)`."""
    messages = [
        SystemMessage(content=prompts.OCR_SYSTEM),
        HumanMessage(content=[page.to_content_block(), {"type": "text", "text": prompts.OCR_USER}]),
    ]
    reply = await _call(model.ainvoke(messages), model, f"page {page.number}")
    text = _text(reply)
    if _truncated(reply):
        text = f"{text}\n{TRUNCATED_MARK}"
    return text
