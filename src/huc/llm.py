"""Thin helpers around LangChain chat models.

Graphs depend on LangChain's `BaseChatModel` interface, so tests can inject a fake
model and the provider/model can be swapped without touching the graphs.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from huc.config import get_settings


@lru_cache
def get_chat_model(
    model: str,
    max_tokens: int | None = None,
    thinking: Literal["adaptive", "disabled"] | None = None,
) -> ChatAnthropic:
    """One cached client per (model, max_tokens, thinking); None -> Settings. Does not
    connect until the first call."""
    settings = get_settings()
    key = settings.anthropic_api_key
    # Adaptive thinking is compatible with native structured outputs (json_schema);
    # it is only incompatible with *forced* tool use, which we don't use.
    return ChatAnthropic(
        model=model,
        max_tokens=max_tokens or settings.llm_max_tokens,
        thinking={"type": thinking or settings.llm_thinking},
        # Key from .env/environment via Settings; ChatAnthropic alone only sees the process env.
        **({"api_key": key.get_secret_value()} if key else {}),
    )


def structured(llm: Any, schema: type) -> Any:
    """Bind a Pydantic output schema. Uses Claude's native structured outputs when possible."""
    if isinstance(llm, ChatAnthropic):
        return llm.with_structured_output(schema, method="json_schema")
    return llm.with_structured_output(schema)


def build_messages(system: str | list[str], content: list[dict]) -> list[BaseMessage]:
    """System prompt + the variable user content.

    Each system block is a cache breakpoint, so pass the stable parts first (e.g. the
    course block shared by many calls, then the question block). At most 3 blocks
    (Anthropic allows 4 breakpoints per request).
    """
    blocks = [system] if isinstance(system, str) else [b for b in system if b]
    assert len(blocks) <= 3, "too many cache breakpoints"
    return [
        SystemMessage(
            content=[
                {"type": "text", "text": b, "cache_control": {"type": "ephemeral"}}
                for b in blocks
            ]
        ),
        HumanMessage(content=content),
    ]
