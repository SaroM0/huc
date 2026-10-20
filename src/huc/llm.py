"""Thin helpers around LangChain chat models.

Graphs depend on LangChain's `BaseChatModel` interface, so tests can inject a fake
model and the provider/model can be swapped without touching the graphs.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

MAX_TOKENS = 16000


@lru_cache
def get_chat_model(model: str) -> ChatAnthropic:
    """One cached client per model name. Reads ANTHROPIC_API_KEY lazily (on first call)."""
    # Adaptive thinking is compatible with native structured outputs (json_schema);
    # it is only incompatible with *forced* tool use, which we don't use.
    return ChatAnthropic(model=model, max_tokens=MAX_TOKENS, thinking={"type": "adaptive"})


def structured(llm: Any, schema: type) -> Any:
    """Bind a Pydantic output schema. Uses Claude's native structured outputs when possible."""
    if isinstance(llm, ChatAnthropic):
        return llm.with_structured_output(schema, method="json_schema")
    return llm.with_structured_output(schema)


def build_messages(system: str, content: list[dict]) -> list[BaseMessage]:
    """System prompt as a cached block (stable prefix) + the variable user content."""
    return [
        SystemMessage(
            content=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        ),
        HumanMessage(content=content),
    ]
