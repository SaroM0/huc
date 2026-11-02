"""Application settings, read from environment variables (prefix `HUC_`) or `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HUC_", env_file=".env", extra="ignore")

    # Anthropic API key. Read without the HUC_ prefix, from the environment or `.env`.
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    # One Claude model per agent role. The whole pipeline runs on Haiku by default.
    grader_model: str = "claude-haiku-5-5"
    reviewer_model: str = "claude-haiku-5-5"
    feedback_auditor_model: str = "claude-haiku-5-5"
    feedback_audit_enabled: bool = True
    # Drafts exam YAMLs (questions + rubric) from reference-solution PDFs
    rubric_model: str = "claude-haiku-5-5"
    # Output budget and thinking mode of the grader/reviewer/rubric calls
    llm_max_tokens: int = 16000
    llm_thinking: Literal["adaptive", "disabled"] = "adaptive"

    # OCR: Claude Haiku reads all page images of a submission and splits them by question.
    ocr_model: str = "claude-haiku-5-5"
    # Per-request timeout (the SDK also retries transient errors). One request reads a
    # whole submission, so it has to cover all of its pages.
    ocr_timeout_s: float = 300
    # Resolution used to rasterize PDF pages before OCR
    pdf_dpi: int = 200
    # Longest side, in pixels, of each page image sent to the OCR model. 1568 px is the
    # largest size Claude uses without downscaling; more only adds upload time.
    ocr_max_side_px: int = 1568

    data_dir: Path = Path("./data")
    # Course profiles (`<id>.yaml`) that exam YAMLs reference with `profile: <id>`
    profiles_dir: Path = Path("./profiles")
    # Max parallel LLM calls per graph run (pass as config={"max_concurrency": ...})
    max_concurrency: int = 8
    # Any score with confidence below this labels the grade "needs teacher verification"
    # (a profile's grading.verification_threshold overrides it)
    verification_threshold: float = 0.7


@lru_cache
def get_settings() -> Settings:
    return Settings()
