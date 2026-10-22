"""Application settings, read from environment variables (prefix `HUC_`) or `.env`."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HUC_", env_file=".env", extra="ignore")

    # Anthropic API key. Read without the HUC_ prefix, from the environment or `.env`.
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    # One Claude model per agent role
    grader_model: str = "claude-opus-5-5"
    reviewer_model: str = "claude-opus-5-5"
    segmenter_model: str = "claude-opus-5-5"

    # OCR: Claude Haiku transcribes each handwritten page image.
    ocr_model: str = "claude-haiku-5-5"
    # Per-page request timeout (the SDK also retries transient errors).
    ocr_timeout_s: float = 120
    # Resolution used to rasterize PDF pages before OCR
    pdf_dpi: int = 200
    # Longest side, in pixels, of each page image sent to the OCR model. 1568 px is the
    # largest size Claude uses without downscaling; more only adds upload time.
    ocr_max_side_px: int = 1568

    data_dir: Path = Path("./data")
    # Max parallel LLM calls per graph run (pass as config={"max_concurrency": ...})
    max_concurrency: int = 8
    # Any score with confidence below this sends the submission to human review
    human_review_threshold: float = 0.7


@lru_cache
def get_settings() -> Settings:
    return Settings()
