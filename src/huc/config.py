"""Application settings, read from environment variables (prefix `HUC_`) or `.env`."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HUC_", env_file=".env", extra="ignore")

    # One Claude model per agent role (text only: page images never leave the machine)
    grader_model: str = "claude-opus-5-5"
    reviewer_model: str = "claude-opus-5-5"
    segmenter_model: str = "claude-opus-5-5"

    # Local OCR: a vision model served by Ollama transcribes the handwritten pages.
    # Use an *instruct* tag: thinking variants spend minutes reasoning on a CPU.
    ocr_model: str = "qwen3-vl:8b-instruct"
    ollama_base_url: str = "http://localhost:11434"
    # Per-page request timeout. CPU-only inference takes 1-3 min per page.
    ocr_timeout_s: float = 600
    # Resolution used to rasterize PDF pages before OCR
    pdf_dpi: int = 200

    data_dir: Path = Path("./data")
    # Max parallel LLM calls per graph run (pass as config={"max_concurrency": ...})
    max_concurrency: int = 8
    # Any score with confidence below this sends the submission to human review
    human_review_threshold: float = 0.7


@lru_cache
def get_settings() -> Settings:
    return Settings()
