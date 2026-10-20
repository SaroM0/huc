"""Application settings, read from environment variables (prefix `HUC_`) or `.env`."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HUC_", env_file=".env", extra="ignore")

    # One model per agent role
    grader_model: str = "claude-opus-5-5"
    reviewer_model: str = "claude-opus-5-5"
    extractor_model: str = "claude-opus-5-5"

    data_dir: Path = Path("./data")
    # Max parallel LLM calls per graph run (pass as config={"max_concurrency": ...})
    max_concurrency: int = 8
    # Any score with confidence below this sends the submission to human review
    human_review_threshold: float = 0.7


@lru_cache
def get_settings() -> Settings:
    return Settings()
