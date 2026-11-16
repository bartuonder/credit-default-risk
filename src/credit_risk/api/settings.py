from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from credit_risk.config import ROOT


class Settings(BaseSettings):
    """Service configuration, read from CREDIT_RISK_* environment variables."""

    model_config = SettingsConfigDict(env_prefix="CREDIT_RISK_", extra="ignore")

    model_path: Path = ROOT / "artifacts" / "model.joblib"
    model_card_path: Path = ROOT / "artifacts" / "model_card.json"
    cors_origins: list[str] = Field(default_factory=list, description="Allowed CORS origins.")
    max_batch_size: int = Field(100, ge=1, le=1000)
    max_body_bytes: int = Field(256_000, ge=1_000)
    log_level: str = "INFO"
    docs_enabled: bool = True
    explanation_top_k: int = Field(5, ge=1, le=15)
