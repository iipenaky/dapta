"""
core/config.py
--------------
Single source of truth for all application configuration.
All environment variables are read here — no other file touches os.environ.

Usage:
    from core.config import settings
"""


from pathlib import Path
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # ── Database ───────────────────────────────────────────────────────────────
    database_url: str = (
        "sqlite+aiosqlite:///./dapta.db"
    )

    # ── Auth ───────────────────────────────────────────────────────────────────
    secret_key: str = "CHANGE_ME_IN_PRODUCTION"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 30

    # ── CORS ───────────────────────────────────────────────────────────────────
    allowed_origins: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
    ]

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def parse_origins(cls, v):
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    # ── DAPTA ─────────────────────────────────────────────────────────────────
    dapta_path: Path = Path(__file__).resolve().parents[3] / "dapta"
    dapta_models_path: Path = Path(__file__).resolve().parents[3] / "models"

    # ── Runtime ────────────────────────────────────────────────────────────────
    environment: str = "development"

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


settings = Settings()
