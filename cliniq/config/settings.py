"""
ClinIQ — Application configuration.
All secrets loaded from environment variables; safe defaults for dev/test.
"""
from __future__ import annotations

import os
from functools import lru_cache


class Settings:
    def __init__(self):
        # Read at instantiation so os.environ patches work in tests
        self.DATABASE_URL: str = os.getenv(
            "DATABASE_URL", "sqlite:///./cliniq_dev.db"
        )
        self.SECRET_KEY: str = os.getenv(
            "SECRET_KEY",
            "dev-secret-key-change-in-production-minimum-32-chars",
        )
        self.ALGORITHM: str = "HS256"
        self.ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
            os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60")
        )
        self.ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
        self.APP_TITLE: str = "ClinIQ API"
        self.APP_VERSION: str = "1.0.0"
        self.DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"


@lru_cache
def get_settings() -> Settings:
    return Settings()
