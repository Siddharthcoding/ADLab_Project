from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import field_validator
import json


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "CP Roadmap Generator"
    VERSION: str = "1.0.0"

    # CORS origins (will be loaded from env properly)
    BACKEND_CORS_ORIGINS: List[str] = []

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v

    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "cp_roadmap"
    DATABASE_URL: Optional[str] = None

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v, values):
        if isinstance(v, str) and v:
            return v
        return (
            f"postgresql+asyncpg://{values.data.get('POSTGRES_USER')}:"
            f"{values.data.get('POSTGRES_PASSWORD')}@"
            f"{values.data.get('POSTGRES_SERVER')}:5432/"
            f"{values.data.get('POSTGRES_DB')}"
        )

    REDIS_URL: str = "redis://localhost:6379/0"

    SECRET_KEY: str = "change-this-to-a-secure-random-string-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7

    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
    }


settings = Settings()