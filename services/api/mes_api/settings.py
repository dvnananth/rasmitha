from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
    database_url: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./mes.db")

    class Config:
        env_prefix = ""
        env_file = "/app/.env"


settings = Settings()

