from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import AliasChoices, BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    telegram_bot_token: str
    owner_chat_id: int
    # Comma-separated extra chat IDs that can use the bot. Owner is auto-included.
    authorized_chat_ids: str = ""

    # LLM_* is canonical; OLLAMA_* kept as fallback so pre-existing .env files
    # from older sift installs keep working without manual migration.
    llm_base_url: str = Field(
        default="http://localhost:11434/v1",
        validation_alias=AliasChoices("LLM_BASE_URL", "OLLAMA_URL"),
    )
    llm_model: str = Field(
        default="qwen3:30b-a3b-instruct-2507-q4_K_M",
        validation_alias=AliasChoices("LLM_MODEL", "OLLAMA_MODEL"),
    )
    llm_api_key: str = Field(
        default="ollama",
        validation_alias=AliasChoices("LLM_API_KEY", "OLLAMA_API_KEY"),
    )

    bluesky_handle: str | None = None
    bluesky_app_password: str | None = None

    db_path: Path = Path("./sift.db")
    preferences_path: Path = Path("./preferences.yaml")

    @property
    def chat_ids(self) -> list[int]:
        """All chat IDs allowed to use the bot — owner first, then friends."""
        ids: list[int] = [self.owner_chat_id]
        for raw in (self.authorized_chat_ids or "").split(","):
            raw = raw.strip()
            if not raw:
                continue
            try:
                cid = int(raw)
            except ValueError:
                continue
            if cid not in ids:
                ids.append(cid)
        return ids


class SourcePref(BaseModel):
    id: str
    enabled: bool = True
    cadence_seconds: int = 1800
    url: str | None = None
    query: str | None = None
    min_points: int | None = None
    handle: str | None = None
    subreddit: str | None = None


class Preferences(BaseModel):
    topics: str
    exclude_keywords: list[str] = Field(default_factory=list)
    relevance_threshold: int = 7
    summary_target_words: int = 80
    max_per_cycle: int = 3  # cap of LLM scoring calls per source per poll
    digest_time: str = "09:00"  # HH:MM local time the daily digest fires
    digest_size: int = 10  # top N articles per digest
    more_size: int = 5  # default page size for /more
    sources: list[SourcePref]


def load_preferences(path: Path) -> Preferences:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Preferences.model_validate(raw)
