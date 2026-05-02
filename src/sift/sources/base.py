from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class Article:
    source_id: str
    url: str
    title: str
    body: str = ""
    author: str | None = None
    posted_at: str | None = None  # ISO-8601 UTC


class Source(ABC):
    id: str
    cadence_seconds: int

    @abstractmethod
    async def poll(self) -> list[Article]:
        """Fetch the latest items. Idempotent — caller dedups by URL."""
