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
    # Set by a source when it determines it's permanently broken (missing
    # credentials, 404'd resource, etc.). The scheduler checks this and exits
    # the poll loop, sparing the agent from logging the same traceback every
    # cadence cycle. Use only for *deterministic* failures — transient errors
    # should let the scheduler's per-poll exception handler retry.
    disabled: bool = False
    disabled_reason: str = ""

    @abstractmethod
    async def poll(self) -> list[Article]:
        """Fetch the latest items. Idempotent — caller dedups by URL."""
