from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from sift.config import Preferences
from sift.sources.base import Article

log = logging.getLogger(__name__)


RELEVANCE_SYSTEM = """You score how relevant an article is to a user's interests on a 1-10 scale.

10 = exactly the user's wheelhouse, would push immediately.
7  = clearly related to their stated topics, useful.
5  = adjacent / occasionally interesting.
1  = unrelated noise.

You also tag the article with 1-3 short topic labels (lowercase, 1-3 words each)
that capture what it's actually about, e.g. "post-training", "rust async", "fed policy".
Topic tags are independent of the user's stated interests — describe the article,
not its match to the user.

Reply with strict JSON only:
{"score": <int 1-10>, "reason": "<one short sentence>", "topic_tags": ["tag1", "tag2"]}
No markdown, no preamble."""


SUMMARY_SYSTEM = """You write the shortest possible summary of an article that conveys what matters.

- Hard ceiling: {target_words} words. Stop when you hit it; never go over.
- Lead with the single most important fact. If the article announces a thing, name it.
- Paraphrase, don't quote. Don't lift sentences from the article body.
- No filler ("In this article…", "The author argues…", "explores how…"). Just the substance."""


@dataclass(slots=True)
class Score:
    score: int
    reason: str
    topic_tags: list[str] = field(default_factory=list)


class LLM:
    def __init__(self, *, base_url: str, api_key: str, model: str) -> None:
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def score_relevance(self, article: Article, prefs: Preferences) -> Score:
        # Cheap exclude check before burning a token.
        haystack = (article.title + " " + article.body).lower()
        for kw in prefs.exclude_keywords:
            if kw.lower() in haystack:
                return Score(score=1, reason=f"excluded by keyword: {kw}")

        user_msg = (
            f"User interests:\n{prefs.topics}\n\n"
            f"Article:\nTitle: {article.title}\n"
            f"Source: {article.source_id}\n"
            f"Body: {_truncate(article.body, 2000)}"
        )
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": RELEVANCE_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        raw = resp.choices[0].message.content or "{}"
        try:
            parsed = json.loads(raw)
            score = int(parsed.get("score", 0))
            reason = str(parsed.get("reason", ""))[:200]
            topic_tags = _parse_topic_tags(parsed.get("topic_tags"))
        except (json.JSONDecodeError, ValueError, TypeError):
            log.warning("relevance returned non-JSON: %r", raw)
            return Score(score=0, reason="parse-failed")
        score = max(1, min(10, score))
        return Score(score=score, reason=reason, topic_tags=topic_tags)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def summarize(self, article: Article, prefs: Preferences) -> str:
        user_msg = (
            f"Title: {article.title}\n"
            f"Source: {article.source_id}\n"
            f"Body:\n{_truncate(article.body, 8000)}"
        )
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SUMMARY_SYSTEM.format(target_words=prefs.summary_target_words),
                },
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )
        return (resp.choices[0].message.content or "").strip()

    async def chat(self, system: str, history: list[dict], user_msg: str) -> str:
        messages = [
            {"role": "system", "content": system},
            *history,
            {"role": "user", "content": user_msg},
        ]
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.5,
        )
        return (resp.choices[0].message.content or "").strip()


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"


def _parse_topic_tags(raw: object) -> list[str]:
    """Coerce the model's topic_tags field into a clean list. Caps at 3 tags
    and 40 chars/tag — the prompt asks for that, but models drift."""
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw[:3]:
        if not isinstance(item, str):
            continue
        cleaned = item.strip().lower()[:40]
        if cleaned:
            out.append(cleaned)
    return out
