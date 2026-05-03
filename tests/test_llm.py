from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from sift.config import Preferences, SourcePref
from sift.llm import LLM, Score
from sift.sources.base import Article


def _prefs(exclude: list[str] | None = None) -> Preferences:
    return Preferences(
        topics="ai tooling",
        exclude_keywords=exclude or [],
        sources=[SourcePref(id="hn", query="x", enabled=True)],
    )


def _article(title: str = "T", body: str = "B") -> Article:
    return Article(source_id="hn", url="https://x.example/a", title=title, body=body)


def _llm_with_response(content: str) -> LLM:
    """Build an LLM whose chat.completions.create returns a fake message with the given content."""
    llm = LLM(base_url="http://test/v1", api_key="x", model="test-model")
    fake_msg = SimpleNamespace(content=content)
    fake_choice = SimpleNamespace(message=fake_msg)
    fake_resp = SimpleNamespace(choices=[fake_choice])
    llm.client.chat.completions.create = AsyncMock(return_value=fake_resp)  # type: ignore[method-assign]
    return llm


async def test_score_parses_valid_json() -> None:
    llm = _llm_with_response(json.dumps({"score": 8, "reason": "matches topics"}))
    out = await llm.score_relevance(_article(), _prefs())
    assert out == Score(score=8, reason="matches topics", topic_tags=[])


async def test_score_extracts_topic_tags() -> None:
    """Topic tags ride along on the same scoring call. Lowercase, trimmed, capped at 3."""
    llm = _llm_with_response(
        json.dumps(
            {
                "score": 9,
                "reason": "directly on-topic",
                "topic_tags": ["  Post-Training  ", "RLHF", "Alignment", "Dropped"],
            }
        )
    )
    out = await llm.score_relevance(_article(), _prefs())
    assert out.topic_tags == ["post-training", "rlhf", "alignment"]


async def test_score_topic_tags_default_empty_when_missing() -> None:
    """Older models / models that ignore the new prompt field must not crash."""
    llm = _llm_with_response(json.dumps({"score": 7, "reason": "ok"}))
    out = await llm.score_relevance(_article(), _prefs())
    assert out.topic_tags == []


async def test_score_topic_tags_robust_to_garbage() -> None:
    """If topic_tags comes back as the wrong shape, the score still survives."""
    llm = _llm_with_response(json.dumps({"score": 7, "reason": "ok", "topic_tags": "not-a-list"}))
    out = await llm.score_relevance(_article(), _prefs())
    assert out.score == 7
    assert out.topic_tags == []


async def test_score_clamps_above_ten() -> None:
    """LLMs sometimes return scores outside the 1-10 range. Clamp, don't fail."""
    llm = _llm_with_response(json.dumps({"score": 15, "reason": "way over"}))
    out = await llm.score_relevance(_article(), _prefs())
    assert out.score == 10


async def test_score_clamps_below_one() -> None:
    llm = _llm_with_response(json.dumps({"score": 0, "reason": "zero"}))
    out = await llm.score_relevance(_article(), _prefs())
    assert out.score == 1


async def test_score_returns_parse_failed_on_invalid_json() -> None:
    """Some hosted models prepend prose before the JSON. We treat the whole
    response as a parse failure rather than raising."""
    llm = _llm_with_response("not actually JSON {whatever")
    out = await llm.score_relevance(_article(), _prefs())
    assert out.score == 0
    assert out.reason == "parse-failed"


async def test_score_exclude_keyword_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    """If an exclude keyword matches title or body, return score=1 WITHOUT calling
    the LLM. Saves a token for obvious filter cases (e.g. crypto noise)."""
    llm = LLM(base_url="http://test/v1", api_key="x", model="test-model")
    create_mock = AsyncMock()
    llm.client.chat.completions.create = create_mock  # type: ignore[method-assign]

    article = _article(title="A new NFT marketplace launched")
    out = await llm.score_relevance(article, _prefs(exclude=["NFT"]))
    assert out.score == 1
    assert "excluded by keyword" in out.reason
    create_mock.assert_not_called()
