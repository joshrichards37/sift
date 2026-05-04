from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sift.adjacency import (
    _parse_topics,
    cadence_blocks_chat,
    suggest_for_chat_via_adjacency,
)
from sift.llm import LLM
from sift.recommender import Suggestion, record_for_chat
from sift.storage import (
    connect,
    init_db,
    record_suggestion_candidate,
    respond_to_suggestion,
)


def _db(tmp_path: Path) -> Path:
    db = tmp_path / "test.db"
    init_db(db)
    return db


def _llm_with_topics(topics: list[str]) -> LLM:
    """Build an LLM whose chat() returns the given topics as JSON. Bypasses the
    real network by mocking the OpenAI client at the chat-completions level."""
    llm = LLM(base_url="http://test/v1", api_key="x", model="m")
    body = json.dumps({"topics": topics})
    fake_msg = SimpleNamespace(content=body)
    fake_choice = SimpleNamespace(message=fake_msg)
    fake_resp = SimpleNamespace(choices=[fake_choice], usage=None)
    llm.client.chat.completions.create = AsyncMock(return_value=fake_resp)  # type: ignore[method-assign]
    return llm


def test_parse_topics_happy_path() -> None:
    raw = json.dumps({"topics": ["model interpretability", "RLHF datasets", "agents"]})
    assert _parse_topics(raw) == ["model interpretability", "rlhf datasets", "agents"]


def test_parse_topics_lowercases_and_trims() -> None:
    raw = json.dumps({"topics": ["  Foo  ", "BAR BAZ"]})
    assert _parse_topics(raw) == ["foo", "bar baz"]


def test_parse_topics_robust_to_garbage() -> None:
    """Common LLM failure modes — handle each without raising."""
    assert _parse_topics("not json") == []
    assert _parse_topics(json.dumps([])) == []  # not a dict
    assert _parse_topics(json.dumps({"topics": "not a list"})) == []
    assert _parse_topics(json.dumps({"wrong_key": ["x"]})) == []
    # Non-string elements get filtered, valid ones kept
    assert _parse_topics(json.dumps({"topics": ["valid", 42, None, "also-valid"]})) == [
        "valid",
        "also-valid",
    ]


def test_cadence_blocks_when_recent_suggestion_exists(tmp_path: Path) -> None:
    """The cadence cap counts ANY suggestion (in-domain or adjacency) — both
    types share the cap so the user gets at most one per cooldown period."""
    db = _db(tmp_path)
    now = datetime.now(UTC)
    with connect(db) as conn:
        record_suggestion_candidate(
            conn, chat_id="42", topic="x", confidence=0.8, evidence_article_ids=["a"]
        )
        # Just-recorded → cooldown blocks
        assert cadence_blocks_chat(conn, chat_id="42", now=now, cooldown_days=7) is True
        # Different chat is unaffected
        assert cadence_blocks_chat(conn, chat_id="other", now=now, cooldown_days=7) is False


def test_cadence_releases_after_cooldown_window(tmp_path: Path) -> None:
    """Once the cooldown elapses, the chat is eligible again. Backdate the
    seeded row's created_at so the test isn't entangled with wall-clock now —
    record_suggestion_candidate uses real time, and on dates close to `base`
    that races the simulated 'past 7 days' assertion."""
    db = _db(tmp_path)
    base = datetime(2026, 5, 1, tzinfo=UTC)
    with connect(db) as conn:
        sid = record_suggestion_candidate(
            conn, chat_id="1", topic="x", confidence=0.8, evidence_article_ids=["a"]
        )
        conn.execute(
            "UPDATE suggestion_candidates SET created_at=? WHERE id=?",
            (base.isoformat(), sid),
        )
        # Within 7 days: still blocked.
        assert (
            cadence_blocks_chat(
                conn, chat_id="1", now=base + timedelta(days=5), cooldown_days=7
            )
            is True
        )
        # Past 7 days: released.
        assert (
            cadence_blocks_chat(
                conn, chat_id="1", now=base + timedelta(days=10), cooldown_days=7
            )
            is False
        )


async def test_suggest_for_chat_returns_first_qualifying_topic(tmp_path: Path) -> None:
    """Happy path: LLM proposes 3 topics, all pass filters, first one wins."""
    db = _db(tmp_path)
    llm = _llm_with_topics(["post-training", "interpretability", "rlhf"])
    with connect(db) as conn:
        out = await suggest_for_chat_via_adjacency(
            llm, conn, chat_id="42", prefs_topics="ai tooling, llms"
        )
    assert out is not None
    assert out.topic == "post-training"
    assert isinstance(out, Suggestion)


async def test_suggest_skips_topics_already_in_prefs(tmp_path: Path) -> None:
    """The model isn't trusted to honour the exclude hint — we filter locally
    against case-insensitive substring match in prefs_topics."""
    db = _db(tmp_path)
    llm = _llm_with_topics(["RLHF datasets", "interpretability", "fresh-topic"])
    with connect(db) as conn:
        out = await suggest_for_chat_via_adjacency(
            llm, conn, chat_id="1", prefs_topics="I follow RLHF datasets and interpretability"
        )
    assert out is not None
    assert out.topic == "fresh-topic"


async def test_suggest_skips_muted_topics(tmp_path: Path) -> None:
    """A muted topic must not bounce back via adjacency — 'mute permanently'
    has to apply across both mechanisms."""
    db = _db(tmp_path)
    llm = _llm_with_topics(["muted-topic", "fresh-topic"])
    base = datetime(2026, 4, 1, tzinfo=UTC)
    with connect(db) as conn:
        sid = record_for_chat(
            conn,
            chat_id="1",
            suggestion=Suggestion(
                topic="muted-topic", confidence=0.5, evidence_article_ids=[]
            ),
        )
        respond_to_suggestion(conn, sid, "muted")
        # Backdate the row so it's outside the cadence-cap window when we query.
        conn.execute(
            "UPDATE suggestion_candidates SET created_at=? WHERE id=?",
            (base.isoformat(), sid),
        )
        out = await suggest_for_chat_via_adjacency(
            llm, conn, chat_id="1", prefs_topics="other",
            now=base + timedelta(days=14),
        )
    assert out is not None
    assert out.topic == "fresh-topic"


async def test_suggest_skips_recently_declined_topics(tmp_path: Path) -> None:
    """Decline cooldown applies — same 30-day window the in-domain suggester uses."""
    db = _db(tmp_path)
    llm = _llm_with_topics(["declined-topic", "fresh-topic"])
    base = datetime(2026, 4, 1, tzinfo=UTC)
    with connect(db) as conn:
        sid = record_for_chat(
            conn,
            chat_id="1",
            suggestion=Suggestion(
                topic="declined-topic", confidence=0.5, evidence_article_ids=[]
            ),
        )
        # Backdate created_at OUTSIDE the cadence window, but the decline
        # response stays inside the 30-day decline cooldown.
        conn.execute(
            "UPDATE suggestion_candidates SET created_at=?, response='declined', responded_at=? "
            "WHERE id=?",
            (base.isoformat(), (base + timedelta(days=4)).isoformat(), sid),
        )
        out = await suggest_for_chat_via_adjacency(
            llm, conn, chat_id="1", prefs_topics="other",
            now=base + timedelta(days=14),  # 14d > 7d cadence; 10d < 30d decline
        )
    assert out is not None
    assert out.topic == "fresh-topic"


async def test_cadence_cap_blocks_within_window(tmp_path: Path) -> None:
    """Cadence cap fires BEFORE the LLM call — the LLM should NOT be invoked
    when a recent suggestion exists. Verifies the cheap check short-circuits."""
    db = _db(tmp_path)
    llm = _llm_with_topics(["whatever"])
    with connect(db) as conn:
        record_suggestion_candidate(
            conn, chat_id="1", topic="x", confidence=0.8, evidence_article_ids=["a"]
        )
        out = await suggest_for_chat_via_adjacency(
            llm, conn, chat_id="1", prefs_topics="ai", cooldown_days=7
        )
    assert out is None
    llm.client.chat.completions.create.assert_not_called()  # type: ignore[attr-defined]


async def test_cadence_releases_after_window_allows_adjacency(tmp_path: Path) -> None:
    """The other half of the cap test — once the 7-day window elapses the next
    eligible cycle invokes the LLM and returns a candidate."""
    db = _db(tmp_path)
    llm = _llm_with_topics(["adjacent-topic"])
    base = datetime(2026, 5, 1, tzinfo=UTC)
    with connect(db) as conn:
        # Seed a row dated to base, then check at base+8 days.
        sid = record_suggestion_candidate(
            conn, chat_id="1", topic="x", confidence=0.8, evidence_article_ids=["a"]
        )
        conn.execute(
            "UPDATE suggestion_candidates SET created_at=? WHERE id=?",
            (base.isoformat(), sid),
        )
        out = await suggest_for_chat_via_adjacency(
            llm, conn, chat_id="1", prefs_topics="ai", cooldown_days=7,
            now=base + timedelta(days=8),
        )
    assert out is not None
    assert out.topic == "adjacent-topic"


async def test_returns_none_when_llm_call_fails(tmp_path: Path) -> None:
    """Transient LLM failure must not break the digest cycle — we log and return None."""
    db = _db(tmp_path)
    llm = LLM(base_url="http://test/v1", api_key="x", model="m")
    llm.client.chat.completions.create = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("network error")
    )
    with connect(db) as conn:
        out = await suggest_for_chat_via_adjacency(
            llm, conn, chat_id="1", prefs_topics="ai"
        )
    assert out is None


async def test_returns_none_when_all_topics_filtered(tmp_path: Path) -> None:
    """LLM produced topics but every one was already in prefs / muted / declined.
    No fallback — return None and let the digest UX show no suggestion footer."""
    db = _db(tmp_path)
    llm = _llm_with_topics(["topic-a", "topic-b"])
    with connect(db) as conn:
        out = await suggest_for_chat_via_adjacency(
            llm, conn, chat_id="1",
            prefs_topics="topic-a is in prefs, and so is topic-b",
            cooldown_days=0,
        )
    assert out is None
