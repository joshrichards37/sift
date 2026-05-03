from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sift.recommender import (
    Suggestion,
    record_for_chat,
    suggest_for_chat,
)
from sift.storage import (
    connect,
    init_db,
    insert_article,
    last_response_for_topic,
    mark_scored,
    record_feedback,
    respond_to_suggestion,
)


def _db(tmp_path: Path) -> Path:
    db = tmp_path / "test.db"
    init_db(db)
    return db


def _seed(
    conn: sqlite3.Connection,
    *,
    url: str,
    title: str,
    topic_tags: list[str],
    rating: int,
    feedback_at: datetime | None = None,
) -> str:
    """Insert a scored article + a thumbs feedback row at the given time."""
    aid = insert_article(conn, source_id="hn", url=url, title=title)
    assert aid is not None
    mark_scored(conn, aid, 8, "summary", topic_tags=topic_tags)
    fb_at = (feedback_at or datetime.now(UTC)).isoformat()
    conn.execute(
        "INSERT INTO feedback (article_id, rating, note, created_at) VALUES (?,?,?,?)",
        (aid, rating, None, fb_at),
    )
    return aid


def test_suggests_topic_user_engages_with_but_hasnt_subscribed(tmp_path: Path) -> None:
    """The headline behaviour: user has thumbed up 3 articles tagged 'post-training',
    'post-training' is not in their stated prefs → suggester surfaces it."""
    db = _db(tmp_path)
    with connect(db) as conn:
        for i in range(3):
            _seed(
                conn,
                url=f"https://x/{i}",
                title=f"a{i}",
                topic_tags=["post-training"],
                rating=1,
            )
        out = suggest_for_chat(conn, chat_id="42", prefs_topics="ai tooling, llms")
    assert out is not None
    assert out.topic == "post-training"
    assert len(out.evidence_article_ids) == 3
    assert out.confidence > 0


def test_does_not_suggest_topics_already_in_prefs(tmp_path: Path) -> None:
    """If 'rust async' is in the user's prefs, never suggest it — case-insensitive."""
    db = _db(tmp_path)
    with connect(db) as conn:
        for i in range(5):
            _seed(
                conn, url=f"https://x/{i}", title=f"a{i}", topic_tags=["rust async"], rating=1
            )
        out = suggest_for_chat(conn, chat_id="1", prefs_topics="Rust Async, distributed systems")
    assert out is None


def test_filters_below_min_samples(tmp_path: Path) -> None:
    """Default min_samples is 3 — two thumbs aren't enough signal."""
    db = _db(tmp_path)
    with connect(db) as conn:
        for i in range(2):
            _seed(conn, url=f"https://x/{i}", title=f"a{i}", topic_tags=["niche"], rating=1)
        out = suggest_for_chat(conn, chat_id="1", prefs_topics="other stuff")
    assert out is None


def test_filters_below_positive_ratio(tmp_path: Path) -> None:
    """3 samples but mostly negative — don't suggest a topic the user dislikes."""
    db = _db(tmp_path)
    with connect(db) as conn:
        _seed(conn, url="https://x/1", title="a1", topic_tags=["crypto"], rating=1)
        _seed(conn, url="https://x/2", title="a2", topic_tags=["crypto"], rating=-1)
        _seed(conn, url="https://x/3", title="a3", topic_tags=["crypto"], rating=-1)
        out = suggest_for_chat(conn, chat_id="1", prefs_topics="other")
    assert out is None


def test_picks_highest_confidence_topic(tmp_path: Path) -> None:
    """Multiple qualifying topics — surface the strongest signal first.
    Both clear the hard floors; confidence ordering decides the winner."""
    db = _db(tmp_path)
    with connect(db) as conn:
        # weak: 3 samples, 100% positive → confidence = 1.0 * 0.3 = 0.30
        for i in range(3):
            _seed(
                conn, url=f"https://w/{i}", title=f"w{i}", topic_tags=["weak-topic"], rating=1
            )
        # strong: 10 samples, 100% positive → confidence = 1.0 * 1.0 = 1.00
        for i in range(10):
            _seed(
                conn,
                url=f"https://s/{i}",
                title=f"s{i}",
                topic_tags=["strong-topic"],
                rating=1,
            )
        out = suggest_for_chat(conn, chat_id="1", prefs_topics="other")
    assert out is not None
    assert out.topic == "strong-topic"


def test_respects_lookback_window(tmp_path: Path) -> None:
    """Old feedback shouldn't keep resurfacing topics the user no longer engages with."""
    db = _db(tmp_path)
    long_ago = datetime.now(UTC) - timedelta(days=90)
    with connect(db) as conn:
        for i in range(5):
            _seed(
                conn,
                url=f"https://x/{i}",
                title=f"a{i}",
                topic_tags=["stale-interest"],
                rating=1,
                feedback_at=long_ago,
            )
        out = suggest_for_chat(conn, chat_id="1", prefs_topics="other", lookback_days=30)
    assert out is None


def test_muted_topic_never_resuggested(tmp_path: Path) -> None:
    """'Mute permanently' must actually mean it. Even with overwhelming signal,
    a muted topic stays muted."""
    db = _db(tmp_path)
    with connect(db) as conn:
        # Seed enough signal to trivially pass the confidence bar.
        for i in range(10):
            _seed(
                conn,
                url=f"https://x/{i}",
                title=f"a{i}",
                topic_tags=["muted-topic"],
                rating=1,
            )
        sid = record_for_chat(
            conn,
            chat_id="1",
            suggestion=Suggestion(
                topic="muted-topic", confidence=0.9, evidence_article_ids=[]
            ),
        )
        respond_to_suggestion(conn, sid, "muted")
        out = suggest_for_chat(conn, chat_id="1", prefs_topics="other")
    assert out is None


def test_declined_topic_blocked_during_cooldown(tmp_path: Path) -> None:
    """Declined topics get a cooldown rather than a permanent block — they can come
    back if the user keeps engaging — but during the cooldown window we don't nag."""
    db = _db(tmp_path)
    now = datetime.now(UTC)
    with connect(db) as conn:
        for i in range(10):
            _seed(
                conn,
                url=f"https://x/{i}",
                title=f"a{i}",
                topic_tags=["declined-topic"],
                rating=1,
            )
        sid = record_for_chat(
            conn,
            chat_id="1",
            suggestion=Suggestion(
                topic="declined-topic", confidence=0.9, evidence_article_ids=[]
            ),
        )
        # Manually set the response_at so we can step over the cooldown.
        conn.execute(
            "UPDATE suggestion_candidates SET response = 'declined', responded_at = ? WHERE id = ?",
            ((now - timedelta(days=10)).isoformat(), sid),
        )

        # 10 days after decline, default 30-day cooldown → still blocked.
        within = suggest_for_chat(
            conn, chat_id="1", prefs_topics="other", now=now, decline_cooldown_days=30
        )
        # Pretend a month has passed — cooldown elapsed, the topic is fair game again.
        after = suggest_for_chat(
            conn,
            chat_id="1",
            prefs_topics="other",
            now=now + timedelta(days=25),  # 35 days post-decline
            decline_cooldown_days=30,
        )
    assert within is None
    assert after is not None
    assert after.topic == "declined-topic"


def test_decline_in_one_chat_does_not_block_other_chat(tmp_path: Path) -> None:
    """Per-chat state isolation: Bob declining 'tts' doesn't suppress it for Alice."""
    db = _db(tmp_path)
    with connect(db) as conn:
        for i in range(10):
            _seed(conn, url=f"https://x/{i}", title=f"a{i}", topic_tags=["tts"], rating=1)
        sid = record_for_chat(
            conn,
            chat_id="bob",
            suggestion=Suggestion(topic="tts", confidence=0.9, evidence_article_ids=[]),
        )
        respond_to_suggestion(conn, sid, "muted")
        bob = suggest_for_chat(conn, chat_id="bob", prefs_topics="other")
        alice = suggest_for_chat(conn, chat_id="alice", prefs_topics="other")
    assert bob is None
    assert alice is not None
    assert alice.topic == "tts"


def test_articles_without_topic_tags_ignored(tmp_path: Path) -> None:
    """Pre-migration articles (NULL topic_tags) shouldn't crash or skew the count."""
    db = _db(tmp_path)
    with connect(db) as conn:
        # 2 tagged thumbs, 5 untagged thumbs — only the tagged ones count.
        for i in range(2):
            _seed(
                conn, url=f"https://t/{i}", title=f"t{i}", topic_tags=["a-tag"], rating=1
            )
        for i in range(5):
            aid = insert_article(conn, source_id="hn", url=f"https://u/{i}", title=f"u{i}")
            assert aid is not None
            mark_scored(conn, aid, 8, "s")  # no topic_tags
            record_feedback(conn, aid, 1, None)
        out = suggest_for_chat(conn, chat_id="1", prefs_topics="other")
    # Below min_samples=3 for the only tagged topic.
    assert out is None


def test_record_for_chat_persists_via_storage_helpers(tmp_path: Path) -> None:
    """record_for_chat is a thin wrapper, but verify it actually round-trips
    through the storage layer rather than silently dropping."""
    db = _db(tmp_path)
    with connect(db) as conn:
        sid = record_for_chat(
            conn,
            chat_id="1",
            suggestion=Suggestion(
                topic="example", confidence=0.8, evidence_article_ids=["a", "b"]
            ),
        )
        row = conn.execute(
            "SELECT topic, confidence, evidence_article_ids FROM suggestion_candidates WHERE id = ?",
            (sid,),
        ).fetchone()
    assert row["topic"] == "example"
    assert row["confidence"] == 0.8
    assert json.loads(row["evidence_article_ids"]) == ["a", "b"]
    # And the response-tracking helpers see nothing yet.
    with connect(db) as conn:
        assert last_response_for_topic(conn, chat_id="1", topic="example") is None
