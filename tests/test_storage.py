from __future__ import annotations

from pathlib import Path

import pytest

from sift.storage import (
    article_id,
    article_topic_tags,
    connect,
    count_unpushed,
    fetch_top_unpushed,
    init_db,
    insert_article,
    last_response_for_topic,
    mark_pushed,
    mark_scored,
    mark_suggestion_surfaced,
    pending_suggestion_for,
    recent_articles,
    record_suggestion_candidate,
    respond_to_suggestion,
)


def _tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "test.db"
    init_db(db)
    return db


def test_init_db_creates_schema(tmp_path: Path) -> None:
    db = _tmp_db(tmp_path)
    with connect(db) as conn:
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"articles", "feedback", "source_cursor", "suggestion_candidates"}.issubset(names)


def test_init_db_is_idempotent(tmp_path: Path) -> None:
    """Running init_db twice on the same file must succeed — covers re-runs and
    the additive topic_tags migration on an already-migrated DB."""
    db = tmp_path / "test.db"
    init_db(db)
    init_db(db)
    with connect(db) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(articles)")}
    assert "topic_tags" in cols


def test_topic_tags_migration_on_legacy_db(tmp_path: Path) -> None:
    """A DB created by an older sift (no topic_tags column) must gain the column
    when init_db runs against it — never blow up, never lose data."""
    import sqlite3

    db = tmp_path / "legacy.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """CREATE TABLE articles (
                id TEXT PRIMARY KEY, source_id TEXT NOT NULL, url TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL, body TEXT NOT NULL DEFAULT '', author TEXT,
                posted_at TEXT, ingested_at TEXT NOT NULL,
                relevance_score INTEGER, summary TEXT, pushed_at TEXT
            )"""
        )
        conn.execute(
            "INSERT INTO articles (id, source_id, url, title, ingested_at) VALUES (?,?,?,?,?)",
            ("abc", "hn", "https://legacy/x", "legacy row", "2026-01-01T00:00:00+00:00"),
        )
    init_db(db)
    with connect(db) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(articles)")}
        row = conn.execute("SELECT title, topic_tags FROM articles WHERE id = 'abc'").fetchone()
    assert "topic_tags" in cols
    assert row["title"] == "legacy row"
    assert row["topic_tags"] is None


def test_insert_article_roundtrip(tmp_path: Path) -> None:
    db = _tmp_db(tmp_path)
    with connect(db) as conn:
        aid = insert_article(
            conn,
            source_id="hn",
            url="https://x.example/a",
            title="Article",
            body="body",
            author="alice",
            posted_at="2026-01-01T00:00:00+00:00",
        )
        assert aid == article_id("https://x.example/a")
        row = conn.execute("SELECT * FROM articles WHERE id = ?", (aid,)).fetchone()
        assert row["title"] == "Article"
        assert row["author"] == "alice"
        assert row["pushed_at"] is None


def test_insert_article_dedup_returns_none(tmp_path: Path) -> None:
    """Second insert of the same URL must return None — that's how the scheduler
    knows it's already seen this URL and skips re-scoring."""
    db = _tmp_db(tmp_path)
    with connect(db) as conn:
        first = insert_article(conn, source_id="hn", url="https://dup.example", title="A")
        second = insert_article(conn, source_id="hn", url="https://dup.example", title="A again")
    assert first is not None
    assert second is None


def test_mark_scored_then_fetch_top_unpushed(tmp_path: Path) -> None:
    """End-to-end: insert several articles, score them, fetch_top_unpushed must
    respect min_score and order by score DESC."""
    db = _tmp_db(tmp_path)
    with connect(db) as conn:
        ids = [
            insert_article(conn, source_id="hn", url=f"https://x/{i}", title=f"a{i}")
            for i in range(4)
        ]
        # Scores: 9, 6, 8, 4 → with min_score=7 we expect [a0(9), a2(8)]
        mark_scored(conn, ids[0], 9, "summary 0")
        mark_scored(conn, ids[1], 6, "summary 1")
        mark_scored(conn, ids[2], 8, "summary 2")
        mark_scored(conn, ids[3], 4, "summary 3")

        rows = fetch_top_unpushed(conn, min_score=7, limit=10)
    titles = [r["title"] for r in rows]
    assert titles == ["a0", "a2"]
    assert count_unpushed_in(db, min_score=7) == 2
    assert count_unpushed_in(db, min_score=5) == 3


def count_unpushed_in(db: Path, min_score: int) -> int:
    with connect(db) as conn:
        return count_unpushed(conn, min_score)


def test_mark_pushed_excludes_from_subsequent_fetches(tmp_path: Path) -> None:
    db = _tmp_db(tmp_path)
    with connect(db) as conn:
        aid = insert_article(conn, source_id="hn", url="https://x/1", title="a")
        assert aid is not None
        mark_scored(conn, aid, 9, "s")
        assert len(fetch_top_unpushed(conn, min_score=7, limit=10)) == 1
        mark_pushed(conn, aid)
        assert fetch_top_unpushed(conn, min_score=7, limit=10) == []


def test_recent_articles_returns_only_pushed(tmp_path: Path) -> None:
    db = _tmp_db(tmp_path)
    with connect(db) as conn:
        a = insert_article(conn, source_id="hn", url="https://x/a", title="pushed")
        b = insert_article(conn, source_id="hn", url="https://x/b", title="not yet pushed")
        assert a is not None and b is not None
        mark_scored(conn, a, 9, "s")
        mark_scored(conn, b, 9, "s")
        mark_pushed(conn, a)
        recent = recent_articles(conn, limit=10)
    assert [r["title"] for r in recent] == ["pushed"]


def test_mark_scored_persists_topic_tags(tmp_path: Path) -> None:
    """Topic tags ride along with score+summary on the same UPDATE — no extra row."""
    db = _tmp_db(tmp_path)
    with connect(db) as conn:
        aid = insert_article(conn, source_id="hn", url="https://x/t1", title="tagged")
        assert aid is not None
        mark_scored(conn, aid, 8, "summary", topic_tags=["post-training", "rlhf"])
        row = conn.execute(
            "SELECT topic_tags FROM articles WHERE id = ?", (aid,)
        ).fetchone()
    assert article_topic_tags(row["topic_tags"]) == ["post-training", "rlhf"]


def test_mark_scored_topic_tags_default_none(tmp_path: Path) -> None:
    """Backwards compat: callers passing only score+summary still work, and the
    column stays NULL rather than storing an empty JSON array."""
    db = _tmp_db(tmp_path)
    with connect(db) as conn:
        aid = insert_article(conn, source_id="hn", url="https://x/t2", title="untagged")
        assert aid is not None
        mark_scored(conn, aid, 8, "summary")
        row = conn.execute(
            "SELECT topic_tags FROM articles WHERE id = ?", (aid,)
        ).fetchone()
    assert row["topic_tags"] is None
    assert article_topic_tags(row["topic_tags"]) == []


def test_article_topic_tags_handles_garbage() -> None:
    assert article_topic_tags(None) == []
    assert article_topic_tags("") == []
    assert article_topic_tags("not json") == []
    assert article_topic_tags('"a string not a list"') == []
    assert article_topic_tags('["ok", 1, "also-ok"]') == ["ok", "1", "also-ok"]


def test_suggestion_candidate_roundtrip(tmp_path: Path) -> None:
    """Record a candidate, surface it in a digest, user clicks 'added' — full lifecycle.
    The recommender + digest UX live in separate issues; this only proves the storage
    primitives work."""
    db = _tmp_db(tmp_path)
    with connect(db) as conn:
        sid = record_suggestion_candidate(
            conn,
            chat_id="42",
            topic="post-training",
            confidence=0.85,
            evidence_article_ids=["a1", "a2", "a3"],
        )
        assert sid > 0
        pending = pending_suggestion_for(conn, "42")
        assert pending is not None
        assert pending["topic"] == "post-training"
        assert pending["surfaced_at"] is None
        mark_suggestion_surfaced(conn, sid)
        respond_to_suggestion(conn, sid, "added")
        # Once responded, it's no longer pending — that's how we cap to 1/digest.
        assert pending_suggestion_for(conn, "42") is None
        last = last_response_for_topic(conn, chat_id="42", topic="post-training")
    assert last is not None
    assert last["response"] == "added"


def test_pending_suggestion_picks_highest_confidence(tmp_path: Path) -> None:
    db = _tmp_db(tmp_path)
    with connect(db) as conn:
        record_suggestion_candidate(
            conn, chat_id="1", topic="low", confidence=0.6, evidence_article_ids=["a"]
        )
        record_suggestion_candidate(
            conn, chat_id="1", topic="high", confidence=0.9, evidence_article_ids=["b"]
        )
        record_suggestion_candidate(
            conn, chat_id="1", topic="mid", confidence=0.75, evidence_article_ids=["c"]
        )
        pending = pending_suggestion_for(conn, "1")
    assert pending is not None
    assert pending["topic"] == "high"


def test_pending_suggestion_isolated_per_chat(tmp_path: Path) -> None:
    """Multi-tenant safety: chat A's suggestion must never surface for chat B."""
    db = _tmp_db(tmp_path)
    with connect(db) as conn:
        record_suggestion_candidate(
            conn, chat_id="A", topic="t", confidence=0.9, evidence_article_ids=["a"]
        )
        assert pending_suggestion_for(conn, "B") is None


def test_respond_to_suggestion_rejects_unknown_response(tmp_path: Path) -> None:
    db = _tmp_db(tmp_path)
    with connect(db) as conn:
        sid = record_suggestion_candidate(
            conn, chat_id="1", topic="t", confidence=0.8, evidence_article_ids=["a"]
        )
        with pytest.raises(ValueError):
            respond_to_suggestion(conn, sid, "maybe")
