from __future__ import annotations

from pathlib import Path

from sift.storage import (
    article_id,
    connect,
    count_unpushed,
    fetch_top_unpushed,
    init_db,
    insert_article,
    mark_pushed,
    mark_scored,
    recent_articles,
)


def _tmp_db(tmp_path: Path) -> Path:
    db = tmp_path / "test.db"
    init_db(db)
    return db


def test_init_db_creates_schema(tmp_path: Path) -> None:
    db = _tmp_db(tmp_path)
    with connect(db) as conn:
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"articles", "feedback", "source_cursor"}.issubset(names)


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
