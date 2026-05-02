from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id              TEXT PRIMARY KEY,
    source_id       TEXT NOT NULL,
    url             TEXT NOT NULL UNIQUE,
    title           TEXT NOT NULL,
    body            TEXT NOT NULL DEFAULT '',
    author          TEXT,
    posted_at       TEXT,
    ingested_at     TEXT NOT NULL,
    relevance_score INTEGER,
    summary         TEXT,
    pushed_at       TEXT
);

CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source_id, ingested_at DESC);
CREATE INDEX IF NOT EXISTS idx_articles_pushed ON articles(pushed_at) WHERE pushed_at IS NOT NULL;

CREATE TABLE IF NOT EXISTS feedback (
    article_id   TEXT NOT NULL,
    rating       INTEGER NOT NULL CHECK (rating IN (-1, 1)),
    note         TEXT,
    created_at   TEXT NOT NULL,
    PRIMARY KEY (article_id, created_at),
    FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS source_cursor (
    source_id    TEXT PRIMARY KEY,
    cursor       TEXT NOT NULL,
    updated_at   TEXT NOT NULL
);
"""


def article_id(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def init_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)


@contextmanager
def connect(path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def already_seen(conn: sqlite3.Connection, url: str) -> bool:
    row = conn.execute("SELECT 1 FROM articles WHERE url = ?", (url,)).fetchone()
    return row is not None


def insert_article(
    conn: sqlite3.Connection,
    *,
    source_id: str,
    url: str,
    title: str,
    body: str = "",
    author: str | None = None,
    posted_at: str | None = None,
) -> str | None:
    """Insert and return article id, or None if URL is already known."""
    aid = article_id(url)
    try:
        conn.execute(
            """
            INSERT INTO articles (id, source_id, url, title, body, author, posted_at, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (aid, source_id, url, title, body, author, posted_at, now_iso()),
        )
    except sqlite3.IntegrityError:
        return None
    return aid


def mark_scored(conn: sqlite3.Connection, article_id: str, score: int, summary: str | None) -> None:
    conn.execute(
        "UPDATE articles SET relevance_score = ?, summary = ? WHERE id = ?",
        (score, summary, article_id),
    )


def mark_pushed(conn: sqlite3.Connection, article_id: str) -> None:
    conn.execute("UPDATE articles SET pushed_at = ? WHERE id = ?", (now_iso(), article_id))


def record_feedback(
    conn: sqlite3.Connection, article_id: str, rating: int, note: str | None
) -> None:
    conn.execute(
        "INSERT INTO feedback (article_id, rating, note, created_at) VALUES (?, ?, ?, ?)",
        (article_id, rating, note, now_iso()),
    )


def get_cursor(conn: sqlite3.Connection, source_id: str) -> str | None:
    row = conn.execute(
        "SELECT cursor FROM source_cursor WHERE source_id = ?", (source_id,)
    ).fetchone()
    return row["cursor"] if row else None


def set_cursor(conn: sqlite3.Connection, source_id: str, cursor: str) -> None:
    conn.execute(
        """
        INSERT INTO source_cursor (source_id, cursor, updated_at) VALUES (?, ?, ?)
        ON CONFLICT(source_id) DO UPDATE SET
            cursor = excluded.cursor,
            updated_at = excluded.updated_at
        """,
        (source_id, cursor, now_iso()),
    )


def fetch_top_unpushed(
    conn: sqlite3.Connection, *, min_score: int, limit: int
) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT id, source_id, url, title, summary, relevance_score, posted_at
            FROM articles
            WHERE pushed_at IS NULL
              AND relevance_score IS NOT NULL
              AND relevance_score >= ?
            ORDER BY relevance_score DESC, posted_at DESC
            LIMIT ?
            """,
            (min_score, limit),
        )
    )


def count_unpushed(conn: sqlite3.Connection, min_score: int) -> int:
    row = conn.execute(
        """
        SELECT COUNT(*) FROM articles
        WHERE pushed_at IS NULL
          AND relevance_score IS NOT NULL
          AND relevance_score >= ?
        """,
        (min_score,),
    ).fetchone()
    return int(row[0])


def mark_many_pushed(conn: sqlite3.Connection, ids: list[str]) -> None:
    if not ids:
        return
    now = now_iso()
    conn.executemany(
        "UPDATE articles SET pushed_at = ? WHERE id = ?",
        [(now, aid) for aid in ids],
    )


def recent_articles(conn: sqlite3.Connection, limit: int = 20) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT id, source_id, url, title, summary, relevance_score, posted_at
            FROM articles
            WHERE pushed_at IS NOT NULL
            ORDER BY pushed_at DESC
            LIMIT ?
            """,
            (limit,),
        )
    )
