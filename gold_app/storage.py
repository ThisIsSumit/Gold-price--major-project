from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from contextlib import closing
from typing import Any

import pandas as pd

from .config import ARTIFACT_DIR, DATABASE_PATH, SNAPSHOT_CACHE_PATH


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with closing(get_connection()) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                payload TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_created_at TEXT NOT NULL,
                article_date TEXT,
                source TEXT,
                title TEXT,
                summary TEXT,
                url TEXT,
                predicted_price REAL,
                predicted_change REAL,
                predicted_sentiment INTEGER,
                model_sentiment INTEGER
            )
            """
        )
        connection.commit()


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.to_datetime(value).isoformat()
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def save_snapshot(snapshot: dict[str, Any]) -> None:
    init_db()
    created_at = snapshot.get("created_at") or datetime.now(timezone.utc).isoformat()
    payload = json.dumps(snapshot, default=_json_default)
    SNAPSHOT_CACHE_PATH.write_text(payload, encoding="utf-8")

    with closing(get_connection()) as connection:
        connection.execute(
            "INSERT INTO snapshots (created_at, payload) VALUES (?, ?)",
            (created_at, payload),
        )

        article_rows = snapshot.get("articles", [])
        for article in article_rows:
            connection.execute(
                """
                INSERT INTO articles (
                    snapshot_created_at, article_date, source, title, summary, url,
                    predicted_price, predicted_change, predicted_sentiment, model_sentiment
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    created_at,
                    article.get("article_date"),
                    article.get("source"),
                    article.get("title"),
                    article.get("summary"),
                    article.get("url"),
                    article.get("predicted_price"),
                    article.get("predicted_change"),
                    article.get("predicted_sentiment"),
                    article.get("model_sentiment"),
                ),
            )
            connection.commit()


def load_latest_snapshot() -> dict[str, Any] | None:
    if SNAPSHOT_CACHE_PATH.exists():
        return json.loads(SNAPSHOT_CACHE_PATH.read_text(encoding="utf-8"))

    init_db()
    with closing(get_connection()) as connection:
        row = connection.execute(
            "SELECT payload FROM snapshots ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    return json.loads(row[0])


def load_snapshot_history(limit: int = 200) -> pd.DataFrame:
    init_db()
    with closing(get_connection()) as connection:
        rows = connection.execute(
            "SELECT created_at, payload FROM snapshots ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(
        [{"created_at": row[0], "payload": json.loads(row[1])} for row in rows]
    )
