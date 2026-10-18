from contextlib import closing

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "chat_scholar.db"

DATA_DIR.mkdir(exist_ok=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect():
    connection = sqlite3.connect(
        DB_PATH,
        timeout=15
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 15000")
    return connection


def initialize_database():
    """Create database tables if they do not exist."""

    with closing(connect()) as db, db:
        db.execute("PRAGMA journal_mode = WAL")

        db.execute("""
            CREATE TABLE IF NOT EXISTS pdf_sessions (
                session_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        db.execute("""
            CREATE TABLE IF NOT EXISTS pdf_messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                standalone_question TEXT NOT NULL,
                answer TEXT NOT NULL,
                status TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                verification_json TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id)
                    REFERENCES pdf_sessions(session_id)
                    ON DELETE CASCADE
            )
        """)

        db.execute("""
            CREATE INDEX IF NOT EXISTS idx_pdf_messages_session
            ON pdf_messages(session_id, message_id)
        """)


def validate_uuid(value: str) -> str:
    return str(UUID(value))


def create_session(
    document_id: str,
    title: str = "PDF Conversation"
) -> dict:

    document_id = validate_uuid(document_id)

    title = title.strip() or "PDF Conversation"

    if len(title) > 200:
        raise ValueError("Session title is too long.")

    session_id = str(uuid4())
    timestamp = utc_now()

    with closing(connect()) as db, db:
        db.execute(
            """
            INSERT INTO pdf_sessions (
                session_id,
                document_id,
                title,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                session_id,
                document_id,
                title,
                timestamp,
                timestamp
            )
        )

    return {
        "session_id": session_id,
        "document_id": document_id,
        "title": title,
        "created_at": timestamp,
        "updated_at": timestamp
    }


def get_session(session_id: str) -> dict | None:

    session_id = validate_uuid(session_id)

    with closing(connect()) as db, db:
        row = db.execute(
            """
            SELECT *
            FROM pdf_sessions
            WHERE session_id = ?
            """,
            (session_id,)
        ).fetchone()

    return dict(row) if row else None


def list_sessions() -> list:

    with closing(connect()) as db, db:
        rows = db.execute(
            """
            SELECT
                s.session_id,
                s.document_id,
                s.title,
                s.created_at,
                s.updated_at,
                COUNT(m.message_id) AS total_turns
            FROM pdf_sessions AS s
            LEFT JOIN pdf_messages AS m
                ON s.session_id = m.session_id
            GROUP BY s.session_id
            ORDER BY s.updated_at DESC
            """
        ).fetchall()

    return [dict(row) for row in rows]


def get_messages(
    session_id: str,
    limit: int | None = None
) -> list:
    """
    Return completed conversation turns in
    chronological order.
    """

    session_id = validate_uuid(session_id)

    if limit is not None and not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100.")

    with closing(connect()) as db, db:

        if limit is None:
            rows = db.execute(
                """
                SELECT *
                FROM pdf_messages
                WHERE session_id = ?
                ORDER BY message_id ASC
                """,
                (session_id,)
            ).fetchall()
        else:
            rows = db.execute(
                """
                SELECT *
                FROM (
                    SELECT *
                    FROM pdf_messages
                    WHERE session_id = ?
                    ORDER BY message_id DESC
                    LIMIT ?
                )
                ORDER BY message_id ASC
                """,
                (session_id, limit)
            ).fetchall()

    messages = []

    for row in rows:
        item = dict(row)

        item["sources"] = json.loads(
            item.pop("sources_json")
        )

        verification_json = item.pop(
            "verification_json"
        )

        item["verification"] = (
            json.loads(verification_json)
            if verification_json
            else None
        )

        messages.append(item)

    return messages


def save_turn(
    session_id: str,
    question: str,
    standalone_question: str,
    answer: str,
    status: str,
    sources: list,
    verification: dict | None
) -> int:
    """
    Save one complete question-answer turn atomically.

    No partial user/assistant turn is stored.
    """

    session_id = validate_uuid(session_id)
    timestamp = utc_now()

    with closing(connect()) as db, db:

        cursor = db.execute(
            """
            INSERT INTO pdf_messages (
                session_id,
                question,
                standalone_question,
                answer,
                status,
                sources_json,
                verification_json,
                created_at
            )
            SELECT ?, ?, ?, ?, ?, ?, ?, ?
            WHERE EXISTS (
                SELECT 1
                FROM pdf_sessions
                WHERE session_id = ?
            )
            """,
            (
                session_id,
                question,
                standalone_question,
                answer,
                status,
                json.dumps(sources, ensure_ascii=False),
                (
                    json.dumps(
                        verification,
                        ensure_ascii=False
                    )
                    if verification is not None
                    else None
                ),
                timestamp,
                session_id
            )
        )

        if cursor.rowcount != 1:
            raise FileNotFoundError(
                "PDF session no longer exists."
            )

        db.execute(
            """
            UPDATE pdf_sessions
            SET updated_at = ?
            WHERE session_id = ?
            """,
            (timestamp, session_id)
        )

        return cursor.lastrowid


def delete_session(session_id: str) -> bool:

    session_id = validate_uuid(session_id)

    with closing(connect()) as db, db:
        cursor = db.execute(
            """
            DELETE FROM pdf_sessions
            WHERE session_id = ?
            """,
            (session_id,)
        )

        return cursor.rowcount > 0


initialize_database()
