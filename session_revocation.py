
import hashlib
import sqlite3
import time

from pathlib import Path

from auth_security import (
    SESSION_DURATION_SECONDS,
    verify_session_token,
)

from contextlib import closing

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_DATABASE_PATH = (
    BASE_DIR / "data" / "session_revocations.sqlite3"
)


class SessionRevocationStore:
    """
    Persistent SQLite storage for revoked sessions.

    Stores only SHA-256 fingerprints of session
    tokens, never the original tokens.

    Each revoked token is stored until expiration.
    """

    def __init__(self, db_path=DEFAULT_DATABASE_PATH):
        self.db_path = Path(db_path)

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        self._initialize_database()

    def _connect(self):
        connection = sqlite3.connect(
            str(self.db_path),
            timeout=10
        )

        connection.execute("PRAGMA busy_timeout = 10000")

        return connection

    def _initialize_database(self):
        with closing(self._connect()) as connection, connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS revoked_sessions (
                    token_hash TEXT PRIMARY KEY,
                    expires_at INTEGER NOT NULL,
                    revoked_at INTEGER NOT NULL
                )
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS
                idx_revoked_sessions_expiry
                ON revoked_sessions(expires_at)
            """)

    @staticmethod
    def _fingerprint(token):
        return hashlib.sha256(
            token.encode("utf-8")
        ).hexdigest()

    def revoke(self, token):
        """
        Revoke a currently valid signed session.

        Returns True when revocation is stored.
        Raises ValueError for invalid/expired tokens.
        """

        if not verify_session_token(token):
            raise ValueError(
                "Cannot revoke an invalid or expired session."
            )

        parts = token.split(".")

        issued_at = int(parts[1])

        expires_at = (
            issued_at + SESSION_DURATION_SECONDS
        )

        now = int(time.time())

        token_hash = self._fingerprint(token)

        with closing(self._connect()) as connection, connection:

            # Remove revocations for expired sessions.
            connection.execute(
                """
                DELETE FROM revoked_sessions
                WHERE expires_at <= ?
                """,
                (now,)
            )

            # Repeated revocation is safe.
            connection.execute(
                """
                INSERT OR IGNORE INTO revoked_sessions
                (token_hash, expires_at, revoked_at)
                VALUES (?, ?, ?)
                """,
                (token_hash, expires_at, now)
            )

        return True

    def is_revoked(self, token):
        """
        Return True if this token has been revoked.

        Note:
        This method checks the revocation database;
        it does not independently validate signatures.
        """

        if not isinstance(token, str) or not token:
            return False

        token_hash = self._fingerprint(token)

        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                """
                SELECT 1
                FROM revoked_sessions
                WHERE token_hash = ?
                AND expires_at > ?
                LIMIT 1
                """,
                (token_hash, int(time.time()))
            ).fetchone()

        return row is not None

    def is_active(self, token):
        """
        A session is active only if:
        1. Its signature and expiration are valid.
        2. It has not been revoked.

        Database errors propagate so the caller
        can fail closed rather than grant access.
        """

        if not verify_session_token(token):
            return False

        return not self.is_revoked(token)


# Shared application store.
session_store = SessionRevocationStore()
