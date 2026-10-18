
import tempfile
import unittest

from pathlib import Path

from auth_security import create_session_token

from session_revocation import SessionRevocationStore
from contextlib import closing

class SessionRevocationTests(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        self.addCleanup(
            self.temp_dir.cleanup
        )

        self.db_path = (
            Path(self.temp_dir.name)
            / "test_revocations.sqlite3"
        )

        self.store = SessionRevocationStore(
            self.db_path
        )

        self.token = create_session_token()

    def test_new_session_is_active(self):
        self.assertTrue(
            self.store.is_active(self.token)
        )

    def test_revoked_session_is_inactive(self):
        self.assertTrue(
            self.store.revoke(self.token)
        )

        self.assertTrue(
            self.store.is_revoked(self.token)
        )

        self.assertFalse(
            self.store.is_active(self.token)
        )

    def test_other_session_remains_active(self):
        another_token = create_session_token()

        self.assertNotEqual(
            self.token,
            another_token
        )

        self.store.revoke(self.token)

        self.assertTrue(
            self.store.is_active(another_token)
        )

    def test_revocation_survives_restart(self):
        self.store.revoke(self.token)

        # Simulate reinitializing the storage module.
        new_store = SessionRevocationStore(
            self.db_path
        )

        self.assertFalse(
            new_store.is_active(self.token)
        )

    def test_invalid_session_is_rejected(self):
        self.assertFalse(
            self.store.is_active(
                "invalid-session"
            )
        )

        with self.assertRaises(ValueError):
            self.store.revoke(
                "invalid-session"
            )

    def test_repeated_revocation_is_safe(self):
        self.assertTrue(
            self.store.revoke(self.token)
        )

        self.assertTrue(
            self.store.revoke(self.token)
        )

        self.assertFalse(
            self.store.is_active(self.token)
        )

    def test_database_does_not_store_raw_token(self):
        self.store.revoke(self.token)

        import sqlite3

        with closing(sqlite3.connect(str(self.db_path))) as connection:
            row = connection.execute(
                """
                SELECT token_hash
                FROM revoked_sessions
                LIMIT 1
                """
            ).fetchone()

        self.assertIsNotNone(row)

        stored_hash = row[0]

        self.assertNotEqual(
            stored_hash,
            self.token
        )

        self.assertEqual(
            len(stored_hash),
            64
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
