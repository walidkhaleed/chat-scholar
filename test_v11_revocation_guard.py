
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app

from auth_security import (
    SESSION_COOKIE_NAME,
    create_session_token,
)

from session_revocation import SessionRevocationStore


class RevocationGuardTests(unittest.TestCase):

    def test_revoked_session_rejected(self):

        with tempfile.TemporaryDirectory() as temp_dir:

            db_path = (
                Path(temp_dir)
                / "test_sessions.sqlite3"
            )

            test_store = SessionRevocationStore(db_path)

            token = create_session_token()

            # Replace the authentication guard's
            # store with an isolated test database.
            with patch(
                "auth_guard.session_store",
                test_store
            ):

                with TestClient(app) as client:

                    client.cookies.set(
                        SESSION_COOKIE_NAME,
                        token
                    )

                    # Test 1: Session initially works.
                    response = client.get("/documents")

                    self.assertEqual(
                        response.status_code,
                        200
                    )

                    print("1. Active session accepted: PASS")

                    # Test 2: Revoke the session.
                    test_store.revoke(token)

                    self.assertTrue(
                        test_store.is_revoked(token)
                    )

                    print("2. Session revoked in SQLite: PASS")

                    # Test 3: The same cookie must
                    # now be rejected by FastAPI.
                    response = client.get("/documents")

                    self.assertEqual(
                        response.status_code,
                        401
                    )

                    print("3. Revoked cookie rejected: PASS")

                    # Test 4: Another valid session
                    # must continue working.
                    another_token = create_session_token()

                    client.cookies.set(
                        SESSION_COOKIE_NAME,
                        another_token
                    )

                    response = client.get("/documents")

                    self.assertEqual(
                        response.status_code,
                        200
                    )

                    print("4. Other valid session accepted: PASS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
