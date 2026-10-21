
import os
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

from csrf_security import create_csrf_token

from session_revocation import SessionRevocationStore


class LogoutRevocationTests(unittest.TestCase):

    def test_logout_revokes_copied_cookie(self):

        with tempfile.TemporaryDirectory() as temp_dir:

            db_path = (
                Path(temp_dir) / "revocation_test.sqlite3"
            )

            test_store = SessionRevocationStore(db_path)

            with (
                patch.dict(
                    os.environ,
                    {
                        "CHAT_SCHOLAR_CSRF_ENFORCE": "true",
                        "CHAT_SCHOLAR_COOKIE_SECURE": "false",
                    }
                ),
                patch("auth_guard.session_store", test_store),
                patch("auth_routes.session_store", test_store),
            ):

                # Simulate an authenticated session.
                token = create_session_token()
                csrf_token = create_csrf_token(token)

                with TestClient(app) as client:

                    client.cookies.set(
                        SESSION_COOKIE_NAME,
                        token
                    )

                    # 1. Session works before logout.
                    response = client.get("/documents")

                    self.assertEqual(
                        response.status_code, 200
                    )

                    print("1. Original session active: PASS")

                    # Keep a copy of the token for replay.
                    copied_token = token

                    # 2. Logout using a valid CSRF token.
                    response = client.post(
                        "/auth/logout",
                        headers={
                            "X-CSRF-Token": csrf_token
                        }
                    )

                    self.assertEqual(
                        response.status_code, 200
                    )

                    print("2. Logout successful: PASS")

                    # 3. SQLite must record the revocation.
                    self.assertTrue(
                        test_store.is_revoked(copied_token)
                    )

                    print("3. Session revoked in SQLite: PASS")

                    # 4. Replay the copied session cookie.
                    client.cookies.set(
                        SESSION_COOKIE_NAME,
                        copied_token
                    )

                    response = client.get("/documents")

                    self.assertEqual(
                        response.status_code, 401
                    )

                    print("4. Copied cookie rejected: PASS")

                    # 5. A different session still works.
                    another_token = create_session_token()

                    client.cookies.set(
                        SESSION_COOKIE_NAME,
                        another_token
                    )

                    response = client.get("/documents")

                    self.assertEqual(
                        response.status_code, 200
                    )

                    print("5. Other session accepted: PASS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
