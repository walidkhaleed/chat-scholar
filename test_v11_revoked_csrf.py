
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


class RevokedCSRFTests(unittest.TestCase):

    def test_revoked_session_cannot_use_csrf(self):

        with tempfile.TemporaryDirectory() as temp_dir:

            test_store = SessionRevocationStore(
                Path(temp_dir) / "csrf_revocation.sqlite3"
            )

            with (
                patch.dict(
                    os.environ,
                    {
                        "CHAT_SCHOLAR_CSRF_ENFORCE": "true",
                        "CHAT_SCHOLAR_COOKIE_SECURE": "false",
                    }
                ),
                patch(
                    "csrf_guard.session_store",
                    test_store
                ),
                patch(
                    "auth_guard.session_store",
                    test_store
                ),
                patch(
                    "auth_routes.session_store",
                    test_store
                ),
            ):

                with TestClient(app) as client:

                    token = create_session_token()
                    csrf_token = create_csrf_token(token)

                    client.cookies.set(
                        SESSION_COOKIE_NAME,
                        token
                    )

                    # TEST 1: Active session + valid CSRF
                    response = client.post(
                        "/rubrics",
                        json={},
                        headers={
                            "X-CSRF-Token": csrf_token
                        }
                    )

                    # The intentionally invalid rubric
                    # reaches normal FastAPI validation.
                    self.assertEqual(
                        response.status_code,
                        422
                    )

                    print(
                        "1. Active session and valid CSRF: PASS"
                    )

                    # TEST 2: Active session without CSRF
                    response = client.post(
                        "/rubrics",
                        json={}
                    )

                    self.assertEqual(
                        response.status_code,
                        403
                    )

                    print(
                        "2. Missing CSRF rejected: PASS"
                    )

                    # TEST 3: Revoke the current session.
                    test_store.revoke(token)

                    self.assertTrue(
                        test_store.is_revoked(token)
                    )

                    print(
                        "3. Session revoked in SQLite: PASS"
                    )

                    # TEST 4: Replay the previous CSRF.
                    response = client.post(
                        "/rubrics",
                        json={},
                        headers={
                            "X-CSRF-Token": csrf_token
                        }
                    )

                    self.assertEqual(
                        response.status_code,
                        401
                    )

                    print(
                        "4. Revoked session + old CSRF rejected: PASS"
                    )

                    # TEST 5: Revoked session without CSRF.
                    response = client.post(
                        "/rubrics",
                        json={}
                    )

                    self.assertEqual(
                        response.status_code,
                        401
                    )

                    print(
                        "5. Revoked session without CSRF rejected: PASS"
                    )

                    # TEST 6: A different session works.
                    new_token = create_session_token()
                    new_csrf = create_csrf_token(new_token)

                    self.assertNotEqual(
                        new_token,
                        token
                    )

                    client.cookies.set(
                        SESSION_COOKIE_NAME,
                        new_token
                    )

                    response = client.post(
                        "/rubrics",
                        json={},
                        headers={
                            "X-CSRF-Token": new_csrf
                        }
                    )

                    self.assertEqual(
                        response.status_code,
                        422
                    )

                    print(
                        "6. Other active session accepted: PASS"
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
