
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


class RevokedAuthenticationEndpointTests(unittest.TestCase):

    def test_revoked_authentication_endpoints(self):

        with tempfile.TemporaryDirectory() as temp_dir:

            db_path = (
                Path(temp_dir) / "test_auth_revocation.sqlite3"
            )

            test_store = SessionRevocationStore(db_path)

            token = create_session_token()

            with (
                patch("auth_routes.session_store", test_store),
                patch("auth_guard.session_store", test_store),
            ):

                with TestClient(app) as client:

                    client.cookies.set(
                        SESSION_COOKIE_NAME,
                        token
                    )

                    # TEST 1: Active session accepted by /auth/me.
                    response = client.get("/auth/me")

                    self.assertEqual(response.status_code, 200)
                    print("1. Active /auth/me: PASS")

                    # TEST 2: Active session receives CSRF token.
                    response = client.get("/auth/csrf")

                    self.assertEqual(response.status_code, 200)
                    self.assertIn("csrf_token", response.json())

                    print("2. Active /auth/csrf: PASS")

                    # TEST 3: Revoke the token.
                    test_store.revoke(token)

                    self.assertTrue(test_store.is_revoked(token))

                    print("3. Session revoked: PASS")

                    # TEST 4: Revoked token rejected by /auth/me.
                    response = client.get("/auth/me")

                    self.assertEqual(response.status_code, 401)

                    print("4. Revoked /auth/me rejected: PASS")

                    # TEST 5: Revoked token cannot obtain CSRF.
                    response = client.get("/auth/csrf")

                    self.assertEqual(response.status_code, 401)

                    print("5. Revoked /auth/csrf rejected: PASS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
