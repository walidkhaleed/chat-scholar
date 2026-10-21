
import os
import unittest

from unittest.mock import patch
from fastapi.testclient import TestClient

from main import app
from csrf_security import verify_csrf_token
from auth_security import SESSION_COOKIE_NAME
from login_rate_limit import LoginRateLimiter


class CSRFEndpointTests(unittest.TestCase):

    def test_csrf_endpoint(self):

        test_limiter = LoginRateLimiter()

        with (
            patch.dict(
                os.environ,
                {"CHAT_SCHOLAR_COOKIE_SECURE": "false"}
            ),
            patch(
                "auth_routes.verify_password",
                return_value=True
            ),
            patch(
                "auth_routes.login_limiter",
                test_limiter
            )
        ):

            with TestClient(app) as client:

                # TEST 1: Anonymous users are rejected.
                response = client.get("/auth/csrf")

                self.assertEqual(
                    response.status_code,
                    401
                )

                print("Anonymous access rejected: PASS")

                # TEST 2: Login using mocked password
                # verification (no real password needed).
                response = client.post(
                    "/auth/login",
                    json={
                        "password": "integration-test-password"
                    }
                )

                self.assertEqual(
                    response.status_code,
                    200
                )

                print("Test login: PASS")

                # TEST 3: Request CSRF token.
                response = client.get("/auth/csrf")

                self.assertEqual(
                    response.status_code,
                    200
                )

                csrf_token = response.json().get(
                    "csrf_token"
                )

                self.assertIsInstance(
                    csrf_token,
                    str
                )

                self.assertEqual(
                    len(csrf_token),
                    43
                )

                print("CSRF token generated: PASS")

                # TEST 4: Token must match the
                # current authenticated session.
                session_token = client.cookies.get(
                    SESSION_COOKIE_NAME
                )

                self.assertTrue(
                    verify_csrf_token(
                        session_token,
                        csrf_token
                    )
                )

                print("Session binding verified: PASS")

                # TEST 5: Response must not be cached.
                self.assertEqual(
                    response.headers.get("Cache-Control"),
                    "no-store"
                )

                print("Cache protection: PASS")


if __name__ == "__main__":
    unittest.main(verbosity=2)
