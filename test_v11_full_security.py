
import os
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app

from auth_security import SESSION_COOKIE_NAME
from login_rate_limit import LoginRateLimiter
from session_revocation import SessionRevocationStore


class FullSecurityRegressionTests(unittest.TestCase):

    def test_complete_authentication_lifecycle(self):

        print("\n=== V11 FULL SECURITY REGRESSION ===\n")

        with tempfile.TemporaryDirectory() as temp_dir:

            # Isolated test database.
            test_store = SessionRevocationStore(
                Path(temp_dir) / "security_test.sqlite3"
            )

            # Isolated login attempt limiter.
            test_limiter = LoginRateLimiter(
                max_attempts=5,
                window_seconds=900
            )

            test_environment = {
                "CHAT_SCHOLAR_SESSION_SECRET":
                    "v11-regression-secret-" + "A" * 48,
                "CHAT_SCHOLAR_CSRF_ENFORCE": "true",
                "CHAT_SCHOLAR_COOKIE_SECURE": "false",
            }

            with (
                patch.dict(os.environ, test_environment),
                patch(
                    "auth_routes.verify_password",
                    return_value=True
                ),
                patch(
                    "auth_routes.login_limiter",
                    test_limiter
                ),
                patch(
                    "auth_guard.session_store",
                    test_store
                ),
                patch(
                    "auth_routes.session_store",
                    test_store
                ),
                patch(
                    "csrf_guard.session_store",
                    test_store
                ),
            ):

                with TestClient(app) as client:

                    # ---------------------------------
                    # 1. Anonymous access
                    # ---------------------------------
                    response = client.get("/documents")

                    self.assertEqual(
                        response.status_code, 401
                    )
                    print("1. Anonymous access blocked: PASS")

                    # ---------------------------------
                    # 2. Login
                    # ---------------------------------
                    # Password verification is mocked
                    # only in this isolated test.
                    response = client.post(
                        "/auth/login",
                        json={
                            "password": "test-only-password"
                        }
                    )

                    self.assertEqual(
                        response.status_code, 200
                    )

                    old_token = client.cookies.get(
                        SESSION_COOKIE_NAME
                    )

                    self.assertTrue(old_token)
                    print("2. Login successful: PASS")

                    # ---------------------------------
                    # 3. Verify authenticated session
                    # ---------------------------------
                    response = client.get("/auth/me")

                    self.assertEqual(
                        response.status_code, 200
                    )
                    print("3. Session verified: PASS")

                    # ---------------------------------
                    # 4. Obtain CSRF token
                    # ---------------------------------
                    response = client.get("/auth/csrf")

                    self.assertEqual(
                        response.status_code, 200
                    )

                    old_csrf = response.json()["csrf_token"]

                    print("4. CSRF token received: PASS")

                    # ---------------------------------
                    # 5. Protected GET succeeds
                    # ---------------------------------
                    response = client.get("/documents")

                    self.assertEqual(
                        response.status_code, 200
                    )
                    print("5. Private API access: PASS")

                    # ---------------------------------
                    # 6. Missing CSRF rejected
                    # ---------------------------------
                    response = client.post(
                        "/rubrics",
                        json={}
                    )

                    self.assertEqual(
                        response.status_code, 403
                    )
                    print("6. Missing CSRF blocked: PASS")

                    # ---------------------------------
                    # 7. Valid CSRF accepted
                    # ---------------------------------
                    response = client.post(
                        "/rubrics",
                        json={},
                        headers={
                            "X-CSRF-Token": old_csrf
                        }
                    )

                    # Empty rubric reaches FastAPI
                    # validation and is rejected.
                    # No real rubric is created.
                    self.assertEqual(
                        response.status_code, 422
                    )
                    print("7. Valid CSRF accepted: PASS")

                    # ---------------------------------
                    # 8. Logout without CSRF rejected
                    # ---------------------------------
                    response = client.post("/auth/logout")

                    self.assertEqual(
                        response.status_code, 403
                    )
                    print("8. Unsafe logout blocked: PASS")

                    # ---------------------------------
                    # 9. Logout with CSRF succeeds
                    # ---------------------------------
                    response = client.post(
                        "/auth/logout",
                        headers={
                            "X-CSRF-Token": old_csrf
                        }
                    )

                    self.assertEqual(
                        response.status_code, 200
                    )

                    self.assertTrue(
                        test_store.is_revoked(old_token)
                    )

                    print("9. Session revoked on logout: PASS")

                    # ---------------------------------
                    # 10. Replay the old session
                    # ---------------------------------
                    client.cookies.set(
                        SESSION_COOKIE_NAME,
                        old_token
                    )

                    response = client.get("/documents")

                    self.assertEqual(
                        response.status_code, 401
                    )
                    print("10. Old cookie replay blocked: PASS")

                    # ---------------------------------
                    # 11. Revoked /auth/me
                    # ---------------------------------
                    response = client.get("/auth/me")

                    self.assertEqual(
                        response.status_code, 401
                    )
                    print("11. Revoked /auth/me blocked: PASS")

                    # ---------------------------------
                    # 12. Revoked /auth/csrf
                    # ---------------------------------
                    response = client.get("/auth/csrf")

                    self.assertEqual(
                        response.status_code, 401
                    )
                    print("12. Revoked CSRF access blocked: PASS")

                    # ---------------------------------
                    # 13. Old session + old CSRF
                    # ---------------------------------
                    response = client.post(
                        "/rubrics",
                        json={},
                        headers={
                            "X-CSRF-Token": old_csrf
                        }
                    )

                    self.assertEqual(
                        response.status_code, 401
                    )
                    print("13. Old session and CSRF blocked: PASS")

                    
                    # ---------------------------------
                    # 14. Login again
                    # ---------------------------------

                    # Remove the manually replayed, revoked cookie.
                    # This resets the test browser's cookie storage,
                    # but does NOT remove revocation from SQLite.
                    client.cookies.clear()

                    response = client.post(
                        "/auth/login",
                        json={
                            "password": "test-only-password"
                        }
                    )


                    self.assertEqual(
                        response.status_code, 200
                    )

                    new_token = response.cookies.get(
                        SESSION_COOKIE_NAME
                    )

                    self.assertIsNotNone(new_token)

                    self.assertNotEqual(
                        new_token,
                        old_token
                    )

                    print("14. New login successful: PASS")

                    # ---------------------------------
                    # 15. New session works
                    # ---------------------------------
                    response = client.get("/documents")

                    self.assertEqual(
                        response.status_code, 200
                    )

                    print("15. New session accepted: PASS")

        print("\n=== ALL 15 SECURITY CHECKS PASSED ===")


if __name__ == "__main__":
    unittest.main(verbosity=2)
