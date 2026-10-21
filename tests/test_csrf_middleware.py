
import os
import unittest

from unittest.mock import patch
from fastapi.testclient import TestClient

from main import app

from auth_security import (
    SESSION_COOKIE_NAME,
    create_session_token,
)

from csrf_security import create_csrf_token


class CSRFEnforcementTests(unittest.TestCase):

    def setUp(self):
        self.environment = patch.dict(
            os.environ,
            {
                "CHAT_SCHOLAR_SESSION_SECRET":
                    "test-only-secret-" + "A" * 48,
                "CHAT_SCHOLAR_CSRF_ENFORCE": "true",
                "CHAT_SCHOLAR_COOKIE_SECURE": "false",
            }
        )

        self.environment.start()
        self.addCleanup(self.environment.stop)

        self.client_context = TestClient(
            app,
            raise_server_exceptions=False
        )

        self.client = self.client_context.__enter__()

        self.addCleanup(
            self.client_context.__exit__,
            None,
            None,
            None
        )

        self.session = create_session_token()
        self.csrf = create_csrf_token(self.session)

    def authenticate(self):
        self.client.cookies.set(
            SESSION_COOKIE_NAME,
            self.session
        )

    def test_anonymous_post_rejected(self):
        response = self.client.post(
            "/rubrics",
            json={}
        )

        self.assertEqual(
            response.status_code,
            401
        )

    def test_missing_csrf_rejected(self):
        self.authenticate()

        response = self.client.post(
            "/rubrics",
            json={}
        )

        self.assertEqual(
            response.status_code,
            403
        )

    def test_invalid_csrf_rejected(self):
        self.authenticate()

        response = self.client.post(
            "/rubrics",
            json={},
            headers={
                "X-CSRF-Token": "invalid-token"
            }
        )

        self.assertEqual(
            response.status_code,
            403
        )

    def test_valid_csrf_passes_middleware(self):
        self.authenticate()

        response = self.client.post(
            "/rubrics",
            json={},
            headers={
                "X-CSRF-Token": self.csrf
            }
        )

        # 422 means the request passed CSRF protection
        # and reached normal rubric payload validation.
        # The intentionally empty rubric is invalid,
        # so no rubric should be created.
        self.assertEqual(
            response.status_code,
            422
        )

    def test_get_does_not_require_csrf(self):
        self.authenticate()

        response = self.client.get("/rubrics")

        self.assertEqual(
            response.status_code,
            200
        )

    def test_logout_requires_csrf(self):
        self.authenticate()

        response = self.client.post(
            "/auth/logout"
        )

        self.assertEqual(
            response.status_code,
            403
        )

        response = self.client.post(
            "/auth/logout",
            headers={
                "X-CSRF-Token": self.csrf
            }
        )

        self.assertEqual(
            response.status_code,
            200
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
