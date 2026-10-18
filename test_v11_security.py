
import unittest

from fastapi.testclient import TestClient

from main import app
from auth_security import (
    SESSION_COOKIE_NAME,
    create_session_token,
)


class ChatScholarSecurityTests(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def authenticate(self):
        token = create_session_token()

        self.client.cookies.set(
            SESSION_COOKIE_NAME,
            token
        )

    def test_environment_file_not_exposed(self):
        # Anonymous access must not succeed.
        response = self.client.get("/.env")

        self.assertIn(
            response.status_code,
            (401, 403, 404)
        )

        # Authentication must not grant access
        # to the application's .env file.
        self.authenticate()

        response = self.client.get("/.env")

        self.assertIn(
            response.status_code,
            (403, 404),
            "Security issue: authenticated user can "
            "potentially access the .env file."
        )

    def test_environment_file_not_exposed_via_static(self):
        # Public static-file handling must never
        # expose the environment file.
        response = self.client.get("/static/.env")

        self.assertIn(
            response.status_code,
            (403, 404)
        )

        self.authenticate()

        response = self.client.get("/static/.env")

        self.assertIn(
            response.status_code,
            (403, 404)
        )

    def test_private_uploads_not_public(self):
        # Anonymous requests cannot access files
        # using the uploads directory URL.
        response = self.client.get(
            "/uploads/test.pdf"
        )

        self.assertIn(
            response.status_code,
            (401, 403, 404)
        )

        # Authentication must not directly expose
        # the private uploads directory.
        self.authenticate()

        response = self.client.get(
            "/uploads/test.pdf"
        )

        self.assertIn(
            response.status_code,
            (403, 404),
            "Security issue: authenticated user can "
            "potentially access private uploaded files."
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
