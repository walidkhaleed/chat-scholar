
import unittest

from fastapi.testclient import TestClient
from fastapi.routing import APIRoute

from main import app
from auth_security import (
    SESSION_COOKIE_NAME,
    create_session_token,
)


class ChatScholarCompatibilityTests(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(
            app,
            raise_server_exceptions=False
        )
        self.addCleanup(self.client.close)

    def login_for_test(self):
        # Create a valid signed test session.
        # No real password is needed.
        token = create_session_token()

        self.client.cookies.set(
            SESSION_COOKIE_NAME,
            token
        )

    def test_health_endpoint(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)

    def test_web_interface(self):
        response = self.client.get("/app")

        self.assertEqual(response.status_code, 200)

        html = response.text

        # Existing V10 dashboard elements.
        for element_id in [
            "view-chat",
            "view-grading",
            "view-saved",
            "pdf-upload-form",
            "rubric-form",
            "essay-form",
        ]:
            self.assertIn(
                f'id="{element_id}"',
                html
            )

        # V11 authentication elements.
        for element_id in [
            "auth-screen",
            "login-form",
            "logout-button",
        ]:
            self.assertIn(
                f'id="{element_id}"',
                html
            )

    def test_static_files(self):
        for path in [
            "/static/style.css",
            "/static/app.js",
        ]:
            response = self.client.get(path)

            self.assertEqual(
                response.status_code,
                200,
                f"Static file failed: {path}"
            )

    def test_private_endpoints_require_login(self):
        for path in [
            "/documents",
            "/pdf-sessions",
            "/rubrics",
        ]:
            response = self.client.get(path)

            self.assertEqual(
                response.status_code,
                401,
                f"Anonymous access allowed: {path}"
            )

    def test_private_endpoints_after_authentication(self):
        self.login_for_test()

        for path in [
            "/documents",
            "/pdf-sessions",
            "/rubrics",
        ]:
            response = self.client.get(path)

            self.assertEqual(
                response.status_code,
                200,
                f"Authenticated request failed: {path}"
            )

    def test_original_api_routes_preserved(self):
        registered = {
            route.path
            for route in app.routes
            if isinstance(route, APIRoute)
        }

        expected = {
            "/health",
            "/chat",
            "/upload-pdf",
            "/documents",
            "/prepare-document",
            "/semantic-search",
            "/advanced-pdf-chat",
            "/pdf-sessions",
            "/pdf-chat-memory",
            "/rubrics",
            "/essays/upload",
            "/essays/text",
            "/grade-essay",
            "/grades/{grade_id}",
            "/app",
        }

        missing = expected - registered

        self.assertFalse(
            missing,
            f"Missing API routes: {sorted(missing)}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
