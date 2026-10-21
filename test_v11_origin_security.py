
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app


class OriginSecurityTests(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_trusted_local_origin(self):
        response = self.client.post(
            "/rubrics",
            json={},
            headers={
                "Origin": "http://127.0.0.1:8000"
            }
        )

        # Origin passes; anonymous authentication
        # still rejects the request.
        self.assertEqual(response.status_code, 401)

    def test_trusted_test_origin(self):
        response = self.client.post(
            "/rubrics",
            json={},
            headers={
                "Origin": "http://testserver"
            }
        )

        self.assertEqual(response.status_code, 401)

    def test_untrusted_origin_rejected(self):
        response = self.client.post(
            "/rubrics",
            json={},
            headers={
                "Origin": "https://attacker.example"
            }
        )

        self.assertEqual(response.status_code, 403)

    def test_null_origin_rejected(self):
        response = self.client.post(
            "/rubrics",
            json={},
            headers={
                "Origin": "null"
            }
        )

        self.assertEqual(response.status_code, 403)

    def test_origin_prefix_attack_rejected(self):
        response = self.client.post(
            "/rubrics",
            json={},
            headers={
                "Origin": "http://localhost:8000.attacker.example"
            }
        )

        self.assertEqual(response.status_code, 403)

    def test_untrusted_login_origin_rejected(self):
        response = self.client.post(
            "/auth/login",
            json={},
            headers={
                "Origin": "https://attacker.example"
            }
        )

        self.assertEqual(response.status_code, 403)

    def test_safe_get_not_blocked_by_origin(self):
        response = self.client.get(
            "/health",
            headers={
                "Origin": "https://attacker.example"
            }
        )

        self.assertEqual(response.status_code, 200)

    def test_missing_origin_preserves_existing_behavior(self):
        response = self.client.post(
            "/rubrics",
            json={}
        )

        self.assertEqual(response.status_code, 401)

    def test_configured_production_origin(self):
        with patch.dict(
            os.environ,
            {
                "CHAT_SCHOLAR_ALLOWED_ORIGINS":
                    "https://scholar.example.com"
            }
        ):
            response = self.client.post(
                "/rubrics",
                json={},
                headers={
                    "Origin": "https://scholar.example.com"
                }
            )

            self.assertEqual(response.status_code, 401)

            response = self.client.post(
                "/rubrics",
                json={},
                headers={
                    "Origin": "http://localhost:8000"
                }
            )

            self.assertEqual(response.status_code, 403)

    def test_wildcard_configuration_rejected(self):
        with patch.dict(
            os.environ,
            {"CHAT_SCHOLAR_ALLOWED_ORIGINS": "*"}
        ):
            response = self.client.post(
                "/rubrics",
                json={}
            )

            self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main(verbosity=2)
