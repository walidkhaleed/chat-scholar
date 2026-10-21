
import unittest

from fastapi.testclient import TestClient
from main import app


class TrustedHostSecurityTests(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def test_localhost_allowed(self):
        response = self.client.get(
            "/health",
            headers={"Host": "localhost:8000"}
        )

        self.assertEqual(response.status_code, 200)

    def test_loopback_allowed(self):
        response = self.client.get(
            "/health",
            headers={"Host": "127.0.0.1:8000"}
        )

        self.assertEqual(response.status_code, 200)

    def test_testserver_allowed(self):
        response = self.client.get(
            "/health",
            headers={"Host": "testserver"}
        )

        self.assertEqual(response.status_code, 200)

    def test_malicious_host_rejected(self):
        response = self.client.get(
            "/health",
            headers={"Host": "malicious.example"}
        )

        self.assertEqual(response.status_code, 400)

    def test_other_untrusted_host_rejected(self):
        response = self.client.get(
            "/health",
            headers={"Host": "attacker.example"}
        )

        self.assertEqual(response.status_code, 400)

    def test_untrusted_host_rejected_on_login(self):
        response = self.client.post(
            "/auth/login",
            json={"password": "test-only-password"},
            headers={"Host": "malicious.example"}
        )

        # The request must be rejected before
        # it reaches the login endpoint.
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main(verbosity=2)
