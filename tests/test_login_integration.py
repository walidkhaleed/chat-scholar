
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app
from login_rate_limit import LoginRateLimiter


class LoginIntegrationTests(unittest.TestCase):

    def test_login_rate_limiting(self):

        # Use a fresh limiter for this test only.
        test_limiter = LoginRateLimiter(
            max_attempts=5,
            window_seconds=900
        )

        with patch(
            "auth_routes.login_limiter",
            test_limiter
        ):
            with TestClient(app) as client:

                # First five incorrect passwords return 401.
                for attempt in range(1, 6):

                    response = client.post(
                        "/auth/login",
                        json={
                            "password": "incorrect-test-password"
                        }
                    )

                    print(
                        f"Attempt {attempt}: "
                        f"HTTP {response.status_code}"
                    )

                    self.assertEqual(
                        response.status_code,
                        401
                    )

                # Sixth attempt must be blocked.
                response = client.post(
                    "/auth/login",
                    json={
                        "password": "incorrect-test-password"
                    }
                )

                print(
                    f"Attempt 6: "
                    f"HTTP {response.status_code}"
                )

                self.assertEqual(
                    response.status_code,
                    429
                )

                # Verify the Retry-After header.
                retry_after = response.headers.get(
                    "Retry-After"
                )

                self.assertIsNotNone(retry_after)
                self.assertGreater(
                    int(retry_after),
                    0
                )

                print(
                    "Retry-After:",
                    retry_after,
                    "seconds"
                )

                print(
                    "Brute-force protection: PASS"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
