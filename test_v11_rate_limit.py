
import unittest

from login_rate_limit import LoginRateLimiter


class LoginRateLimitTests(unittest.TestCase):

    def setUp(self):
        self.limiter = LoginRateLimiter(
            max_attempts=5,
            window_seconds=900
        )

    def test_first_five_attempts_allowed(self):

        for attempt in range(5):

            result = self.limiter.consume(
                "127.0.0.1",
                now=100 + attempt
            )

            self.assertEqual(result, 0)

    def test_sixth_attempt_blocked(self):

        for attempt in range(5):
            self.limiter.consume(
                "127.0.0.1",
                now=100 + attempt
            )

        result = self.limiter.consume(
            "127.0.0.1",
            now=105
        )

        self.assertGreater(result, 0)

    def test_retry_after_seconds(self):

        for _ in range(5):
            self.limiter.consume(
                "127.0.0.1",
                now=100
            )

        result = self.limiter.consume(
            "127.0.0.1",
            now=400
        )

        self.assertEqual(result, 600)

    def test_attempts_expire_after_fifteen_minutes(self):

        for _ in range(5):
            self.limiter.consume(
                "127.0.0.1",
                now=100
            )

        result = self.limiter.consume(
            "127.0.0.1",
            now=1001
        )

        self.assertEqual(result, 0)

    def test_different_clients_are_independent(self):

        for _ in range(5):
            self.limiter.consume(
                "192.0.2.10",
                now=100
            )

        blocked = self.limiter.consume(
            "192.0.2.10",
            now=101
        )

        allowed = self.limiter.consume(
            "192.0.2.11",
            now=101
        )

        self.assertGreater(blocked, 0)
        self.assertEqual(allowed, 0)

    def test_successful_login_resets_counter(self):

        for _ in range(5):
            self.limiter.consume(
                "127.0.0.1",
                now=100
            )

        self.limiter.reset("127.0.0.1")

        result = self.limiter.consume(
            "127.0.0.1",
            now=101
        )

        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
