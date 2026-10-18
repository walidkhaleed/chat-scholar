
import os
import unittest
from unittest.mock import patch

from auth_security import create_session_token

from csrf_security import (
    create_csrf_token,
    verify_csrf_token,
)


class CSRFProtectionTests(unittest.TestCase):

    def setUp(self):
        # Use an isolated test secret.
        # Never use or print your real .env secret.
        self.secret_patch = patch.dict(
            os.environ,
            {
                "CHAT_SCHOLAR_SESSION_SECRET":
                    "v11-test-secret-" + "A" * 48
            }
        )

        self.secret_patch.start()
        self.addCleanup(self.secret_patch.stop)

        self.session = create_session_token()
        self.csrf = create_csrf_token(self.session)

    def test_valid_csrf_token(self):
        self.assertTrue(
            verify_csrf_token(
                self.session,
                self.csrf
            )
        )

    def test_missing_csrf_token(self):
        self.assertFalse(
            verify_csrf_token(
                self.session,
                ""
            )
        )

    def test_invalid_csrf_token(self):
        self.assertFalse(
            verify_csrf_token(
                self.session,
                "invalid-token"
            )
        )

    def test_modified_csrf_token(self):
        modified = (
            ("A" if self.csrf[0] != "A" else "B")
            + self.csrf[1:]
        )

        self.assertFalse(
            verify_csrf_token(
                self.session,
                modified
            )
        )

    def test_different_session_rejected(self):
        another_session = create_session_token()

        self.assertNotEqual(
            self.session,
            another_session
        )

        self.assertFalse(
            verify_csrf_token(
                another_session,
                self.csrf
            )
        )

    def test_invalid_session_rejected(self):
        self.assertFalse(
            verify_csrf_token(
                "invalid-session",
                self.csrf
            )
        )

    def test_cannot_create_token_anonymously(self):
        with self.assertRaises(ValueError):
            create_csrf_token("")

    def test_token_is_deterministic_for_session(self):
        second_token = create_csrf_token(
            self.session
        )

        self.assertEqual(
            self.csrf,
            second_token
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
