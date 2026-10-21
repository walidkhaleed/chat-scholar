
import os
import unittest
from unittest.mock import patch

from csrf_guard import csrf_enforcement_enabled


class CSRFDefaultTests(unittest.TestCase):

    def test_enabled_when_not_configured(self):
        with patch.dict(os.environ):
            os.environ.pop(
                "CHAT_SCHOLAR_CSRF_ENFORCE",
                None
            )

            self.assertTrue(
                csrf_enforcement_enabled()
            )

    def test_enabled_when_true(self):
        with patch.dict(
            os.environ,
            {"CHAT_SCHOLAR_CSRF_ENFORCE": "true"}
        ):
            self.assertTrue(
                csrf_enforcement_enabled()
            )

    def test_enabled_for_invalid_configuration(self):
        with patch.dict(
            os.environ,
            {"CHAT_SCHOLAR_CSRF_ENFORCE": "tru"}
        ):
            self.assertTrue(
                csrf_enforcement_enabled()
            )

    def test_explicit_false_disables(self):
        with patch.dict(
            os.environ,
            {"CHAT_SCHOLAR_CSRF_ENFORCE": "false"}
        ):
            self.assertFalse(
                csrf_enforcement_enabled()
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
