
import os
import subprocess
import sys
import unittest


# Test-only secret. Never uses your real signing secret.
TEST_SECRET = "v12-test-secret-" + ("x" * 64)


def run_in_new_process(code, ephemeral):
    """Run authentication code in a fresh Python process."""

    env = os.environ.copy()

    env["CHAT_SCHOLAR_SESSION_SECRET"] = TEST_SECRET
    env["CHAT_SCHOLAR_EPHEMERAL_SESSIONS"] = (
        "true" if ephemeral else "false"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )

    return result.stdout.strip()


class EphemeralSessionTests(unittest.TestCase):

    def test_01_ephemeral_session_works_in_same_process(self):
        result = run_in_new_process(
            """
from auth_security import (
    create_session_token,
    verify_session_token,
)

token = create_session_token()
print(verify_session_token(token))
""",
            ephemeral=True,
        )

        self.assertEqual(result, "True")

    def test_02_restart_invalidates_ephemeral_session(self):
        old_token = run_in_new_process(
            """
from auth_security import create_session_token
print(create_session_token())
""",
            ephemeral=True,
        )

        # This is a different process, simulating a restart.
        result = run_in_new_process(
            """
import sys
from auth_security import verify_session_token

print(verify_session_token(sys.argv[1]))
""".replace(
                "sys.argv[1]",
                repr(old_token),
            ),
            ephemeral=True,
        )

        self.assertEqual(result, "False")

    def test_03_local_mode_preserves_sessions(self):
        token = run_in_new_process(
            """
from auth_security import create_session_token
print(create_session_token())
""",
            ephemeral=False,
        )

        result = run_in_new_process(
            f"""
from auth_security import verify_session_token
print(verify_session_token({token!r}))
""",
            ephemeral=False,
        )

        self.assertEqual(result, "True")

    def test_04_invalid_signature_rejected(self):
        result = run_in_new_process(
            """
from auth_security import (
    create_session_token,
    verify_session_token,
)

token = create_session_token()
modified = token[:-1] + (
    "A" if token[-1] != "A" else "B"
)

print(verify_session_token(modified))
""",
            ephemeral=True,
        )

        self.assertEqual(result, "False")


if __name__ == "__main__":
    unittest.main(verbosity=2)
