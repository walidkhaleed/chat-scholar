
import base64
import hashlib
import secrets
from getpass import getpass
from pathlib import Path

from dotenv import set_key


ENV_PATH = Path(__file__).resolve().parent / ".env"
ITERATIONS = 600_000


def create_password_hash(password):
    salt = secrets.token_bytes(16)

    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        ITERATIONS
    )

    salt_encoded = base64.b64encode(salt).decode()
    hash_encoded = base64.b64encode(digest).decode()

    return (
        f"pbkdf2_sha256${ITERATIONS}$"
        f"{salt_encoded}${hash_encoded}"
    )


def main():
    if not ENV_PATH.exists():
        print("ERROR: .env file not found.")
        return

    print("=== Chat Scholar Authentication Setup ===")

    password = getpass("Create login password: ")
    confirmation = getpass("Confirm login password: ")

    if len(password) < 12:
        print("ERROR: Use at least 12 characters.")
        return

    if password != confirmation:
        print("ERROR: Passwords do not match.")
        return

    password_hash = create_password_hash(password)
    session_secret = secrets.token_urlsafe(48)

    set_key(
        str(ENV_PATH),
        "CHAT_SCHOLAR_PASSWORD_HASH",
        password_hash
    )

    set_key(
        str(ENV_PATH),
        "CHAT_SCHOLAR_SESSION_SECRET",
        session_secret
    )

    print()
    print("Authentication credentials created.")
    print("Password stored as salted PBKDF2 hash.")
    print("Session signing secret generated.")
    print("Saved securely in local .env.")
    print("Do not share or commit the .env file.")


if __name__ == "__main__":
    main()
