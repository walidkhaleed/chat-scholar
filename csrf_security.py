
import base64
import hashlib
import hmac

from auth_security import (
    get_session_secret,
    verify_session_token,
)


CSRF_HEADER_NAME = "X-CSRF-Token"

# Domain separation: ensures the CSRF signature
# cannot be confused with a login session signature.
CSRF_CONTEXT = b"chat-scholar-csrf-v1:"


def create_csrf_token(session_token: str) -> str:
    """
    Generate a CSRF token bound to a valid
    authenticated session.

    A different session receives a different token.
    """

    if not verify_session_token(session_token):
        raise ValueError(
            "A valid login session is required."
        )

    secret = get_session_secret()

    message = (
        CSRF_CONTEXT
        + session_token.encode("utf-8")
    )

    signature = hmac.new(
        secret,
        message,
        hashlib.sha256
    ).digest()

    return (
        base64.urlsafe_b64encode(signature)
        .decode("ascii")
        .rstrip("=")
    )


def verify_csrf_token(
    session_token: str,
    csrf_token: str
) -> bool:
    """
    Verify that a CSRF token belongs to
    the current authenticated session.

    Returns False for missing, malformed,
    or incorrect tokens.
    """

    if not isinstance(csrf_token, str):
        return False

    # SHA-256 Base64URL without padding is 43 chars.
    if len(csrf_token) != 43:
        return False

    try:
        csrf_token.encode("ascii")

        expected = create_csrf_token(
            session_token
        )

        return hmac.compare_digest(
            csrf_token,
            expected
        )

    except (
        ValueError,
        TypeError,
        UnicodeError
    ):
        return False
