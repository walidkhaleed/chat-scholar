
import os

from fastapi import Request
from fastapi.responses import JSONResponse

from auth_security import (
    SESSION_COOKIE_NAME,
    verify_session_token,
)

from csrf_security import (
    CSRF_HEADER_NAME,
    verify_csrf_token,
)
import sqlite3

from session_revocation import session_store

# Requests that can modify server-side data.
UNSAFE_METHODS = {
    "POST",
    "PUT",
    "PATCH",
    "DELETE",
}

# Login is exempt because there is no authenticated
# session from which to obtain a CSRF token.
CSRF_EXEMPT_PATHS = {
    "/auth/login",
}



def csrf_enforcement_enabled() -> bool:
    """
    CSRF protection is enabled by default.

    Missing or unrecognized configuration values
    must not silently disable protection.

    Explicit 'false' remains available for
    controlled local development only.
    Do not use 'false' in production.
    """
    value = os.getenv(
        "CHAT_SCHOLAR_CSRF_ENFORCE",
        "true"
    ).strip().lower()

    return value != "false"


def allowed_request_origins() -> set[str]:
    """
    Return explicitly trusted browser origins.

    Local development origins are used by default.
    Configure the production HTTPS origin before
    exposing Chat Scholar publicly.
    """
    raw = os.getenv(
        "CHAT_SCHOLAR_ALLOWED_ORIGINS",
        (
            "http://127.0.0.1:8000,"
            "http://localhost:8000,"
            "http://testserver"
        )
    )

    origins = {
        origin.strip()
        for origin in raw.split(",")
        if origin.strip()
    }

    if not origins or "*" in origins or "null" in origins:
        raise RuntimeError(
            "Invalid CHAT_SCHOLAR_ALLOWED_ORIGINS configuration."
        )

    return origins

async def csrf_guard(
    request: Request,
    call_next
):


    # Validate the Origin header on state-changing
    # requests, including the login endpoint.
    if request.method in UNSAFE_METHODS:

        try:
            trusted_origins = allowed_request_origins()

        except RuntimeError:
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "Origin security unavailable"
                },
                headers={
                    "Cache-Control": "no-store"
                }
            )

        request_origin = request.headers.get("origin")

        # Requests without Origin retain existing
        # authentication and CSRF requirements.
        if (
            request_origin is not None
            and request_origin not in trusted_origins
        ):
            return JSONResponse(
                status_code=403,
                content={
                    "detail": "Untrusted request origin"
                },
                headers={
                    "Cache-Control": "no-store"
                }
            )

    # Keep enforcement disabled until the frontend
    # has been upgraded to send CSRF tokens.
    if not csrf_enforcement_enabled():
        return await call_next(request)

    # Safe HTTP methods do not need CSRF tokens.
    if request.method not in UNSAFE_METHODS:
        return await call_next(request)

    # Login cannot require a session-bound CSRF token.
    if request.url.path in CSRF_EXEMPT_PATHS:
        return await call_next(request)

    session_token = request.cookies.get(
        SESSION_COOKIE_NAME
    )



    try:
        signed_session = verify_session_token(
            session_token
        )

        if not signed_session:
            return await call_next(request)

        # Reject sessions revoked in SQLite.
        if session_store.is_revoked(session_token):
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Session revoked. Please sign in again."
                },
                headers={
                    "Cache-Control": "no-store"
                }
            )

    except (RuntimeError, sqlite3.Error, OSError):
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Authentication unavailable"
            },
            headers={
                "Cache-Control": "no-store"
            }
        )

    # Only active sessions reach CSRF verification.
    supplied_token = request.headers.get(
        CSRF_HEADER_NAME,
        ""
    )

    if not verify_csrf_token(
        session_token,
        supplied_token
    ):
        return JSONResponse(
            status_code=403,
            content={
                "detail": "CSRF token missing or invalid"
            },
            headers={
                "Cache-Control": "no-store"
            }
        )

    return await call_next(request)

