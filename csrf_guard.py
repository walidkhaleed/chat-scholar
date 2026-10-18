
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
    Activation switch.

    False until the frontend is ready in Step 10C.
    Set CHAT_SCHOLAR_CSRF_ENFORCE=true to activate.
    """
    return (
        os.getenv(
            "CHAT_SCHOLAR_CSRF_ENFORCE",
            "false"
        ).strip().lower() == "true"
    )


async def csrf_guard(
    request: Request,
    call_next
):
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

