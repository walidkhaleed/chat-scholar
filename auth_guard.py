
from fastapi import Request
from fastapi.responses import JSONResponse

from auth_security import (
    SESSION_COOKIE_NAME,
    verify_session_token,
)

import sqlite3

from session_revocation import session_store
# Only these exact routes are allowed without
# authentication.
PUBLIC_PATHS = {
    "/",
    "/health",
    "/app",
    "/auth/login",
    "/auth/logout",
    "/auth/me",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/docs/oauth2-redirect",
}


async def authentication_guard(
    request: Request,
    call_next
):
    path = request.url.path

    # Public endpoints.
    if path in PUBLIC_PATHS:
        return await call_next(request)

    # Frontend assets remain accessible so the
    # future login page can load CSS and JavaScript.
    if path.startswith("/static/"):
        return await call_next(request)

    # Everything else requires authentication.
    token = request.cookies.get(
        SESSION_COOKIE_NAME
    )

    
    try:
        # Check both the cryptographic signature
        # and the SQLite revocation database.
        authenticated = session_store.is_active(token)

    except (RuntimeError, sqlite3.Error, OSError):
        # Fail closed if the authentication secret
        # or revocation database is unavailable.
        return JSONResponse(
            status_code=503,
            content={
                "detail": "Authentication unavailable"
            },
            headers={
                "Cache-Control": "no-store"
            }
        )


    if not authenticated:
        return JSONResponse(
            status_code=401,
            content={
                "detail": "Authentication required"
            },
            headers={"Cache-Control": "no-store"},
        )

    return await call_next(request)
