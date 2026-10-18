
import os

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from login_rate_limit import login_limiter

from auth_security import (
    SESSION_COOKIE_NAME,
    SESSION_DURATION_SECONDS,
    create_session_token,
    verify_password,
    verify_session_token,
)

from csrf_security import create_csrf_token
from fastapi.responses import JSONResponse

import sqlite3

from session_revocation import session_store

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"]
)


class LoginRequest(BaseModel):
    password: str = Field(min_length=1, max_length=1024)


def use_secure_cookies() -> bool:
    """
    Secure cookies require HTTPS.

    During local HTTP development, configure:
        CHAT_SCHOLAR_COOKIE_SECURE=false

    The default is True for safer deployment.
    """
    return (
        os.getenv(
            "CHAT_SCHOLAR_COOKIE_SECURE", "true"
        ).lower() == "true"
    )


def require_active_session(token: str | None):
    """
    Require a valid, unexpired, non-revoked session.

    A revoked token is rejected even when its
    signature and expiration are otherwise valid.
    """
    try:
        active = session_store.is_active(token)

    except (RuntimeError, sqlite3.Error, OSError) as exc:
        raise HTTPException(
            status_code=503,
            detail="Authentication unavailable"
        ) from exc

    if not active:
        raise HTTPException(
            status_code=401,
            detail="Authentication required"
        )


@router.post("/login")
def login(
    data: LoginRequest,
    request: Request,
    response: Response
):
    # Identify the connecting client.
    # Do not trust client-supplied X-Forwarded-For headers.
    client_ip = (
        request.client.host
        if request.client
        else "unknown"
    )

    # Check the rate limit before verifying the password.
    retry_after = login_limiter.consume(client_ip)

    if retry_after > 0:
        raise HTTPException(
            status_code=429,
            detail=(
                "Too many login attempts. "
                "Please try again later."
            ),
            headers={
                "Retry-After": str(retry_after)
            }
        )

    # Verify the password.
    if not verify_password(data.password):
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

    # Successful login: reset this client's counter.
    login_limiter.reset(client_ip)

    # Create the signed login session.
    token = create_session_token()

    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=use_secure_cookies(),
        samesite="strict",
        max_age=SESSION_DURATION_SECONDS,
        path="/"
    )

    return {
        "authenticated": True,
        "message": "Login successful"
    }



@router.get("/me")
def current_user(request: Request):

    token = request.cookies.get(
        SESSION_COOKIE_NAME
    )

    require_active_session(token)

    return {
        "authenticated": True
    }


@router.post("/logout")
def logout(request: Request, response: Response):
    """
    Revoke the current session on the server,
    then remove the browser's authentication cookie.
    """

    token = request.cookies.get(
        SESSION_COOKIE_NAME
    )

    try:
        # Revoke any currently valid signed session.
        if token and verify_session_token(token):
            session_store.revoke(token)

    except (RuntimeError, sqlite3.Error, OSError):
        # Do not claim successful server-side logout
        # when revocation could not be saved.
        raise HTTPException(
            status_code=503,
            detail="Unable to revoke session. Please try again."
        )

    # Clear the browser cookie after revocation.
    response.delete_cookie(
        key=SESSION_COOKIE_NAME,
        path="/",
        secure=use_secure_cookies(),
        httponly=True,
        samesite="strict"
    )

    return {
        "authenticated": False,
        "message": "Logged out and session revoked"
    }



# ---------------------------------------------
# V11 — CSRF Token Endpoint
# ---------------------------------------------

@router.get("/csrf")
def get_csrf_token(request: Request):

    # Read the authenticated session cookie.
    session_token = request.cookies.get(
        SESSION_COOKIE_NAME
    )

    require_active_session(session_token)

    # Create a token bound to this login session.
    csrf_token = create_csrf_token(
        session_token
    )

    return JSONResponse(
        content={
            "csrf_token": csrf_token
        },
        headers={
            "Cache-Control": "no-store"
        }
    )
