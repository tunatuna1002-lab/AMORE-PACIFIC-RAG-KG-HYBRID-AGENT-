"""CSRF protection middleware for state-changing endpoints."""

import hashlib
import hmac
import os
import secrets
from collections.abc import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

CSRF_SECRET = os.environ.get("CSRF_SECRET", secrets.token_hex(32))
CSRF_HEADER = "X-CSRF-Token"
CSRF_COOKIE = "csrf_token"
SAFE_METHODS = {"GET", "HEAD", "OPTIONS", "TRACE"}


def generate_csrf_token(session_id: str = "") -> str:
    """Generate a CSRF token tied to a session."""
    payload = f"{session_id}:{secrets.token_hex(16)}"
    signature = hmac.new(CSRF_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}:{signature}"


def validate_csrf_token(token: str) -> bool:
    """Validate a CSRF token's signature."""
    if not token or ":" not in token:
        return False
    parts = token.rsplit(":", 1)
    if len(parts) != 2:
        return False
    payload, signature = parts
    expected = hmac.new(CSRF_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, expected)


class CSRFMiddleware(BaseHTTPMiddleware):
    """CSRF protection for dashboard form submissions.

    Only enforced for non-API state-changing requests from browsers.
    API endpoints using API keys are exempt.
    """

    def __init__(self, app, exempt_paths: list[str] | None = None):
        super().__init__(app)
        self.exempt_paths = exempt_paths or ["/api/"]

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip safe methods
        if request.method in SAFE_METHODS:
            response = await call_next(request)
            # Set CSRF cookie on GET requests to dashboard
            if request.url.path.startswith("/dashboard"):
                token = generate_csrf_token()
                response.set_cookie(
                    CSRF_COOKIE, token, httponly=False, samesite="strict", secure=False
                )
            return response

        # Skip exempt paths (API endpoints with API key auth)
        if any(request.url.path.startswith(p) for p in self.exempt_paths):
            return await call_next(request)

        # Validate CSRF token for state-changing requests
        token = request.headers.get(CSRF_HEADER, "")
        if not token:
            # Also check form data
            token = request.cookies.get(CSRF_COOKIE, "")

        if not validate_csrf_token(token):
            from fastapi.responses import JSONResponse

            return JSONResponse(
                status_code=403,
                content={"detail": "CSRF validation failed"},
            )

        return await call_next(request)
