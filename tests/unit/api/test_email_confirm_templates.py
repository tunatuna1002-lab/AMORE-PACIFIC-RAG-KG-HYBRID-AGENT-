"""
Email confirmation pages (``GET /api/alerts/confirm-email``).
============================================================

The two landing pages moved out of ``src/api/routes/alerts.py`` into
``src/api/templates/*.html``. The error page already had a test; the **success**
page had none, so the branch that renders it was uncovered. These tests drive
both pages through the real route with a real signed token.
"""

from __future__ import annotations

import pytest

from src.api.templates import render


@pytest.fixture
def jwt_secret(monkeypatch):
    monkeypatch.setenv("JWT_SECRET_KEY", "test-secret-for-email-confirmation")
    return "test-secret-for-email-confirmation"


def _token(email: str) -> str:
    from src.api.dependencies import create_email_verification_token

    return create_email_verification_token(email)


# ---------------------------------------------------------------------------
# Route: success page
# ---------------------------------------------------------------------------


def test_confirm_email_page_renders_the_success_page_for_a_valid_token(
    client, isolated_cwd, jwt_secret
):
    email = "user@example.com"
    r = client.get("/api/alerts/confirm-email", params={"token": _token(email), "email": email})

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "이메일 인증 완료!" in r.text
    assert email in r.text
    # the template is a complete document, not a fragment
    assert r.text.lstrip().startswith("<!DOCTYPE html>")
    assert r.text.rstrip().endswith("</html>")


def test_confirm_email_page_rejects_a_token_issued_for_another_address(
    client, isolated_cwd, jwt_secret
):
    r = client.get(
        "/api/alerts/confirm-email",
        params={"token": _token("owner@example.com"), "email": "someone@else.com"},
    )

    assert r.status_code == 400
    assert "이메일이 일치하지 않습니다." in r.text


def test_confirm_email_page_shows_the_reason_on_the_error_page(client, isolated_cwd, jwt_secret):
    r = client.get("/api/alerts/confirm-email", params={"token": "nope", "email": "a@b.c"})

    assert r.status_code == 400
    assert "인증 실패" in r.text
    # the error slot is filled in, not left as a literal placeholder
    assert "$error_message" not in r.text


# ---------------------------------------------------------------------------
# Template loader
# ---------------------------------------------------------------------------


def test_render_substitutes_named_placeholders():
    out = render("email_confirm_success", email="a@b.c")

    assert "a@b.c" in out
    assert "$email" not in out


def test_render_leaves_a_stray_dollar_sign_in_the_value_alone():
    """safe_substitute: interpolated content is data, never re-scanned as a template."""
    out = render("email_confirm_error", error_message="cost: $5 and $missing")

    assert "cost: $5 and $missing" in out


def test_render_is_cached_across_calls():
    """The file is parsed once; repeated renders return equal output."""
    first = render("email_confirm_error", error_message="x")
    second = render("email_confirm_error", error_message="x")

    assert first == second
