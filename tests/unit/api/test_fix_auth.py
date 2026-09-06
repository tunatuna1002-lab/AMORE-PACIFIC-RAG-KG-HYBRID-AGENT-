"""
Fix D8 - authentication on mutating / destructive routes
========================================================
`verify_api_key` reads API_KEY from the environment at call time (so tests and
re-configured deployments do not need a process restart) and is wired to every
mutating route. Subscribe / verification / confirm-email flows stay public.
"""

from __future__ import annotations

import pytest

from tests.unit.api.conftest import FakeSqliteStorage

PROTECTED_ROUTES = [
    ("POST", "/api/data/refresh", {}),
    ("PUT", "/api/v4/alert-settings", {"json": {"email": "a@b.c", "alert_types": ["rank_drop"]}}),
    ("DELETE", "/api/v4/alert-settings", {"params": {"email": "a@b.c"}}),
    ("DELETE", "/api/signals/clear", {}),
    ("POST", "/api/signals/fetch/rss", {}),
    ("POST", "/api/signals/fetch/reddit", {}),
    (
        "POST",
        "/api/signals/manual",
        {"json": {"source": "reddit", "date": "2026-09-01", "title": "t"}},
    ),
    ("POST", "/api/signals/trend-radar", {"json": []}),
    ("POST", "/api/sync/upload", {"json": {"records": []}}),
    ("POST", "/api/export/docx", {"json": {}}),
    (
        "POST",
        "/api/export/analyst-report",
        {"json": {"start_date": "2026-09-01", "end_date": "2026-09-02"}},
    ),
    ("POST", "/api/export/excel", {"json": {}}),
    ("POST", "/api/export/async/start", {"json": {"job_type": "export_docx"}}),
    ("POST", "/api/v4/brain/check-alerts", {}),
    ("DELETE", "/api/chat/memory/some-session", {}),
    ("POST", "/api/alerts/send", {}),
    ("POST", "/api/alerts/test", {}),
]

PUBLIC_ROUTES = [
    ("POST", "/api/v4/subscribe", {"json": {"email": "a@b.c"}}),
    ("POST", "/api/alerts/send-verification", {"json": {"email": "a@b.c"}}),
    ("POST", "/api/alerts/verify-email", {"json": {"token": "x", "email": "a@b.c"}}),
    ("GET", "/api/alerts/confirm-email", {"params": {"token": "x", "email": "a@b.c"}}),
]


class _FakeSignal:
    def to_dict(self) -> dict:
        return {"title": "t"}


class FakeCollector:
    signals: list = []

    def _save_signals(self) -> None:
        pass

    async def fetch_all_rss_feeds(self, keywords=None):
        return []

    async def fetch_reddit_trends(self, subreddits=None, keywords=None, max_posts=10):
        return []

    def add_manual_media_input(self, data):
        return _FakeSignal()

    def add_weekly_trend_radar(self, items):
        return []


class FakeAlertService:
    _slack_enabled = False
    _email_enabled = False

    async def send_single_alert(self, alert):
        return {"slack": False, "email": False}


@pytest.fixture
def no_side_effects(monkeypatch, fresh_job_queue):
    """Keep the "authenticated" calls offline: no RSS/Reddit/SMTP/Slack/SQLite."""
    monkeypatch.setattr("src.api.routes.signals._collector", FakeCollector())
    monkeypatch.setattr("src.api.routes.alerts.get_alert_service", lambda: FakeAlertService())
    monkeypatch.setattr("src.api.routes.alerts.get_sqlite_storage", lambda: FakeSqliteStorage())


def _call(client, method, path, kwargs, headers=None):
    return client.request(method, path, headers=headers, **kwargs)


@pytest.mark.parametrize(
    "method,path,kwargs", PROTECTED_ROUTES, ids=lambda v: v if isinstance(v, str) else ""
)
def test_protected_route_rejects_missing_key_with_401(
    lenient_client, isolated_cwd, configured_api_key, reset_rate_limits, method, path, kwargs
):
    r = _call(lenient_client, method, path, kwargs)
    assert r.status_code == 401, (path, r.text)


@pytest.mark.parametrize(
    "method,path,kwargs", PROTECTED_ROUTES, ids=lambda v: v if isinstance(v, str) else ""
)
def test_protected_route_rejects_wrong_key_with_403(
    lenient_client, isolated_cwd, configured_api_key, reset_rate_limits, method, path, kwargs
):
    r = _call(lenient_client, method, path, kwargs, headers={"X-API-Key": "wrong"})
    assert r.status_code == 403, (path, r.text)


@pytest.mark.parametrize(
    "method,path,kwargs", PROTECTED_ROUTES, ids=lambda v: v if isinstance(v, str) else ""
)
def test_protected_route_accepts_correct_key(
    lenient_client,
    isolated_cwd,
    auth_headers,
    reset_rate_limits,
    no_side_effects,
    method,
    path,
    kwargs,
):
    r = _call(lenient_client, method, path, kwargs, headers=auth_headers)
    assert r.status_code not in (401, 403, 503), (path, r.status_code, r.text)


@pytest.mark.parametrize(
    "method,path,kwargs", PUBLIC_ROUTES, ids=lambda v: v if isinstance(v, str) else ""
)
def test_subscription_flow_stays_public(
    lenient_client, isolated_cwd, configured_api_key, reset_rate_limits, method, path, kwargs
):
    r = _call(lenient_client, method, path, kwargs)
    assert r.status_code not in (401, 403), (path, r.text)


def test_verify_api_key_reads_env_at_call_time(monkeypatch):
    import asyncio

    from fastapi import HTTPException

    from src.api import dependencies as deps

    monkeypatch.delenv("API_KEY", raising=False)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(deps.verify_api_key("anything"))
    assert exc.value.status_code == 503

    monkeypatch.setenv("API_KEY", "runtime-key")
    assert asyncio.run(deps.verify_api_key("runtime-key")) == "runtime-key"
    with pytest.raises(HTTPException) as exc:
        asyncio.run(deps.verify_api_key(None))
    assert exc.value.status_code == 401
    with pytest.raises(HTTPException) as exc:
        asyncio.run(deps.verify_api_key("nope"))
    assert exc.value.status_code == 403


# --- /api/sync/upload body key -------------------------------------------------


def test_sync_upload_body_key_is_checked_with_constant_time_compare(
    client, isolated_cwd, auth_headers, configured_api_key, reset_rate_limits, monkeypatch
):
    monkeypatch.setattr("src.api.routes.sync.get_sqlite_storage", lambda: FakeSqliteStorage())
    # wrong body key is rejected even though the header is valid
    r = client.post(
        "/api/sync/upload",
        json={"records": [{"asin": "A"}], "api_key": "wrong"},
        headers=auth_headers,
    )
    assert r.status_code == 401
    # matching body key passes through to the handler
    r = client.post(
        "/api/sync/upload",
        json={"records": [], "api_key": configured_api_key},
        headers=auth_headers,
    )
    assert r.status_code == 400
    assert r.json() == {"detail": "No records provided"}


def test_sync_upload_never_skips_the_check_when_api_key_unset(
    lenient_client, isolated_cwd, monkeypatch, reset_rate_limits
):
    monkeypatch.delenv("API_KEY", raising=False)
    r = lenient_client.post("/api/sync/upload", json={"records": [{"asin": "A"}], "api_key": ""})
    assert r.status_code in (401, 503)


def test_sync_upload_uses_hmac_compare_digest():
    import inspect

    from src.api.routes import sync

    assert "compare_digest" in inspect.getsource(sync.sync_upload)
