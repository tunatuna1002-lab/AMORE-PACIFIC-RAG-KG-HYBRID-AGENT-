"""
Fix D20 (minimal) - synchronous sqlite3 work runs off the event loop
====================================================================
Routes and the JobQueue used the synchronous sqlite3 connection directly inside
`async def` handlers, blocking the event loop. They now run those blocks via
`asyncio.to_thread`. The dashboard HTML is read once and cached at module level.
"""

from __future__ import annotations

import asyncio
import threading

import pytest

from tests.unit.api.conftest import FakeSqliteStorage


class _ThreadRecordingStorage(FakeSqliteStorage):
    """Also records the thread that *requested* the storage (i.e. the event-loop thread)."""

    def __init__(self, rows=None):
        super().__init__(rows)
        self.requested_from: list[int] = []


@pytest.fixture
def storage_spy(monkeypatch):
    spy = _ThreadRecordingStorage()

    def _get_storage():
        spy.requested_from.append(threading.get_ident())
        return spy

    for module in ("analytics", "sync", "deals"):
        monkeypatch.setattr(f"src.api.routes.{module}.get_sqlite_storage", _get_storage)
    return spy


def _assert_ran_off_loop(spy: _ThreadRecordingStorage):
    assert spy.executed, "route did not touch sqlite"
    loop_thread = spy.requested_from[-1]
    for _query, _params, ident in spy.executed:
        assert ident != loop_thread, "sqlite3 execute() ran on the event-loop thread"
    assert all(i != loop_thread for i in spy.get_connection_thread_idents)


@pytest.mark.parametrize(
    "path,params",
    [
        ("/api/category/kpi", {"category_id": "lip_care"}),
        ("/api/sos/category", {}),
        ("/api/sos/brands", {}),
        ("/api/sos/trend", {}),
        ("/api/sos/trend/competitors-avg", {}),
    ],
)
def test_analytics_sqlite_runs_off_event_loop(
    client, isolated_cwd, reset_rate_limits, storage_spy, path, params
):
    r = client.get(path, params=params)
    assert r.status_code == 200, r.text
    _assert_ran_off_loop(storage_spy)


@pytest.mark.parametrize(
    "path",
    ["/api/sync/status", "/api/sync/dates", "/api/sync/download/2026-09-01"],
)
def test_sync_sqlite_runs_off_event_loop(
    client, isolated_cwd, reset_rate_limits, storage_spy, path
):
    r = client.get(path)
    assert r.status_code in (200, 404), r.text
    _assert_ran_off_loop(storage_spy)


def test_sync_upload_sqlite_runs_off_event_loop(
    client, isolated_cwd, reset_rate_limits, storage_spy, auth_headers
):
    r = client.post(
        "/api/sync/upload",
        json={"records": [{"snapshot_date": "2026-09-01", "category_id": "lip_care", "asin": "A"}]},
        headers=auth_headers,
    )
    assert r.status_code == 200, r.text
    assert r.json()["inserted"] == 1
    _assert_ran_off_loop(storage_spy)


@pytest.mark.parametrize(
    "method,path,params",
    [
        ("GET", "/api/deals/alerts", {}),
        ("POST", "/api/deals/export", {"format": "json"}),
    ],
)
def test_deals_sqlite_runs_off_event_loop(
    client, isolated_cwd, reset_rate_limits, storage_spy, method, path, params
):
    r = client.request(method, path, params=params)
    assert r.status_code == 200, r.text
    _assert_ran_off_loop(storage_spy)


# --- JobQueue ---------------------------------------------------------------


def test_job_queue_sqlite_runs_off_event_loop(tmp_path):
    from src.tools.utilities.job_queue import JobQueue

    queue = JobQueue(db_path=str(tmp_path / "jobs.db"))
    original = queue._get_connection
    idents: list[int] = []

    def spy():
        idents.append(threading.get_ident())
        return original()

    queue._get_connection = spy

    async def scenario():
        loop_thread = threading.get_ident()
        await queue.initialize()
        job_id = await queue.create_job("export_docx", {"a": 1})
        await queue.update_progress(job_id, 10, "x")
        await queue._mark_running(job_id)
        status = await queue.get_job_status(job_id)
        assert status["status"] == "running"
        await queue._mark_completed(job_id, "f.txt")
        await queue._mark_failed(job_id, "boom")
        assert await queue.get_pending_jobs() == []
        assert len(await queue.get_all_jobs()) == 1
        await queue._cleanup_expired_jobs()
        return loop_thread

    loop_thread = asyncio.run(scenario())
    assert len(idents) >= 10
    assert all(i != loop_thread for i in idents), "JobQueue ran sqlite3 on the event-loop thread"


# --- dashboard HTML cache ---------------------------------------------------


def test_dashboard_html_is_read_once_and_cached(
    client, isolated_cwd, reset_rate_limits, monkeypatch
):
    from src.api.routes import health

    page = isolated_cwd / "dashboard" / "amore_unified_dashboard_v4.html"
    page.parent.mkdir(parents=True)
    page.write_text("<html><head></head><body>v1</body></html>", encoding="utf-8")

    reads: list[str] = []
    real_read = health._read_html_file

    def counting_read(path):
        reads.append(str(path))
        return real_read(path)

    monkeypatch.setattr(health, "_read_html_file", counting_read)
    health._DASHBOARD_HTML_CACHE.clear()

    r1 = client.get("/dashboard")
    assert r1.status_code == 200
    assert "v1" in r1.text
    r2 = client.get("/dashboard")
    assert r2.status_code == 200
    assert r2.text == r1.text
    assert len(reads) == 1

    health._DASHBOARD_HTML_CACHE.clear()


def test_dashboard_injects_read_token_from_cached_html(
    client, isolated_cwd, reset_rate_limits, monkeypatch
):
    from src.api.routes import health

    page = isolated_cwd / "dashboard" / "amore_unified_dashboard_v4.html"
    page.parent.mkdir(parents=True)
    page.write_text("<html><head></head><body>v1</body></html>", encoding="utf-8")
    health._DASHBOARD_HTML_CACHE.clear()
    monkeypatch.setattr(health, "DASHBOARD_READ_TOKEN", "read-only-token")

    r = client.get("/dashboard")
    assert 'window.DASHBOARD_API_KEY = "read-only-token"' in r.text
    health._DASHBOARD_HTML_CACHE.clear()
