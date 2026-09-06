"""
Fix D19 (API part) - analytics default dates use KST "today"
=============================================================
The crawler stamps snapshot_date in KST (22:00 KST daily run). The analytics routes
defaulted end_date to the *server-local* (UTC on Railway) date, so between 00:00 and
09:00 KST the latest snapshot was silently excluded.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tests.unit.api.conftest import FakeSqliteStorage

FROZEN_UTC = datetime(2026, 9, 2, 23, 30, tzinfo=UTC)  # == 2026-09-03 08:30 KST


class FrozenDatetime(datetime):
    """datetime whose now() is pinned at FROZEN_UTC (tz-aware when tz is given)."""

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return FROZEN_UTC.replace(tzinfo=None)
        return FROZEN_UTC.astimezone(tz)


@pytest.fixture
def frozen_clock(monkeypatch):
    from src.api.routes import analytics

    monkeypatch.setattr(analytics, "datetime", FrozenDatetime)
    monkeypatch.setattr(analytics, "get_sqlite_storage", lambda: FakeSqliteStorage())
    monkeypatch.setattr(analytics, "_load_crawl_data_for_sos", lambda: None)


@pytest.mark.parametrize(
    "path,params,expected_start",
    [
        ("/api/category/kpi", {"category_id": "lip_care"}, "2026-08-27"),
        ("/api/sos/category", {}, "2026-09-03"),
        ("/api/sos/brands", {}, "2026-08-27"),
        ("/api/sos/trend", {"days": 7}, "2026-08-27"),
        ("/api/sos/trend/competitors-avg", {"days": 7}, "2026-08-27"),
    ],
)
def test_default_end_date_is_kst_today(
    client, isolated_cwd, reset_rate_limits, frozen_clock, path, params, expected_start
):
    r = client.get(path, params=params)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True, body
    assert body["period"]["end"] == "2026-09-03"
    assert body["period"]["start"] == expected_start


def test_explicit_dates_are_untouched(client, isolated_cwd, reset_rate_limits, frozen_clock):
    r = client.get("/api/sos/trend", params={"start_date": "2026-01-01", "end_date": "2026-01-05"})
    assert r.json()["period"] == {"start": "2026-01-01", "end": "2026-01-05", "days": 7}
