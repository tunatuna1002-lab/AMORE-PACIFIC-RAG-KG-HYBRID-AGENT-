"""
F6 — API data access & date defaults converge on the shared services
====================================================================

Two single sources of truth are pinned here:

1. ``src/application/services/dashboard_data_service.py`` owns data-path
   resolution (``resolve_data_dir``: env ``DATA_DIR`` > ``/data`` on Railway /
   when the volume exists > ``./data``).  No route module may carry its own
   data-directory literal — on Railway a route that hard-codes ``./data`` reads
   a different directory than the rest of the process (the JSON fallback then
   silently never fires).
2. ``src/application/services/date_range.py`` owns "default the missing
   start/end date".  Defaults are KST-based because the crawler stamps
   ``snapshot_date`` in KST (defect family D19).
"""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.unit.api.conftest import FakeSqliteStorage

ROUTES_DIR = Path(__file__).resolve().parents[3] / "src" / "api" / "routes"

# Route modules still allowed to carry a data-directory literal.
# Each entry must be (module_name, literal) with a TODO naming the owner.
# Empty as of F6: every routes/*.py goes through DashboardDataService.
ALLOWLIST: set[tuple[str, str]] = set()


def _is_data_dir_literal(value: str) -> bool:
    """
    True for filesystem data paths, False for URL paths.

    ``"/api/data"`` (a route decorator) must not match; ``"/data"``,
    ``"./data/x.json"`` and ``"data/amore_data.db"`` must.
    """
    return (
        value in ("./data", "/data")
        or value.startswith("./data/")
        or value.startswith("/data/")
        or value.startswith("data/")
    )


def _string_constants(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


# --------------------------------------------------------------------------- 1
def test_route_modules_have_no_data_directory_literals():
    """Every routes/*.py resolves data paths through DashboardDataService."""
    offenders = []
    for module in sorted(ROUTES_DIR.glob("*.py")):
        for lineno, value in _string_constants(module):
            if not _is_data_dir_literal(value):
                continue
            if (module.name, value) in ALLOWLIST:
                continue
            offenders.append(f"{module.relative_to(ROUTES_DIR.parents[2])}:{lineno}: {value!r}")

    assert not offenders, (
        "Hard-coded data-directory literals found in route modules — use "
        "get_data_service().path_for(...) instead:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize(
    "value,expected",
    [
        ("/api/data", False),
        ("/api/data/refresh", False),
        ("/data", True),
        ("./data", True),
        ("./data/exports/x.xlsx", True),
        ("/data/amore_data.db", True),
        ("data/amore_data.db", True),
        ("database", False),
        ("", False),
    ],
)
def test_data_dir_literal_matcher_ignores_url_paths(value, expected):
    assert _is_data_dir_literal(value) is expected


# --------------------------------------------------------------------------- 2
class TestLoadDashboardDataIsAServiceWrapper:
    """`src.api.dependencies.load_dashboard_data` == DashboardDataService.load_dashboard_json."""

    def test_reads_from_DATA_DIR_and_annotates_staleness(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        payload = {"metadata": {"data_date": "2026-09-01"}, "brand": {"kpis": {"sos": 12.5}}}
        (tmp_path / "dashboard_data.json").write_text(json.dumps(payload), encoding="utf-8")

        from src.api.dependencies import load_dashboard_data

        data = load_dashboard_data()

        assert data["brand"]["kpis"]["sos"] == 12.5
        assert data["metadata"]["data_date"] == "2026-09-01"
        assert data["metadata"]["_cache_age_hours"] == pytest.approx(0, abs=0.5)
        assert data["metadata"]["_is_stale"] is False

    def test_returns_empty_dict_when_file_absent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "nonexistent"))

        from src.api.dependencies import load_dashboard_data

        assert load_dashboard_data() == {}

    def test_dashboard_data_path_follows_DATA_DIR_at_call_time(self, tmp_path, monkeypatch):
        from src.api.dependencies import dashboard_data_path

        monkeypatch.setenv("DATA_DIR", str(tmp_path / "first"))
        assert dashboard_data_path() == tmp_path / "first" / "dashboard_data.json"
        # No module-level caching: a later env change must be picked up.
        monkeypatch.setenv("DATA_DIR", str(tmp_path / "second"))
        assert dashboard_data_path() == tmp_path / "second" / "dashboard_data.json"


# --------------------------------------------------------------------------- 3
FROZEN_UTC = datetime(2026, 9, 2, 23, 30, tzinfo=UTC)  # == 2026-09-03 08:30 KST
KST_TODAY = "2026-09-03"
SERVER_LOCAL_TODAY = "2026-09-02"  # what naive datetime.now() would have produced


class FrozenDatetime(datetime):
    """datetime pinned at FROZEN_UTC (tz-aware only when a tz is passed)."""

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return FROZEN_UTC.replace(tzinfo=None)
        return FROZEN_UTC.astimezone(tz)


@pytest.fixture
def frozen_kst_clock(monkeypatch):
    """Freeze the clock inside the single date-range service the routes now call."""
    from src.application.services import date_range

    monkeypatch.setattr(date_range, "datetime", FrozenDatetime)


class TestAnalyticsDefaultWindow:
    """Default windows are unchanged per endpoint, and end_date is KST today."""

    @pytest.fixture(autouse=True)
    def _stub_storage(self, monkeypatch):
        from src.api.routes import analytics

        monkeypatch.setattr(analytics, "get_sqlite_storage", lambda: FakeSqliteStorage())
        monkeypatch.setattr(analytics, "_load_crawl_data_for_sos", lambda: None)

    @pytest.mark.parametrize(
        "path,params,expected_start",
        [
            # default_days=7
            ("/api/category/kpi", {"category_id": "lip_care"}, "2026-08-27"),
            # default_days=0 (start defaults to end)
            ("/api/sos/category", {}, KST_TODAY),
            # default_days=7
            ("/api/sos/brands", {}, "2026-08-27"),
            # default_days=`days` query param
            ("/api/sos/trend", {"days": 7}, "2026-08-27"),
            ("/api/sos/trend", {"days": 30}, "2026-08-04"),
            ("/api/sos/trend/competitors-avg", {"days": 7}, "2026-08-27"),
            ("/api/sos/trend/competitors-avg", {"days": 14}, "2026-08-20"),
        ],
    )
    def test_default_window(
        self,
        client,
        isolated_cwd,
        reset_rate_limits,
        frozen_kst_clock,
        path,
        params,
        expected_start,
    ):
        r = client.get(path, params=params)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["success"] is True, body
        assert body["period"]["end"] == KST_TODAY
        assert body["period"]["start"] == expected_start

    def test_start_is_relative_to_an_explicit_end_date(
        self, client, isolated_cwd, reset_rate_limits, frozen_kst_clock
    ):
        """
        BEHAVIOUR CHANGE (F6): with only `end_date` given, the missing `start_date`
        is now `end_date - N`, not `KST today - N`. The old route code anchored the
        implicit start on "today" regardless of the requested end, so asking for a
        historical end_date returned a window that did not contain it.
        """
        r = client.get(
            "/api/category/kpi", params={"category_id": "lip_care", "end_date": "2026-05-10"}
        )
        assert r.status_code == 200, r.text
        assert r.json()["period"] == {"start": "2026-05-03", "end": "2026-05-10"}

    def test_explicit_dates_are_untouched(
        self, client, isolated_cwd, reset_rate_limits, frozen_kst_clock
    ):
        r = client.get(
            "/api/sos/trend", params={"start_date": "2026-01-01", "end_date": "2026-01-05"}
        )
        assert r.json()["period"] == {"start": "2026-01-01", "end": "2026-01-05", "days": 7}


class TestDealsCutoffIsKst:
    """POST /api/deals/export?format=json cutoff is KST-based (was naive/UTC)."""

    def test_cutoff_uses_kst_today(
        self, client, isolated_cwd, reset_rate_limits, frozen_kst_clock, monkeypatch
    ):
        from src.api.routes import deals

        captured: dict[str, str] = {}

        def _fake_fetch(storage, cutoff_date):
            captured["cutoff"] = cutoff_date
            return []

        monkeypatch.setattr(deals, "get_sqlite_storage", lambda: FakeSqliteStorage())
        monkeypatch.setattr(deals, "_fetch_deals_since", _fake_fetch)

        r = client.post("/api/deals/export", params={"days": 7, "format": "json"})
        assert r.status_code == 200, r.text
        assert r.json()["period_days"] == 7
        # KST today (2026-09-03) - 7 days, NOT server-local 2026-09-02 - 7 days.
        assert captured["cutoff"] == "2026-08-27"
        assert captured["cutoff"] != "2026-08-26"


class TestSignalsCutoffIsKst:
    """GET /api/signals/ cutoff is KST-based (was naive/UTC)."""

    def test_signal_on_the_kst_boundary_is_included(
        self, client, isolated_cwd, reset_rate_limits, frozen_kst_clock, monkeypatch
    ):
        from src.api.routes import signals as signals_route

        class _Signal:
            def __init__(self, published_at: str):
                self.published_at = published_at
                self.tier = "tier1_viral"
                self.source = "reddit"

            def to_dict(self) -> dict:
                return {"published_at": self.published_at}

        class _Collector:
            signals = [
                _Signal("2026-08-27"),  # exactly KST today - 7 → included
                _Signal("2026-08-26"),  # one day earlier → excluded
            ]

        monkeypatch.setattr(signals_route, "_collector", _Collector())

        r = client.get("/api/signals/", params={"days": 7})
        assert r.status_code == 200, r.text
        body = r.json()
        assert [s["published_at"] for s in body["signals"]] == ["2026-08-27"]


class TestHistoricalFallbackDateIsKst:
    """`_get_historical_from_local` falls back to KST today for a missing data_date."""

    @pytest.mark.asyncio
    async def test_missing_data_date_falls_back_to_kst_today(
        self, isolated_cwd, frozen_kst_clock, monkeypatch
    ):
        from src.api.routes import data as data_route

        monkeypatch.setattr(
            data_route,
            "load_dashboard_data",
            lambda: {"metadata": {}, "brand": {"kpis": {"sos": 10, "avg_rank": 5}}},
        )

        result = await data_route._get_historical_from_local(
            start_date="2026-08-01", end_date="2026-09-30"
        )
        assert result["success"] is True, result
        dates = [h["date"] for h in result["data"]["sos_history"]]
        assert KST_TODAY in dates
        assert SERVER_LOCAL_TODAY not in dates
        assert result["available_dates"] == [KST_TODAY]
