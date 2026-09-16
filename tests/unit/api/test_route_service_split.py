"""
F6 / Phase 4 — "route = validate + call a service"
==================================================
Pins the HTTP contract of the four route modules that Phase 4 splits
(``export.py``, ``alerts.py``, ``data.py``, ``analytics.py``) BEFORE the pure data
transformation moves into ``src/application/services/``.

Two kinds of test live here:

1. **Contract pins** (sections 1-5). They describe what the endpoints answer today
   and must stay green through the move; a failure during a move-only step means
   the move was wrong.
2. **Structure contracts** (section 6). They encode the F6 targets — services that
   import without FastAPI, no ``tools -> api`` inversion, a single analysis-report
   document path. They are RED before the split and GREEN after it.

Everything goes through public entry points: the TestClient against the real app,
a real SQLite file / real JSON cache in an isolated CWD, or the service classes
themselves. No route internals are monkeypatched.
"""

from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[3] / "src"
ROUTES = SRC / "api" / "routes"
SERVICES = SRC / "application" / "services"


# --------------------------------------------------------------------------- utils
def _import_targets(path: Path) -> set[str]:
    """Every module a file imports — top level *and* inside function bodies."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    targets: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            targets.add(node.module)
    return targets


def _referenced_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} | {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }


# --------------------------------------------------------------------------- fixtures
RAW_ROWS = [
    # (snapshot_date, category_id, rank, asin, product_name, brand, price, rating)
    ("2026-09-01", "lip_care", 1, "B07GFJWPDQ", "LANEIGE Lip Sleeping Mask", "LANEIGE", 24.0, 4.6),
    ("2026-09-01", "lip_care", 2, "B00KBZCJEG", "Burt's Bees Lip Balm", "Burt's Bees", 9.0, 4.5),
    ("2026-09-02", "lip_care", 1, "B07GFJWPDQ", "LANEIGE Lip Sleeping Mask", "LANEIGE", 25.0, 4.6),
    ("2026-09-02", "lip_care", 3, "B00KBZCJEG", "Burt's Bees Lip Balm", "Burt's Bees", 9.5, 4.5),
]


@pytest.fixture
def sqlite_db(isolated_cwd):
    """A real ./data/amore_data.db (the path get_sqlite_storage() resolves) with raw rows."""
    from src.tools.storage.sqlite_storage import SQLiteStorage

    db_dir = isolated_cwd / "data"
    db_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_dir / "amore_data.db"))
    try:
        conn.executescript(SQLiteStorage.SCHEMA)
        conn.executemany(
            "INSERT INTO raw_data "
            "(snapshot_date, category_id, rank, asin, product_name, brand, price, rating) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            RAW_ROWS,
        )
        conn.commit()
    finally:
        conn.close()
    return db_dir / "amore_data.db"


CRAWL_RESULT = {
    "snapshot_date": "2026-09-01",
    "categories": {
        "lip_care": {
            "products": [
                {"rank": 1, "brand": "LANEIGE", "price": 24.0, "product_name": "Lip Mask"},
                {"rank": 2, "brand": "Burt's Bees", "price": 9.0, "product_name": "Lip Balm"},
            ]
        },
        "lip_makeup": {
            "products": [
                {"rank": 1, "brand": "Maybelline", "price": 8.0, "product_name": "Lipstick"},
            ]
        },
    },
}


@pytest.fixture
def crawl_file(isolated_cwd):
    """./data/latest_crawl_result.json — the analytics JSON fallback source."""
    data_dir = isolated_cwd / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "latest_crawl_result.json"
    path.write_text(json.dumps(CRAWL_RESULT, ensure_ascii=False), encoding="utf-8")
    return path


# ===========================================================================
# 1. data.py — GET /api/data, GET /api/historical
# ===========================================================================


def test_data_returns_the_cached_json_untouched(client, dashboard_file, reset_rate_limits):
    body = client.get("/api/data").json()
    assert body["brand"]["kpis"]["sos"] == 12.5
    assert body["metadata"]["_is_stale"] is False
    assert body["metadata"]["_source"] == "json"


def test_data_falls_back_to_sqlite_when_the_cache_is_missing(client, sqlite_db, reset_rate_limits):
    body = client.get("/api/data").json()
    meta = body["metadata"]
    assert meta["_source"] == "sqlite_fallback"
    # get_latest_data() returns the most recent snapshot only
    assert meta["data_date"] == "2026-09-02"
    assert meta["total_products"] == 2
    assert meta["laneige_products"] == 1
    assert body["brand"]["kpis"]["top10_count"] == 1
    assert body["brand"]["kpis"]["hhi"] == "N/A"
    assert [c["brand"] for c in body["brand"]["competitors"]] == ["LANEIGE", "Burt's Bees"]
    assert list(body["products"]) == ["B07GFJWPDQ"]
    assert body["home"]["status"]["position"] == "Top 1"
    assert body["home"]["action_items"][0]["brand_variant"] == "LANEIGE"


def test_historical_from_sqlite(client, sqlite_db, reset_rate_limits):
    r = client.get("/api/historical", params={"start_date": "2026-09-01", "end_date": "2026-09-02"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["data_source"] == "sqlite"
    assert body["available_dates"] == ["2026-09-01", "2026-09-02"]
    # brand filter (default LANEIGE) applies to the daily aggregation ...
    assert body["data"]["sos_history"] == [
        {"date": "2026-09-01", "sos": 1.0, "product_count": 1, "top10_count": 1},
        {"date": "2026-09-02", "sos": 1.0, "product_count": 1, "top10_count": 1},
    ]
    assert body["data"]["raw_data"] == [
        {"date": "2026-09-01", "rank": 1.0, "best_rank": 1, "worst_rank": 1},
        {"date": "2026-09-02", "rank": 1.0, "best_rank": 1, "worst_rank": 1},
    ]
    assert body["data"]["period"] == {"start": "2026-09-01", "end": "2026-09-02", "days": 2}
    # ... while brand_metrics covers every brand in the period (ASIN-unique counts)
    metrics = {m["brand"]: m for m in body["brand_metrics"]}
    assert metrics["LANEIGE"]["product_count"] == 1
    assert metrics["LANEIGE"]["is_laneige"] is True
    assert metrics["LANEIGE"]["avg_rank"] == 1.0
    assert metrics["LANEIGE"]["avg_price"] == 24.5
    assert metrics["Burt's Bees"]["is_laneige"] is False
    # Summer Fridays is always appended as a tracked competitor, even with no rows
    assert metrics["Summer Fridays"]["no_data"] is True
    assert metrics["Summer Fridays"]["is_tracked"] is True
    assert sorted(body["rank_history"]) == ["2026-09-01", "2026-09-02"]
    assert body["rank_history"]["2026-09-01"]["products"][0]["rank"] == 1


def test_historical_local_fallback_uses_the_dashboard_cache(
    client, dashboard_file, reset_rate_limits
):
    """No SQLite rows -> the local JSON fallback answers with source=local."""
    r = client.get("/api/historical", params={"start_date": "2026-08-01", "end_date": "2026-09-30"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["success"] is True
    assert body["data"]["source"] == "local"
    assert body["available_dates"] == ["2026-09-01"]
    assert body["data"]["sos_history"] == [
        {"date": "2026-09-01", "sos": 12.5, "product_count": 0, "top10_count": 1}
    ]
    assert body["data"]["raw_data"] == [
        {"date": "2026-09-01", "rank": 1.0, "best_rank": 1.0, "worst_rank": 1.0}
    ]
    # brand_metrics come from brand.competitors when charts.brand_matrix is absent
    assert body["brand_metrics"] == [
        {
            "brand": "LANEIGE",
            "sos": 50.0,
            "avg_rank": 1.0,
            "product_count": 1,
            "bubble_size": 5,
            "is_laneige": True,
        },
        {
            "brand": "Burt's Bees",
            "sos": 50.0,
            "avg_rank": 2.0,
            "product_count": 1,
            "bubble_size": 5,
            "is_laneige": False,
        },
    ]


def test_historical_without_any_data_reports_failure(client, isolated_cwd, reset_rate_limits):
    body = client.get(
        "/api/historical", params={"start_date": "2026-08-01", "end_date": "2026-08-02"}
    ).json()
    assert body == {
        "success": False,
        "error": "No historical data found for the specified period",
        "available_dates": [],
        "brand_metrics": [],
        "rank_history": {},
        "data": None,
    }


# ===========================================================================
# 2. analytics.py — KPI / SoS endpoints
# ===========================================================================


def test_category_kpi_from_sqlite(client, sqlite_db, reset_rate_limits):
    body = client.get(
        "/api/category/kpi",
        params={
            "category_id": "lip_care",
            "start_date": "2026-09-01",
            "end_date": "2026-09-02",
        },
    ).json()
    assert body["success"] is True
    assert body["period"] == {"start": "2026-09-01", "end": "2026-09-02"}
    assert body["data"] == {
        "category_id": "lip_care",
        "sos": 50.0,
        "best_rank": 1,
        "cpi": 145.0,
        "new_competitors": 0,
        "brand": "LANEIGE",
        "product_count": 2,
        "total_products": 4,
    }


def test_category_kpi_without_rows_reports_the_empty_window(
    client, isolated_cwd, reset_rate_limits
):
    body = client.get(
        "/api/category/kpi",
        params={"category_id": "lip_care", "start_date": "2026-01-01", "end_date": "2026-01-02"},
    ).json()
    assert body == {
        "success": True,
        "message": "해당 기간(2026-01-01 ~ 2026-01-02)에 데이터가 없습니다.",
        "data": None,
        "period": {"start": "2026-01-01", "end": "2026-01-02"},
    }


def test_category_kpi_json_fallback(client, crawl_file, reset_rate_limits):
    """No SQLite rows -> latest_crawl_result.json feeds the same KPI block."""
    body = client.get(
        "/api/category/kpi",
        params={"category_id": "lip_care", "start_date": "2026-09-01", "end_date": "2026-09-01"},
    ).json()
    assert body["success"] is True
    assert body["data"]["total_products"] == 2
    assert body["data"]["product_count"] == 1
    assert body["data"]["sos"] == 50.0
    assert body["data"]["best_rank"] == 1


# The category metadata (name/level/parent_id/indent/order) the SoS rows carry.
# config/category_hierarchy.json is the single source; this is what it yields.
EXPECTED_CATEGORY_META = {
    "beauty": ("Beauty & Personal Care", 0, None, 0, 0),
    "skin_care": ("Skin Care", 1, "beauty", 1, 1),
    "lip_care": ("Lip Care", 2, "skin_care", 2, 2),
    "lip_makeup": ("Lip Makeup", 2, "makeup", 1, 3),
    "face_powder": ("Face Powder", 3, "face_makeup", 2, 4),
}


def test_sos_by_category_rows_and_hierarchy_metadata(client, sqlite_db, reset_rate_limits):
    body = client.get(
        "/api/sos/category",
        params={
            "start_date": "2026-09-01",
            "end_date": "2026-09-02",
            "compare_brands": "Burt's Bees",
        },
    ).json()
    assert body["success"] is True
    assert body["period"] == {"start": "2026-09-01", "end": "2026-09-02", "days": 2}
    assert body["compare_brands"] == ["Burt's Bees"]
    assert set(body["hierarchy_info"]) == {"description", "note"}

    (row,) = body["data"]
    name, level, parent_id, indent, order = EXPECTED_CATEGORY_META["lip_care"]
    assert row["category_id"] == "lip_care"
    assert (row["category_name"], row["level"], row["parent_id"]) == (name, level, parent_id)
    assert (row["indent"], row["order"]) == (indent, order)
    assert row["total_products"] == 2
    assert row["laneige_sos"] == 50.0
    assert row["laneige_count"] == 1.0
    assert row["laneige_appearance_days"] == 2
    assert row["laneige_appearance_rate"] == 100.0
    assert row["avg_sos"] == 50.0
    assert row["compare_brands"] == {"Burt's Bees": 50.0}
    assert row["num_dates"] == 2


def test_sos_by_category_json_fallback_orders_rows_by_hierarchy(
    client, crawl_file, reset_rate_limits
):
    body = client.get(
        "/api/sos/category", params={"start_date": "2026-09-01", "end_date": "2026-09-01"}
    ).json()
    assert body["success"] is True
    assert [r["category_id"] for r in body["data"]] == ["lip_care", "lip_makeup"]
    assert [r["order"] for r in body["data"]] == [2, 3]
    assert [r["category_name"] for r in body["data"]] == ["Lip Care", "Lip Makeup"]


def test_sos_brands_lists_brands_with_laneige_flag(client, sqlite_db, reset_rate_limits):
    body = client.get("/api/sos/brands", params={"category_id": "lip_care"}).json()
    assert body["success"] is True
    assert body["category_id"] == "lip_care"
    assert body["total_brands"] == len(body["brands"])
    assert set(body) == {"success", "period", "category_id", "brands", "total_brands"}


def test_sos_brands_json_fallback(client, crawl_file, reset_rate_limits):
    body = client.get("/api/sos/brands").json()
    assert body["success"] is True
    assert body["brands"] == [
        {"name": "LANEIGE", "product_count": 1, "days_present": 1, "is_laneige": True},
        {"name": "Burt's Bees", "product_count": 1, "days_present": 1, "is_laneige": False},
        {"name": "Maybelline", "product_count": 1, "days_present": 1, "is_laneige": False},
    ]


def test_sos_trend_daily_series(client, sqlite_db, reset_rate_limits):
    body = client.get(
        "/api/sos/trend",
        params={"start_date": "2026-09-01", "end_date": "2026-09-02", "category_id": "lip_care"},
    ).json()
    assert body["success"] is True
    assert body["brand"] == "LANEIGE"
    assert body["category_id"] == "lip_care"
    assert body["period"] == {"start": "2026-09-01", "end": "2026-09-02", "days": 7}
    assert body["trend"] == [
        {"date": "2026-09-01", "total_products": 2, "brand_count": 1, "sos": 50.0},
        {"date": "2026-09-02", "total_products": 2, "brand_count": 1, "sos": 50.0},
    ]


def test_competitors_avg_sos_trend(client, sqlite_db, reset_rate_limits):
    body = client.get(
        "/api/sos/trend/competitors-avg",
        params={"start_date": "2026-09-01", "end_date": "2026-09-02", "top_n": 5},
    ).json()
    assert body["success"] is True
    assert body["excluded_brand"] == "LANEIGE"
    assert body["top_n"] == 5
    assert body["trend"] == [
        {"date": "2026-09-01", "total_products": 2, "top_brands_count": 1, "avg_sos": 50.0},
        {"date": "2026-09-02", "total_products": 2, "top_brands_count": 1, "avg_sos": 50.0},
    ]


# ===========================================================================
# 3. alerts.py
# ===========================================================================


def test_alert_service_status(client, isolated_cwd):
    body = client.get("/api/alerts/status").json()
    assert body["success"] is True
    assert body["slack_enabled"] is False
    assert body["email_enabled"] is False
    assert "competitor_brands" in body


def test_alert_settings_v3_without_subscribers(client, isolated_cwd):
    assert client.get("/api/v3/alert-settings").json() == {
        "email": "",
        "consent": False,
        "alert_types": [],
        "consent_date": None,
    }


def test_alert_settings_v4_without_subscribers(client, isolated_cwd):
    assert client.get("/api/v4/alert-settings").json() == {
        "found": False,
        "email": "",
        "consent": False,
        "alert_types": [],
    }


def test_alert_settings_v4_unknown_email(client, isolated_cwd):
    assert client.get("/api/v4/alert-settings", params={"email": "nobody@x.io"}).json() == {
        "found": False,
        "email": "nobody@x.io",
        "message": "등록되지 않은 이메일입니다.",
    }


def test_alerts_list_v3(client, isolated_cwd):
    body = client.get("/api/v3/alerts").json()
    assert set(body) == {"alerts", "pending_count", "stats"}
    assert body["alerts"] == []


def test_verification_status_for_unknown_email(client, isolated_cwd):
    assert client.get("/api/alerts/verification-status", params={"email": "a@b.c"}).json() == {
        "verified": False,
        "status": "not_found",
    }


def test_subscribe_v4_rejects_a_malformed_email(client, isolated_cwd, reset_rate_limits):
    r = client.post("/api/v4/subscribe", json={"email": "not-an-email", "alert_types": ["x"]})
    assert r.status_code == 400
    assert r.json() == {"detail": "올바른 이메일 주소를 입력해주세요."}


def test_subscribe_v4_requires_at_least_one_alert_type(client, isolated_cwd, reset_rate_limits):
    r = client.post("/api/v4/subscribe", json={"email": "a@b.c", "alert_types": []})
    assert r.status_code == 400
    assert r.json() == {"detail": "최소 하나 이상의 알림 유형을 선택해주세요."}


def test_send_insight_report_requires_a_verified_email(client, dashboard_file):
    r = client.post("/api/alerts/send-insight-report", json={"email": "a@b.c"})
    assert r.status_code == 403
    assert r.json() == {"detail": "이메일 인증이 필요합니다. 먼저 이메일을 인증해주세요."}


def test_send_insight_report_requires_an_email(client, dashboard_file):
    r = client.post("/api/alerts/send-insight-report", json={})
    assert r.status_code == 400
    assert r.json() == {"detail": "이메일 주소가 필요합니다."}


def test_confirm_email_page_rejects_an_invalid_token(client, isolated_cwd):
    r = client.get("/api/alerts/confirm-email", params={"token": "nope", "email": "a@b.c"})
    assert r.status_code == 400
    assert r.headers["content-type"].startswith("text/html")
    assert "인증 실패" in r.text


# ===========================================================================
# 4. export.py
# ===========================================================================


def test_export_signal_source_status(client, isolated_cwd, reset_rate_limits):
    body = client.get("/api/export/signals/status").json()
    assert set(body) == {
        "tavily",
        "gnews",
        "rss_feeds",
        "reddit",
        "public_data",
        "signal_classification",
    }
    assert body["tavily"]["configured"] is False
    assert body["rss_feeds"]["count"] == 10
    assert set(body["signal_classification"]) == {
        "tier1_core",
        "tier2_background",
        "tier3_archive",
    }


def test_export_docx_streams_a_document(client, dashboard_file, auth_headers, reset_rate_limits):
    r = client.post(
        "/api/export/docx", json={"include_external_signals": False}, headers=auth_headers
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-disposition"].startswith("attachment; filename=AMORE_Insight_Report_")
    assert r.content[:2] == b"PK"


def test_export_async_job_lifecycle(
    client, isolated_cwd, auth_headers, reset_rate_limits, fresh_job_queue
):
    started = client.post(
        "/api/export/async/start",
        json={"job_type": "export_analyst_report", "start_date": "2026-09-01"},
        headers=auth_headers,
    ).json()
    assert started["status"] == "pending"
    job_id = started["job_id"]

    status = client.get(f"/api/export/async/status/{job_id}").json()
    assert status["id"] == job_id
    assert status["status"] in {"pending", "running"}

    jobs = client.get("/api/export/async/jobs").json()
    assert jobs["total"] == len(jobs["jobs"]) >= 1

    # not finished -> download refuses
    r = client.get(f"/api/export/download/{job_id}")
    assert r.status_code == 400
    assert "Job not completed yet" in r.json()["detail"]


def test_export_async_status_404_for_unknown_job(
    client, isolated_cwd, reset_rate_limits, fresh_job_queue
):
    r = client.get("/api/export/async/status/does-not-exist")
    assert r.status_code == 404
    assert r.json() == {"detail": "Job not found: does-not-exist"}


def test_export_analyst_report_without_data_is_404(
    lenient_client, isolated_cwd, auth_headers, reset_rate_limits
):
    r = lenient_client.post(
        "/api/export/analyst-report",
        json={"start_date": "2026-09-01", "end_date": "2026-09-02"},
        headers=auth_headers,
    )
    assert r.status_code == 404
    assert r.json()["detail"].startswith("No data found for period")


# ===========================================================================
# 5. Auth matrix — every mutating route of the four modules
# ===========================================================================

MUTATING_ROUTES = [
    # data.py
    ("POST", "/api/data/refresh", {}),
    # alerts.py
    ("POST", "/api/alerts/send", {}),
    ("POST", "/api/alerts/test", {}),
    ("POST", "/api/v3/alert-settings", {"json": {"email": "a@b.c", "alert_types": []}}),
    ("POST", "/api/v3/alert-settings/revoke", {}),
    ("PUT", "/api/v4/alert-settings", {"json": {"email": "a@b.c", "alert_types": ["x"]}}),
    ("DELETE", "/api/v4/alert-settings", {"params": {"email": "a@b.c"}}),
    # export.py
    ("POST", "/api/export/docx", {"json": {}}),
    (
        "POST",
        "/api/export/analyst-report",
        {"json": {"start_date": "2026-09-01", "end_date": "2026-09-02"}},
    ),
    ("POST", "/api/export/excel", {"json": {}}),
    ("POST", "/api/export/async/start", {"json": {"job_type": "export_excel"}}),
]


@pytest.mark.parametrize(
    "method,path,kwargs", MUTATING_ROUTES, ids=lambda v: v if isinstance(v, str) else ""
)
def test_mutating_route_without_a_key_is_401(
    lenient_client, isolated_cwd, configured_api_key, reset_rate_limits, method, path, kwargs
):
    assert lenient_client.request(method, path, **kwargs).status_code == 401, path


@pytest.mark.parametrize(
    "method,path,kwargs", MUTATING_ROUTES, ids=lambda v: v if isinstance(v, str) else ""
)
def test_mutating_route_with_a_wrong_key_is_403(
    lenient_client, isolated_cwd, configured_api_key, reset_rate_limits, method, path, kwargs
):
    r = lenient_client.request(method, path, headers={"X-API-Key": "wrong"}, **kwargs)
    assert r.status_code == 403, path


def test_every_mutating_route_of_the_split_modules_is_covered_above():
    """A new POST/PUT/DELETE in these modules must be added to MUTATING_ROUTES."""
    from src.api.routes import alerts, analytics, data, export

    covered = {(m, p) for m, p, _ in MUTATING_ROUTES}
    public = {
        # subscription / verification flow stays public by design
        ("POST", "/api/v4/subscribe"),
        ("POST", "/api/alerts/send-verification"),
        ("POST", "/api/alerts/verify-email"),
        ("POST", "/api/alerts/send-insight-report"),
    }
    found = set()
    for module in (alerts, analytics, data, export):
        for route in module.router.routes:
            for method in getattr(route, "methods", set()):
                if method in {"POST", "PUT", "DELETE"}:
                    found.add((method, route.path))
    assert found - public == covered - public
    assert not (found & public) - public


# ===========================================================================
# 6. Structure contracts (F6 targets)
# ===========================================================================

SERVICE_MODULES = [
    "export_service.py",
    "external_signals_service.py",
    "alert_service.py",
    "analytics_service.py",
    "sos_trend_service.py",
    "historical_service.py",
    "brand_matrix.py",
    "sql_rows.py",
]


@pytest.mark.parametrize("name", SERVICE_MODULES)
def test_service_modules_exist_and_import_without_fastapi(name):
    path = SERVICES / name
    assert path.exists(), f"missing service module: {path}"
    offenders = sorted(
        t for t in _import_targets(path) if t.split(".")[0] in {"fastapi", "starlette", "slowapi"}
    )
    assert offenders == [], f"{name} must be usable without the web framework: {offenders}"


@pytest.mark.parametrize("name", SERVICE_MODULES)
def test_service_modules_do_not_import_the_api_layer(name):
    offenders = sorted(t for t in _import_targets(SERVICES / name) if t.startswith("src.api"))
    assert offenders == [], f"{name} must not import src.api: {offenders}"


def test_tools_no_longer_reach_back_into_the_api_layer():
    """The `_get_external_signals` inversion (tools -> api) is gone."""
    offenders = sorted(
        t
        for t in _import_targets(SRC / "tools" / "exporters" / "export_handlers.py")
        if t.startswith("src.api")
    )
    assert offenders == [], f"export_handlers must not import src.api: {offenders}"


def test_external_signal_collection_lives_in_the_service_layer():
    from src.application.services.external_signals_service import (
        classify_signal_relevance,
        get_external_signals,
    )

    assert callable(get_external_signals)
    assert callable(classify_signal_relevance)


def test_document_rendering_left_the_route_module():
    """export.py no longer builds documents itself (docx / charts / analysis)."""
    targets = _import_targets(ROUTES / "export.py")
    for forbidden in (
        "docx",
        "docx.shared",
        "src.tools.calculators.period_analyzer",
        "src.tools.exporters.chart_generator",
        "src.tools.utilities.reference_tracker",
        "src.tools.collectors.external_signal_collector",
    ):
        assert forbidden not in targets, f"export.py still imports {forbidden}"


def test_analysis_report_has_a_single_document_path():
    """
    The sync route and the async job render the SAME document: both go through
    ``render_analyst_report`` in src/tools/exporters/export_handlers.py.
    """
    from src.tools.exporters.export_handlers import render_analyst_report

    assert callable(render_analyst_report)
    assert "render_analyst_report" in _referenced_names(ROUTES / "export.py")


def test_the_simple_docx_variant_is_gone():
    """
    The async "simple docx" handler assumed ``brand.kpis`` was a dict of
    ``{"value", "change"}`` objects; the exporter writes scalars, so it raised
    AttributeError on every real payload. The insight report now has one renderer.
    """
    from src.tools.exporters import export_handlers

    source = (SRC / "tools" / "exporters" / "export_handlers.py").read_text(encoding="utf-8")
    assert "kpi_data.get" not in source
    assert callable(export_handlers.render_insight_report)


@pytest.mark.parametrize(
    "module,names",
    [
        ("analytics.py", ["AnalyticsService"]),
        ("data.py", ["HistoricalService"]),
    ],
)
def test_routes_delegate_to_their_service(module, names):
    referenced = _referenced_names(ROUTES / module)
    missing = [n for n in names if n not in referenced]
    assert missing == [], f"{module} does not use {missing}"
