"""
API contract characterization
=============================
Pins the CURRENT behaviour of the public HTTP surface of `src.api.dashboard_api:app`.
These tests describe what the code does today, including behaviour that looks like a
bug (marked `PINS CURRENT BEHAVIOR`). They must be updated deliberately when the
contract changes.
"""

import json

import pytest

# ---------------------------------------------------------------------------
# Fixtures: dashboard JSON shaped like src/tools/exporters/dashboard_exporter.py
# ---------------------------------------------------------------------------

DASHBOARD_TOP_LEVEL_KEYS = ["metadata", "data_source", "home", "brand", "categories", "products", "charts"]


def _minimal_dashboard_json() -> dict:
    """Shape mirrors DashboardExporter.export_dashboard_data output (products keyed by ASIN)."""
    return {
        "metadata": {
            "generated_at": "2026-09-01T22:30:00+09:00",
            "data_date": "2026-09-01",
            "total_products": 2,
            "laneige_products": 1,
            "ontology_enabled": False,
        },
        "data_source": {
            "platform": "Amazon US Best Sellers",
            "collected_at": "2026-09-01T22:30:00+09:00",
            "snapshot_date": "2026-09-01",
            "disclaimer": "snapshot",
            "url": "https://www.amazon.com/gp/bestsellers/beauty",
        },
        "home": {
            "insight_message": "LANEIGE Lip Sleeping Mask #1 in Lip Care",
            "status": {
                "exposure": "Strong",
                "exposure_type": "success",
                "position": "Top 1",
                "warning_count": 0,
            },
            "action_items": [
                {
                    "asin": "B07GFJWPDQ",
                    "product_name": "LANEIGE Lip Sleeping Mask",
                    "brand_variant": "LANEIGE",
                    "rank": 1,
                    "rank_change": 0,
                    "signal": "순위 #1",
                    "signal_detail": "",
                    "action_tag": "Monitor",
                    "priority": "P1",
                }
            ],
        },
        "brand": {
            "kpis": {
                "sos": 12.5,
                "sos_delta": "+2.1%p",
                "top10_count": 1,
                "avg_rank": 1.0,
                "avg_price": 24.0,
                "hhi": 812.5,
            },
            "competitors": [
                {"brand": "LANEIGE", "sos": 50.0, "avg_rank": 1.0, "product_count": 1},
                {"brand": "Burt's Bees", "sos": 50.0, "avg_rank": 2.0, "product_count": 1},
            ],
        },
        "categories": {
            "lip_care": {
                "name": "Lip Care",
                "sos": 50.0,
                "best_rank": 1,
                "cpi": 150.0,
                "new_competitors": 2,
            }
        },
        "products": {
            "B07GFJWPDQ": {
                "name": "LANEIGE Lip Sleeping Mask",
                "rank": 1,
                "rank_delta": "0",
                "rating": 4.6,
                "volatility_status": "안정적",
                "price": 24.0,
                "category": "lip_care",
            }
        },
        "charts": {},
    }


@pytest.fixture
def dashboard_file(isolated_cwd):
    """Write a valid ./data/dashboard_data.json in the isolated CWD (DATA_PATH is relative)."""
    data_dir = isolated_cwd / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "dashboard_data.json"
    path.write_text(json.dumps(_minimal_dashboard_json(), ensure_ascii=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Health / OpenAPI
# ---------------------------------------------------------------------------


def test_health_returns_200_with_status_and_timestamp(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"status", "timestamp"}
    assert body["status"] == "healthy"


def test_root_returns_feature_list(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {"status", "message", "features"}
    assert body["features"] == ["chatbot", "rag", "ontology", "memory", "docx_export"]


def test_openapi_path_count_is_pinned(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    paths = r.json()["paths"]
    # Snapshot of the route surface; bump deliberately when routes are added/removed.
    assert len(paths) == 71
    for must_have in [
        "/api/health",
        "/api/data",
        "/api/data/refresh",
        "/api/v4/chat",
        "/api/v4/chat/stream",
        "/api/crawl/start",
        "/api/v4/brain/status",
        "/api/v4/alert-settings",
        "/api/signals/clear",
        "/api/sync/upload",
        "/api/export/docx",
    ]:
        assert must_have in paths, must_have


def test_trusted_host_rejects_unknown_host(app):
    """TrustedHostMiddleware allows only localhost/127.0.0.1/.railway.app by default."""
    from fastapi.testclient import TestClient

    r = TestClient(app, base_url="http://testserver").get("/api/health")
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/data
# ---------------------------------------------------------------------------


def test_get_data_with_valid_cache_file(client, dashboard_file):
    r = client.get("/api/data")
    assert r.status_code == 200
    body = r.json()
    assert list(body.keys()) == DASHBOARD_TOP_LEVEL_KEYS

    meta = body["metadata"]
    # load_dashboard_data() annotates staleness onto metadata
    assert "_cache_age_hours" in meta
    assert "_is_stale" in meta
    assert meta["_is_stale"] is False
    assert meta["_cache_age_hours"] == 0.0
    assert "_is_empty" not in meta

    # payload passes through untouched
    assert body["brand"]["kpis"]["sos"] == 12.5
    assert list(body["products"].keys()) == ["B07GFJWPDQ"]
    assert body["categories"]["lip_care"]["name"] == "Lip Care"


def test_get_data_without_file_or_sqlite_returns_empty_skeleton(client, isolated_cwd):
    """No ./data/dashboard_data.json and no SQLite rows -> 200 with an empty skeleton (not 404)."""
    r = client.get("/api/data")
    assert r.status_code == 200
    body = r.json()
    assert list(body.keys()) == ["metadata", "home", "brand", "products", "categories", "charts"]
    assert body["metadata"] == {
        "data_date": None,
        "total_products": 0,
        "_is_stale": False,
        "_is_empty": True,
        "_message": "데이터가 없습니다. 크롤링을 실행하여 데이터를 수집하세요.",
    }
    assert body["home"] == {"action_items": [], "status": {}, "summary": {}}
    assert body["brand"] == {"kpis": {}, "competitors": []}
    assert body["products"] == {}
    assert body["categories"] == {}
    assert body["charts"] == {}


def test_get_data_with_corrupted_file_falls_back_to_skeleton(client, isolated_cwd):
    data_dir = isolated_cwd / "data"
    data_dir.mkdir()
    (data_dir / "dashboard_data.json").write_text("{not json", encoding="utf-8")
    r = client.get("/api/data")
    assert r.status_code == 200
    assert r.json()["metadata"]["_is_empty"] is True


# ---------------------------------------------------------------------------
# Auth matrix — WITHOUT an X-API-Key header, process has no API_KEY configured
# ---------------------------------------------------------------------------
# verify_api_key (src/api/dependencies.py) dev-mode behaviour: when API_KEY is None
# it raises 503 "Server not configured for authenticated access" for every caller,
# key or not. Routes NOT wired to verify_api_key are fully open (bug D8).


def test_auth_verify_api_key_routes_return_503_when_server_has_no_api_key(client, isolated_cwd):
    # Routes protected by Depends(verify_api_key)
    assert client.post("/api/crawl/start").status_code == 503
    assert client.post("/api/v4/chat", json={"message": "hi"}).status_code == 503
    assert client.post("/api/v4/chat/stream", json={"message": "hi"}).status_code == 503
    assert client.post("/api/chat", json={"message": "hi"}).status_code == 503

    r = client.post("/api/crawl/start")
    assert r.json() == {"detail": "Server not configured for authenticated access"}

    # Even a supplied header cannot pass: 503 wins over 401/403 in dev mode.
    r = client.post("/api/crawl/start", headers={"X-API-Key": "anything"})
    assert r.status_code == 503


def test_auth_post_data_refresh_is_unauthenticated(client, isolated_cwd):
    # PINS CURRENT BEHAVIOR (bug D8): no auth on a mutating route.
    r = client.post("/api/data/refresh")
    assert r.status_code == 200
    assert r.json() == {"success": False, "error": "No data found"}


def test_auth_put_alert_settings_is_unauthenticated(client, isolated_cwd):
    # PINS CURRENT BEHAVIOR (bug D8): validation runs, no auth check -> 422 for missing field.
    r = client.put("/api/v4/alert-settings", json={"email": "a@b.c"})
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "alert_types"]

    # With a complete body the handler executes and reports unknown subscriber.
    r = client.put("/api/v4/alert-settings", json={"email": "a@b.c", "alert_types": ["rank_drop"]})
    assert r.status_code == 404
    assert r.json() == {"detail": "등록되지 않은 이메일입니다."}


def test_auth_delete_alert_settings_is_unauthenticated(client, isolated_cwd):
    # PINS CURRENT BEHAVIOR (bug D8): anyone can attempt to unsubscribe any email.
    r = client.delete("/api/v4/alert-settings", params={"email": "a@b.c"})
    assert r.status_code == 404
    assert r.json() == {"detail": "등록되지 않은 이메일입니다."}


def test_auth_delete_signals_clear_is_unauthenticated_and_destructive(client, isolated_cwd):
    # PINS CURRENT BEHAVIOR (bug D8): DELETE /api/signals/clear wipes persisted signals with no auth.
    r = client.delete("/api/signals/clear")
    assert r.status_code == 200
    assert r.json() == {"status": "success", "message": "All signals cleared"}
    written = isolated_cwd / "data" / "external_signals" / "signals.json"
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8"))["signals"] == []


def test_auth_post_sync_upload_is_unauthenticated(client, isolated_cwd):
    # PINS CURRENT BEHAVIOR (bug D8): sync only checks the body "api_key" when API_KEY env is set.
    # No body at all -> JSON decode error is swallowed into a 500.
    r = client.post("/api/sync/upload")
    assert r.status_code == 500
    assert r.json()["detail"].startswith("Expecting value")

    r = client.post("/api/sync/upload", json={"records": []})
    assert r.status_code == 400
    assert r.json() == {"detail": "No records provided"}


def test_auth_post_export_docx_is_unauthenticated_and_500s(lenient_client, dashboard_file):
    # PINS CURRENT BEHAVIOR (bug D8 + bug: slowapi decorator): the handler names its
    # pydantic body `request` and the starlette Request `http_request`, so slowapi's
    # limiter rejects the call before the handler body runs. The global exception
    # handler turns it into a 500 JSON envelope.
    r = lenient_client.post("/api/export/docx", json={})
    assert r.status_code == 500
    body = r.json()
    assert body["error"] == "Internal server error"
    assert "parameter `request` must be an instance of starlette.requests.Request" in body["detail"]


def test_auth_post_brain_check_alerts_is_unauthenticated(client, isolated_cwd):
    # PINS CURRENT BEHAVIOR (bug D8): no auth. Offline (no OPENAI_API_KEY) the brain's
    # initialize() raises inside the handler and the error is returned as a 200 payload.
    r = client.post("/api/v4/brain/check-alerts")
    assert r.status_code == 200
    body = r.json()
    assert body["alerts"] == []
    assert "error" in body
    assert body["error"].startswith("Vector search is required but not available")


# ---------------------------------------------------------------------------
# Chat v4 (auth bypassed via FastAPI dependency_overrides, LLM patched)
# ---------------------------------------------------------------------------

BRAIN_CHAT_RESPONSE_KEYS = {
    "text",
    "confidence",
    "sources",
    "reasoning",
    "tools_used",
    "processing_time_ms",
    "from_cache",
    "brain_mode",
    "suggestions",
    "query_type",
}


def test_chat_v4_offline_returns_error_fallback_payload(client, isolated_cwd, auth_bypass, patched_llm):
    """
    PINS CURRENT BEHAVIOR: with no OPENAI_API_KEY, UnifiedBrain.initialize() fails
    (DocumentRetriever refuses to start without vector search) and chat_v4 answers
    200 with an error-mode BrainChatResponse instead of a 5xx. The canned LLM is
    never reached.
    """
    r = client.post("/api/v4/chat", json={"message": "안녕"})
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == BRAIN_CHAT_RESPONSE_KEYS
    assert body["brain_mode"] == "error"
    assert body["confidence"] == 0.0
    assert body["text"].startswith("처리 중 오류가 발생했습니다: Vector search is required")
    assert body["sources"] == []
    assert body["tools_used"] == []
    assert body["suggestions"] == []
    assert body["query_type"] == "unknown"
    assert body["reasoning"] is None
    assert body["from_cache"] is False
    assert isinstance(body["processing_time_ms"], float)
    assert patched_llm == []  # LLM never called on the fallback path


def test_chat_v4_rejects_blank_message(client, isolated_cwd, auth_bypass):
    r = client.post("/api/v4/chat", json={"message": "   "})
    assert r.status_code == 400
    assert r.json() == {"detail": "Message is required"}


def test_chat_v4_missing_message_is_422(client, isolated_cwd, auth_bypass):
    r = client.post("/api/v4/chat", json={})
    assert r.status_code == 422


def test_chat_v4_stream_is_post_only(client):
    r = client.get("/api/v4/chat/stream", params={"message": "안녕"})
    assert r.status_code == 405


def test_chat_v4_stream_offline_returns_500_json_before_sse(client, isolated_cwd, auth_bypass, patched_llm):
    """
    PINS CURRENT BEHAVIOR: the SSE contract (text/event-stream, `data:` frames) is only
    reachable once the brain initialises. Offline the route raises HTTPException(500)
    with a JSON body — no event-stream headers are ever sent.
    """
    r = client.post("/api/v4/chat/stream", json={"message": "안녕"})
    assert r.status_code == 500
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["detail"].startswith("Vector search is required but not available")
    assert patched_llm == []


def test_chat_v4_stream_rejects_blank_message(client, isolated_cwd, auth_bypass):
    r = client.post("/api/v4/chat/stream", json={"message": ""})
    assert r.status_code == 400
    assert r.json() == {"detail": "Message is required"}


def test_chat_memory_delete_is_unauthenticated(client):
    # PINS CURRENT BEHAVIOR (bug D8): any caller can wipe any session's memory.
    r = client.delete("/api/chat/memory/some-session")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "message": "Session some-session memory cleared"}
