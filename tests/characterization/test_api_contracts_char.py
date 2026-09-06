"""
API contract characterization
=============================
Pins the CURRENT behaviour of the public HTTP surface of `src.api.dashboard_api:app`.
These tests describe what the code does today, including behaviour that looks like a
bug (marked `PINS CURRENT BEHAVIOR`). They must be updated deliberately when the
contract changes.
"""

import json

# Fixtures (env setup + app/client/auth/LLM fakes) live in _api_fixtures.py and are
# registered as a pytest plugin for this module, independent of the shared conftest.
pytest_plugins = ["tests.characterization._api_fixtures"]

# Dashboard JSON fixture (exporter-shaped) is shared with the unit tests via
# tests/characterization/_api_fixtures.py: `dashboard_file`, `exporter_shaped_dashboard()`.
from tests.characterization._api_fixtures import DASHBOARD_TOP_LEVEL_KEYS  # noqa: E402

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
# Auth matrix
# ---------------------------------------------------------------------------
# verify_api_key (src/api/dependencies.py) reads API_KEY from the environment at call
# time. Dev-mode behaviour (no API_KEY configured, the default for this suite): it raises
# 503 "Server not configured for authenticated access" for every caller, key or not.
# With API_KEY configured (`configured_api_key` / `auth_headers` fixtures): missing header
# -> 401, wrong header -> 403. Every mutating route is wired to it (D8 fixed); the
# subscribe / send-verification / verify-email / confirm-email flow stays public.


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


def test_auth_post_data_refresh_requires_api_key(client, isolated_cwd, auth_headers):
    # D8 fixed: header required; the handler still reports "No data found" on an empty DB.
    assert client.post("/api/data/refresh").status_code == 401
    r = client.post("/api/data/refresh", headers=auth_headers)
    assert r.status_code == 200
    assert r.json() == {"success": False, "error": "No data found"}


def test_auth_post_data_refresh_is_503_in_dev_mode(client, isolated_cwd):
    assert client.post("/api/data/refresh").status_code == 503


def test_auth_put_alert_settings_requires_api_key(client, isolated_cwd, auth_headers):
    # D8 fixed: the auth dependency runs before body validation.
    r = client.put("/api/v4/alert-settings", json={"email": "a@b.c"})
    assert r.status_code == 401

    # Authenticated: validation runs -> 422 for the missing field.
    r = client.put("/api/v4/alert-settings", json={"email": "a@b.c"}, headers=auth_headers)
    assert r.status_code == 422
    assert r.json()["detail"][0]["loc"] == ["body", "alert_types"]

    # With a complete body the handler executes and reports unknown subscriber.
    r = client.put(
        "/api/v4/alert-settings",
        json={"email": "a@b.c", "alert_types": ["rank_drop"]},
        headers=auth_headers,
    )
    assert r.status_code == 404
    assert r.json() == {"detail": "등록되지 않은 이메일입니다."}


def test_auth_delete_alert_settings_requires_api_key(client, isolated_cwd, auth_headers):
    # D8 fixed: unsubscribing an arbitrary email needs the API key.
    assert client.delete("/api/v4/alert-settings", params={"email": "a@b.c"}).status_code == 401
    r = client.delete("/api/v4/alert-settings", params={"email": "a@b.c"}, headers=auth_headers)
    assert r.status_code == 404
    assert r.json() == {"detail": "등록되지 않은 이메일입니다."}


def test_auth_delete_signals_clear_requires_api_key(client, isolated_cwd, auth_headers):
    # D8 fixed: the destructive route no longer touches disk without a valid key.
    written = isolated_cwd / "data" / "external_signals" / "signals.json"
    assert client.delete("/api/signals/clear").status_code == 401
    assert not written.exists()

    r = client.delete("/api/signals/clear", headers=auth_headers)
    assert r.status_code == 200
    assert r.json() == {"status": "success", "message": "All signals cleared"}
    assert written.exists()
    assert json.loads(written.read_text(encoding="utf-8"))["signals"] == []


def test_auth_post_sync_upload_requires_api_key(
    client, isolated_cwd, auth_headers, reset_rate_limits
):
    # D8 fixed: header auth via verify_api_key (the route is limited to 2/minute, so at
    # most two authenticated calls per test).
    assert client.post("/api/sync/upload", json={"records": []}).status_code == 401

    # No body at all -> JSON decode error is swallowed into a 500 (unchanged).
    r = client.post("/api/sync/upload", headers=auth_headers)
    assert r.status_code == 500
    assert r.json()["detail"].startswith("Expecting value")

    r = client.post("/api/sync/upload", json={"records": []}, headers=auth_headers)
    assert r.status_code == 400
    assert r.json() == {"detail": "No records provided"}


def test_auth_post_sync_upload_body_key_must_match_when_present(
    client, isolated_cwd, auth_headers, reset_rate_limits
):
    # The legacy body "api_key" (scripts/sync_to_railway.py) is compared with
    # hmac.compare_digest when present and never skipped when API_KEY is unset.
    r = client.post(
        "/api/sync/upload", json={"records": [{}], "api_key": "wrong"}, headers=auth_headers
    )
    assert r.status_code == 401
    assert r.json() == {"detail": "Invalid API key"}


def test_auth_post_sync_upload_is_503_in_dev_mode(client, isolated_cwd):
    assert client.post("/api/sync/upload", json={"records": []}).status_code == 503
    assert client.post("/api/sync/upload").status_code == 503


def test_auth_post_export_docx_requires_api_key_and_returns_docx(
    lenient_client, dashboard_file, auth_headers, reset_rate_limits
):
    # D8 fixed: auth required. D22 fixed: the handler names the starlette Request
    # `request` and its pydantic body `payload`, so slowapi accepts the call and the
    # route streams a .docx (the exporter-shaped JSON is adapted via src/api/dashboard_shape).
    assert lenient_client.post("/api/export/docx", json={}).status_code == 401

    r = lenient_client.post(
        "/api/export/docx", json={"include_external_signals": False}, headers=auth_headers
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )
    assert r.headers["content-disposition"].startswith("attachment; filename=AMORE_Insight_Report_")
    assert r.content[:2] == b"PK"  # OOXML zip container


def test_auth_post_export_docx_is_503_in_dev_mode(lenient_client, dashboard_file):
    assert lenient_client.post("/api/export/docx", json={}).status_code == 503


def test_auth_post_brain_check_alerts_requires_api_key(client, isolated_cwd, auth_headers):
    # D8 fixed: auth required. Offline (no OPENAI_API_KEY) the brain's initialize()
    # raises inside the handler and the error is still returned as a 200 payload.
    assert client.post("/api/v4/brain/check-alerts").status_code == 401
    r = client.post("/api/v4/brain/check-alerts", headers=auth_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["alerts"] == []
    assert "error" in body
    assert body["error"].startswith("Vector search is required but not available")


def test_auth_public_subscription_flow_needs_no_api_key(client, isolated_cwd, configured_api_key):
    # Subscribe / verification routes stay public even when API_KEY is configured.
    assert client.post("/api/v4/subscribe", json={"email": "a@b.c"}).status_code not in (401, 403)
    assert client.post(
        "/api/alerts/send-verification", json={"email": "a@b.c"}
    ).status_code not in (401, 403)
    assert client.post(
        "/api/alerts/verify-email", json={"token": "x", "email": "a@b.c"}
    ).status_code not in (401, 403)
    assert client.get(
        "/api/alerts/confirm-email", params={"token": "x", "email": "a@b.c"}
    ).status_code not in (401, 403)


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


def test_chat_v4_offline_returns_error_fallback_payload(
    client, isolated_cwd, auth_bypass, patched_llm
):
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


def test_chat_v4_stream_offline_emits_sse_error_frame(
    client, isolated_cwd, auth_bypass, patched_llm
):
    """
    D23 fixed: the SSE contract (text/event-stream, `data:` frames) holds even when the
    brain cannot initialise. Offline the route answers 200 with a single `error` frame
    instead of a 500 JSON body.
    """
    r = client.post("/api/v4/chat/stream", json={"message": "안녕"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    frames = [
        json.loads(line[len("data: ") :])
        for line in r.text.splitlines()
        if line.startswith("data: ")
    ]
    assert len(frames) == 1
    assert frames[0]["type"] == "error"
    assert frames[0]["content"].startswith("Vector search is required but not available")
    assert patched_llm == []


def test_chat_v4_stream_rejects_blank_message(client, isolated_cwd, auth_bypass):
    r = client.post("/api/v4/chat/stream", json={"message": ""})
    assert r.status_code == 400
    assert r.json() == {"detail": "Message is required"}


def test_chat_memory_delete_requires_api_key(client, auth_headers):
    # D8 fixed: wiping a session's memory needs the API key.
    assert client.delete("/api/chat/memory/some-session").status_code == 401
    r = client.delete("/api/chat/memory/some-session", headers=auth_headers)
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "message": "Session some-session memory cleared"}


def test_chat_memory_delete_is_503_in_dev_mode(client):
    assert client.delete("/api/chat/memory/some-session").status_code == 503
