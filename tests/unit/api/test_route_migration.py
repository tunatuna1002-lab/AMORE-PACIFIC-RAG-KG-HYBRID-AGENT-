"""
Phase 3: Route Migration Verification Tests
=============================================
dashboard_api.py 엔드포인트가 src/api/routes/로 올바르게 이전되었는지 검증.

각 Step 완료 후 해당 테스트를 활성화 (skip 제거).
"""

import importlib

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_router_paths(module_path: str) -> list[str]:
    """라우트 모듈에서 등록된 엔드포인트 경로 목록을 추출."""
    mod = importlib.import_module(module_path)
    router = getattr(mod, "router", None)
    if router is None:
        return []
    return [route.path for route in router.routes if hasattr(route, "path")]


def _get_app_paths() -> list[str]:
    """dashboard_api.py의 app에 등록된 모든 경로를 추출."""
    from dashboard_api import app

    return [route.path for route in app.routes if hasattr(route, "path")]


# ---------------------------------------------------------------------------
# Step 1: Health + Crawl
# ---------------------------------------------------------------------------


class TestHealthRouteMigration:
    """health.py 라우트 마이그레이션 검증."""

    def test_health_router_has_endpoints(self):
        """health 라우터에 필수 엔드포인트 존재."""
        paths = _get_router_paths("src.api.routes.health")
        assert "/" in paths or "/api/health" in paths

    def test_health_router_included_in_app(self):
        """app에 health 라우터가 include되어 있는지."""
        app_paths = _get_app_paths()
        assert "/api/health" in app_paths


class TestCrawlRouteMigration:
    """crawl.py 라우트 마이그레이션 검증."""

    def test_crawl_router_has_endpoints(self):
        """crawl 라우터에 필수 엔드포인트 존재."""
        paths = _get_router_paths("src.api.routes.crawl")
        assert "/status" in paths or "/start" in paths

    def test_crawl_router_included_in_app(self):
        """app에 crawl 라우터가 include되어 있는지."""
        app_paths = _get_app_paths()
        assert "/api/crawl/status" in app_paths


# ---------------------------------------------------------------------------
# Step 2: Data + Historical
# ---------------------------------------------------------------------------


class TestDataRouteMigration:
    """data.py 라우트 마이그레이션 검증."""

    def test_data_router_has_endpoints(self):
        paths = _get_router_paths("src.api.routes.data")
        assert any("data" in p for p in paths)

    def test_historical_endpoint_exists(self):
        app_paths = _get_app_paths()
        assert "/api/historical" in app_paths


# ---------------------------------------------------------------------------
# Step 3: Deals
# ---------------------------------------------------------------------------


class TestDealsRouteMigration:
    """deals.py 라우트 마이그레이션 검증."""

    def test_deals_router_has_endpoints(self):
        paths = _get_router_paths("src.api.routes.deals")
        expected = ["/", "/summary", "/scrape", "/alerts", "/export"]
        for ep in expected:
            assert any(ep in p for p in paths), f"Missing deals endpoint: {ep}"

    def test_deals_router_included_in_app(self):
        app_paths = _get_app_paths()
        assert "/api/deals/" in app_paths or "/api/deals" in app_paths


# ---------------------------------------------------------------------------
# Step 4: Alerts
# ---------------------------------------------------------------------------


class TestAlertsRouteMigration:
    """alerts.py 라우트 마이그레이션 검증."""

    def test_alerts_v3_endpoints(self):
        paths = _get_router_paths("src.api.routes.alerts")
        assert any("alert-settings" in p for p in paths)
        assert any("alerts" in p for p in paths)

    def test_alerts_v4_endpoints(self):
        paths = _get_router_paths("src.api.routes.alerts")
        assert any("subscribe" in p for p in paths)

    def test_email_verification_endpoints(self):
        paths = _get_router_paths("src.api.routes.alerts")
        assert any("verify-email" in p for p in paths)
        assert any("send-verification" in p for p in paths)
        assert any("confirm-email" in p for p in paths)

    def test_alerts_router_included_in_app(self):
        app_paths = _get_app_paths()
        assert "/api/alerts/status" in app_paths
        assert "/api/v3/alert-settings" in app_paths


# ---------------------------------------------------------------------------
# Step 5: Chat
# ---------------------------------------------------------------------------


class TestChatRouteMigration:
    """chat.py 라우트 마이그레이션 검증."""

    def test_chat_v1_endpoint(self):
        paths = _get_router_paths("src.api.routes.chat")
        assert any("chat" in p for p in paths)

    def test_chat_v4_endpoints(self):
        paths = _get_router_paths("src.api.routes.chat")
        assert any("v4/chat" in p or "chat" in p for p in paths)

    def test_memory_endpoint_in_chat(self):
        """Memory 엔드포인트가 chat.py에 합쳐졌는지."""
        paths = _get_router_paths("src.api.routes.chat")
        assert any("memory" in p for p in paths)

    def test_chat_router_included_in_app(self):
        app_paths = _get_app_paths()
        assert "/api/chat" in app_paths or any("chat" in p for p in app_paths)


# ---------------------------------------------------------------------------
# Step 6: Export
# ---------------------------------------------------------------------------


class TestExportRouteMigration:
    """export.py 라우트 마이그레이션 검증."""

    def test_export_router_has_endpoints(self):
        paths = _get_router_paths("src.api.routes.export")
        assert any("docx" in p for p in paths)
        assert any("excel" in p for p in paths)

    def test_export_router_included_in_app(self):
        app_paths = _get_app_paths()
        assert "/api/export/docx" in app_paths
        assert "/api/export/excel" in app_paths


# ---------------------------------------------------------------------------
# Step 7: Final — dashboard_api.py 축소 검증
# ---------------------------------------------------------------------------


class TestDashboardApiSlimmed:
    """dashboard_api.py가 모놀리스에서 thin shell로 축소되었는지 검증."""

    def test_dashboard_api_line_count(self):
        """dashboard_api.py가 200줄 이하인지."""
        from pathlib import Path

        api_file = Path("dashboard_api.py")
        line_count = len(api_file.read_text().splitlines())
        assert line_count <= 200, f"dashboard_api.py is {line_count} lines (target: <=200)"

    def test_no_inline_endpoints(self):
        """dashboard_api.py에 @app.get/@app.post 인라인 엔드포인트가 없는지."""
        from pathlib import Path

        content = Path("dashboard_api.py").read_text()
        inline_endpoints = [
            line
            for line in content.splitlines()
            if line.strip().startswith("@app.get(")
            or line.strip().startswith("@app.post(")
            or line.strip().startswith("@app.put(")
            or line.strip().startswith("@app.delete(")
        ]
        assert (
            len(inline_endpoints) == 0
        ), f"Found {len(inline_endpoints)} inline endpoints: {inline_endpoints[:3]}"

    def test_all_routers_included(self):
        """모든 라우트 모듈이 app_factory.py에 include되어 있는지."""
        from pathlib import Path

        content = Path("src/api/app_factory.py").read_text()
        expected_routers = [
            "health",
            "crawl",
            "data",
            "deals",
            "alerts",
            "chat",
            "export",
        ]
        for router_name in expected_routers:
            assert (
                router_name in content
            ), f"Router '{router_name}' not found in app_factory.py includes"


# ---------------------------------------------------------------------------
# Step 8: Smoke test — app 기동 가능 여부
# ---------------------------------------------------------------------------


class TestAppBootstrap:
    """FastAPI app이 정상 기동되는지 검증."""

    def test_app_creates_successfully(self):
        """dashboard_api.app이 import 가능하고 FastAPI 인스턴스인지."""
        from fastapi import FastAPI

        from dashboard_api import app

        assert isinstance(app, FastAPI)

    def test_app_has_routes(self):
        """app에 최소 20개 라우트가 등록되어 있는지."""
        from dashboard_api import app

        route_paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert len(route_paths) >= 20, f"Only {len(route_paths)} routes found"
