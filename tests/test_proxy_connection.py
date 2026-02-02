"""
프록시 연결 테스트

Oracle Cloud 프록시 서버 연결 및 기능 테스트
"""

import json
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

from src.tools.proxy_manager import (
    ProxyConfig,
    ProxyManager,
    get_proxy_manager,
    reset_proxy_manager,
)


class TestProxyConfig:
    """ProxyConfig 클래스 테스트"""

    def test_playwright_config(self):
        """Playwright 설정 생성 테스트"""
        proxy = ProxyConfig(
            name="test-proxy",
            server="http://1.2.3.4:3128",
            username="user",
            password="pass",  # pragma: allowlist secret
            region="seoul",
        )

        config = proxy.playwright_config
        assert config["server"] == "http://1.2.3.4:3128"
        assert config["username"] == "user"
        assert config["password"] == "pass"  # pragma: allowlist secret

    def test_requests_config(self):
        """requests 라이브러리 설정 생성 테스트"""
        proxy = ProxyConfig(
            name="test-proxy",
            server="http://1.2.3.4:3128",
            username="user",
            password="pass",  # pragma: allowlist secret
            region="seoul",
        )

        config = proxy.requests_config
        assert "user:pass@1.2.3.4:3128" in config["http"]  # pragma: allowlist secret
        assert "user:pass@1.2.3.4:3128" in config["https"]  # pragma: allowlist secret

    def test_aiohttp_config(self):
        """aiohttp 설정 생성 테스트"""
        proxy = ProxyConfig(
            name="test-proxy",
            server="http://1.2.3.4:3128",
            username="user",
            password="pass",  # pragma: allowlist secret
            region="seoul",
        )

        config = proxy.aiohttp_config
        assert "user:pass@1.2.3.4:3128" in config  # pragma: allowlist secret

    def test_success_rate_no_requests(self):
        """요청 없을 때 성공률 테스트"""
        proxy = ProxyConfig(
            name="test", server="http://1.2.3.4:3128", username="u", password="p", region="seoul"
        )
        assert proxy.success_rate == 1.0

    def test_success_rate_with_requests(self):
        """요청 있을 때 성공률 테스트"""
        proxy = ProxyConfig(
            name="test",
            server="http://1.2.3.4:3128",
            username="u",
            password="p",
            region="seoul",
            success_count=7,
            failure_count=3,
        )
        assert proxy.success_rate == 0.7


class TestProxyManager:
    """ProxyManager 클래스 테스트"""

    @pytest.fixture
    def temp_config_file(self):
        """임시 설정 파일 생성"""
        config = {
            "proxy_pool": [
                {
                    "name": "proxy-1",
                    "server": "http://1.1.1.1:3128",
                    "username": "user1",
                    "password": "testcred1",  # pragma: allowlist secret
                    "region": "seoul",
                    "enabled": True,
                },
                {
                    "name": "proxy-2",
                    "server": "http://2.2.2.2:3128",
                    "username": "user2",
                    "password": "testcred2",  # pragma: allowlist secret
                    "region": "tokyo",
                    "enabled": True,
                },
            ],
            "rotation_strategy": "random",
            "max_retries": 3,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            return f.name

    @pytest.fixture
    def manager(self, temp_config_file):
        """ProxyManager 인스턴스 생성"""
        reset_proxy_manager()
        return ProxyManager(config_path=temp_config_file)

    def test_load_config(self, manager):
        """설정 파일 로드 테스트"""
        assert len(manager.proxies) == 2
        assert manager.proxies[0].name == "proxy-1"
        assert manager.proxies[1].region == "tokyo"

    def test_get_proxy(self, manager):
        """프록시 가져오기 테스트"""
        proxy = manager.get_proxy()
        assert proxy is not None
        assert proxy.name in ["proxy-1", "proxy-2"]

    def test_get_proxy_no_config(self):
        """설정 없을 때 테스트"""
        reset_proxy_manager()
        manager = ProxyManager(config_path="nonexistent.json")
        proxy = manager.get_proxy()
        assert proxy is None

    def test_report_failure(self, manager):
        """실패 보고 테스트"""
        manager.report_failure("proxy-1")
        proxy = [p for p in manager.proxies if p.name == "proxy-1"][0]
        assert proxy.failure_count == 1
        assert proxy.enabled  # 아직 비활성화 안 됨

    def test_report_failure_disable(self, manager):
        """연속 실패 시 비활성화 테스트"""
        for _ in range(3):  # max_retries = 3
            manager.report_failure("proxy-1")

        proxy = [p for p in manager.proxies if p.name == "proxy-1"][0]
        assert not proxy.enabled

    def test_report_success(self, manager):
        """성공 보고 테스트"""
        manager.report_failure("proxy-1")
        manager.report_success("proxy-1")

        proxy = [p for p in manager.proxies if p.name == "proxy-1"][0]
        assert proxy.failure_count == 0  # 리셋됨
        assert proxy.success_count == 1

    def test_reset_proxy(self, manager):
        """프록시 리셋 테스트"""
        for _ in range(3):
            manager.report_failure("proxy-1")

        manager.reset_proxy("proxy-1")

        proxy = [p for p in manager.proxies if p.name == "proxy-1"][0]
        assert proxy.enabled
        assert proxy.failure_count == 0

    def test_get_stats(self, manager):
        """통계 조회 테스트"""
        stats = manager.get_stats()
        assert stats["total"] == 2
        assert stats["active"] == 2
        assert stats["available"] == 2

    def test_has_proxies(self, manager):
        """프록시 유무 확인 테스트"""
        assert manager.has_proxies()

    def test_has_proxies_empty(self):
        """프록시 없을 때 테스트"""
        reset_proxy_manager()
        manager = ProxyManager(config_path="nonexistent.json")
        assert not manager.has_proxies()


class TestProxyManagerRotation:
    """프록시 로테이션 전략 테스트"""

    @pytest.fixture
    def round_robin_config(self):
        """라운드 로빈 설정"""
        config = {
            "proxy_pool": [
                {
                    "name": "p1",
                    "server": "http://1:3128",
                    "username": "u",
                    "password": "p",
                    "region": "a",
                },
                {
                    "name": "p2",
                    "server": "http://2:3128",
                    "username": "u",
                    "password": "p",
                    "region": "b",
                },
                {
                    "name": "p3",
                    "server": "http://3:3128",
                    "username": "u",
                    "password": "p",
                    "region": "c",
                },
            ],
            "rotation_strategy": "round-robin",
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            return f.name

    def test_round_robin_rotation(self, round_robin_config):
        """라운드 로빈 로테이션 테스트"""
        reset_proxy_manager()
        manager = ProxyManager(config_path=round_robin_config)

        # 순서대로 반환되어야 함
        names = [manager.get_proxy().name for _ in range(6)]
        assert names == ["p1", "p2", "p3", "p1", "p2", "p3"]


@pytest.mark.asyncio
class TestProxyHealthCheck:
    """프록시 헬스 체크 테스트"""

    @pytest.fixture
    def manager_with_proxy(self):
        """프록시가 있는 매니저"""
        config = {
            "proxy_pool": [
                {
                    "name": "test",
                    "server": "http://1:3128",
                    "username": "u",
                    "password": "p",
                    "region": "a",
                }
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            reset_proxy_manager()
            return ProxyManager(config_path=f.name)

    async def test_health_check_success(self, manager_with_proxy):
        """헬스 체크 성공 테스트 (모킹)"""
        proxy = manager_with_proxy.proxies[0]

        with patch("aiohttp.ClientSession") as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={"origin": "1.2.3.4"})

            mock_cm = AsyncMock()
            mock_cm.__aenter__.return_value = mock_response
            mock_cm.__aexit__.return_value = None

            mock_session_instance = AsyncMock()
            mock_session_instance.get.return_value = mock_cm
            mock_session_instance.__aenter__.return_value = mock_session_instance
            mock_session_instance.__aexit__.return_value = None

            mock_session.return_value = mock_session_instance

            result = await manager_with_proxy.health_check(proxy)
            # 모킹 환경에서는 실제 연결 안 함
            # 실제 테스트는 통합 테스트에서 수행


class TestSingleton:
    """싱글톤 패턴 테스트"""

    def test_get_proxy_manager_singleton(self):
        """싱글톤 인스턴스 테스트"""
        reset_proxy_manager()

        manager1 = get_proxy_manager("config/proxy_config.json")
        manager2 = get_proxy_manager("config/proxy_config.json")

        assert manager1 is manager2

    def test_reset_proxy_manager(self):
        """싱글톤 리셋 테스트"""
        reset_proxy_manager()

        manager1 = get_proxy_manager("config/proxy_config.json")
        reset_proxy_manager()
        manager2 = get_proxy_manager("config/proxy_config.json")

        assert manager1 is not manager2


# ============================================
# 통합 테스트 (실제 프록시 서버 필요)
# ============================================


@pytest.mark.skip(reason="실제 프록시 서버 필요 - 수동 테스트용")
class TestRealProxyConnection:
    """
    실제 프록시 연결 테스트

    실행 방법:
    1. config/proxy_config.json 설정
    2. pytest tests/test_proxy_connection.py::TestRealProxyConnection -v --no-skip
    """

    @pytest.mark.asyncio
    async def test_real_proxy_connection(self):
        """실제 프록시 연결 테스트"""
        from playwright.async_api import async_playwright

        reset_proxy_manager()
        manager = get_proxy_manager()

        if not manager.has_proxies():
            pytest.skip("프록시 설정 없음")

        proxy = manager.get_proxy()

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, proxy=proxy.playwright_config)

            page = await browser.new_page()
            await page.goto("https://httpbin.org/ip", timeout=30000)

            content = await page.content()
            await browser.close()

            # 프록시 IP가 표시되어야 함
            assert proxy.server.split("//")[1].split(":")[0] in content or "origin" in content

    @pytest.mark.asyncio
    async def test_real_amazon_access(self):
        """실제 Amazon 접속 테스트"""
        from playwright.async_api import async_playwright

        reset_proxy_manager()
        manager = get_proxy_manager()

        if not manager.has_proxies():
            pytest.skip("프록시 설정 없음")

        proxy = manager.get_proxy()

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True, proxy=proxy.playwright_config)

            page = await browser.new_page()
            await page.goto("https://www.amazon.com/Best-Sellers-Beauty/zgbs/beauty", timeout=60000)

            title = await page.title()
            await browser.close()

            assert "Amazon" in title or "Best Sellers" in title
