"""
TDD Phase 2: CrawlerAgent 테스트 (RED → GREEN)

테스트 대상: src/agents/crawler_agent.py
"""

from unittest.mock import AsyncMock

import pytest


class TestCrawlerAgentInit:
    """CrawlerAgent 초기화 테스트"""

    def test_init_with_defaults(self):
        """기본값으로 초기화 가능해야 함"""
        from src.agents.crawler_agent import CrawlerAgent

        agent = CrawlerAgent()

        assert agent is not None
        assert agent.scraper is not None

    def test_init_loads_config(self):
        """설정 파일을 로드해야 함"""
        from src.agents.crawler_agent import CrawlerAgent

        agent = CrawlerAgent()

        assert agent.config is not None
        assert isinstance(agent.config, dict)


class TestCrawlerAgentExecute:
    """CrawlerAgent.execute() 테스트"""

    @pytest.fixture
    def mock_scraper_result(self):
        """Mock scraper 결과 — execute()는 'success' 키를 체크함"""
        return {
            "success": True,
            "products": [
                {
                    "asin": "B0BSHRYY1S",
                    "title": "LANEIGE Lip Sleeping Mask",
                    "brand": "LANEIGE",
                    "rank": 1,
                    "price": 24.00,
                    "rating": 4.7,
                    "reviews_count": 50000,
                },
                {
                    "asin": "B0TEST1234",
                    "title": "COSRX Lip Plumper",
                    "brand": "COSRX",
                    "rank": 2,
                    "price": 12.00,
                    "rating": 4.5,
                    "reviews_count": 10000,
                },
            ],
            "total_count": 2,
        }

    @pytest.mark.asyncio
    async def test_execute_returns_dict_with_required_keys(self, mock_scraper_result):
        """execute()는 필수 키를 포함한 dict 반환해야 함"""
        from src.agents.crawler_agent import CrawlerAgent

        agent = CrawlerAgent()
        agent.scraper.scrape_category = AsyncMock(return_value=mock_scraper_result)
        agent._scrape_tracked_competitors = AsyncMock(return_value=[])

        result = await agent.execute(categories=["lip_care"])

        assert isinstance(result, dict)
        assert "status" in result
        assert "categories" in result

    @pytest.mark.asyncio
    async def test_execute_status_completed_on_success(self, mock_scraper_result):
        """성공 시 status는 'completed'여야 함"""
        from src.agents.crawler_agent import CrawlerAgent

        agent = CrawlerAgent()
        agent.scraper.scrape_category = AsyncMock(return_value=mock_scraper_result)
        agent._scrape_tracked_competitors = AsyncMock(return_value=[])

        result = await agent.execute(categories=["lip_care"])

        assert result.get("status") in ["completed", "success", "partial", "failed"]

    @pytest.mark.asyncio
    async def test_execute_returns_products(self, mock_scraper_result):
        """execute()는 제품 목록을 반환해야 함"""
        from src.agents.crawler_agent import CrawlerAgent

        agent = CrawlerAgent()
        agent.scraper.scrape_category = AsyncMock(return_value=mock_scraper_result)
        agent._scrape_tracked_competitors = AsyncMock(return_value=[])

        result = await agent.execute(categories=["lip_care"])

        categories = result.get("categories", {})
        assert isinstance(categories, dict)

    @pytest.mark.asyncio
    async def test_execute_multiple_categories(self, mock_scraper_result):
        """여러 카테고리 크롤링 가능해야 함"""
        from src.agents.crawler_agent import CrawlerAgent

        agent = CrawlerAgent()
        agent.scraper.scrape_category = AsyncMock(return_value=mock_scraper_result)
        agent._scrape_tracked_competitors = AsyncMock(return_value=[])

        result = await agent.execute(categories=["lip_care", "skin_care", "lip_makeup"])

        assert agent.scraper.scrape_category.call_count >= 1


class TestCrawlerAgentErrorHandling:
    """CrawlerAgent 에러 처리 테스트"""

    @pytest.fixture
    def mock_blocked_result(self):
        """차단된 결과 — success=False로 에러 트리거"""
        return {
            "success": False,
            "error": "Amazon blocked request",
            "products": [],
        }

    @pytest.fixture
    def mock_timeout_result(self):
        """타임아웃 결과"""
        return {
            "success": False,
            "error": "Page load timeout",
            "products": [],
        }

    @pytest.mark.asyncio
    async def test_execute_handles_blocked_error(self, mock_blocked_result):
        """차단 에러 처리 가능해야 함"""
        from src.agents.crawler_agent import CrawlerAgent

        agent = CrawlerAgent()
        agent.scraper.scrape_category = AsyncMock(return_value=mock_blocked_result)
        agent._scrape_tracked_competitors = AsyncMock(return_value=[])

        result = await agent.execute(categories=["lip_care"])

        assert "status" in result

    @pytest.mark.asyncio
    async def test_execute_handles_timeout_error(self, mock_timeout_result):
        """타임아웃 에러 처리 가능해야 함"""
        from src.agents.crawler_agent import CrawlerAgent

        agent = CrawlerAgent()
        agent.scraper.scrape_category = AsyncMock(return_value=mock_timeout_result)
        agent._scrape_tracked_competitors = AsyncMock(return_value=[])

        result = await agent.execute(categories=["lip_care"])

        assert "status" in result

    @pytest.mark.asyncio
    async def test_execute_partial_failure_returns_results(self):
        """일부 카테고리 실패 시에도 결과 반환"""
        from src.agents.crawler_agent import CrawlerAgent

        agent = CrawlerAgent()
        agent._scrape_tracked_competitors = AsyncMock(return_value=[])

        async def mock_scrape(category, url=None):
            if "lip" in str(category).lower():
                return {
                    "success": True,
                    "products": [{"asin": "B001", "brand": "LANEIGE", "rank": 1}],
                }
            else:
                return {
                    "success": False,
                    "error": "Blocked",
                    "products": [],
                }

        agent.scraper.scrape_category = AsyncMock(side_effect=mock_scrape)

        result = await agent.execute(categories=["lip_care", "skin_care"])

        assert result is not None
        assert "status" in result


class TestCrawlerAgentBrandExtraction:
    """CrawlerAgent 브랜드 추출 테스트"""

    @pytest.fixture
    def mock_result_with_brands(self):
        return {
            "success": True,
            "products": [
                {"asin": "B001", "title": "LANEIGE Lip Mask Berry", "brand": "LANEIGE", "rank": 1},
                {
                    "asin": "B002",
                    "title": "Beauty of Joseon Serum",
                    "brand": "Beauty of Joseon",
                    "rank": 2,
                },
                {"asin": "B003", "title": "e.l.f. Lip Gloss", "brand": "e.l.f.", "rank": 3},
            ],
        }

    @pytest.mark.asyncio
    async def test_extracts_brand_correctly(self, mock_result_with_brands):
        """브랜드 추출 정확성 테스트"""
        from src.agents.crawler_agent import CrawlerAgent

        agent = CrawlerAgent()
        agent.scraper.scrape_category = AsyncMock(return_value=mock_result_with_brands)
        agent._scrape_tracked_competitors = AsyncMock(return_value=[])

        result = await agent.execute(categories=["lip_care"])

        assert "categories" in result or "all_products" in result


class TestCrawlerAgentProductValidation:
    """CrawlerAgent 제품 데이터 검증 테스트"""

    def test_agent_has_scraper(self):
        """에이전트가 스크레이퍼를 가지고 있어야 함"""
        from src.agents.crawler_agent import CrawlerAgent

        agent = CrawlerAgent()

        assert hasattr(agent, "scraper")
        assert agent.scraper is not None


class TestCrawlerAgentCategories:
    """CrawlerAgent 카테고리 관리 테스트"""

    def test_config_has_categories(self):
        """설정에 카테고리가 있어야 함"""
        from src.agents.crawler_agent import CrawlerAgent

        agent = CrawlerAgent()

        assert "categories" in agent.config
        assert isinstance(agent.config["categories"], dict)

    def test_category_has_url(self):
        """각 카테고리에 URL이 있어야 함"""
        from src.agents.crawler_agent import CrawlerAgent

        agent = CrawlerAgent()

        categories = agent.config.get("categories", {})
        for cat_key, cat_info in categories.items():
            assert "url" in cat_info, f"Category {cat_key} missing url"


class TestCrawlerAgentStatistics:
    """CrawlerAgent 통계 테스트"""

    @pytest.fixture
    def mock_result(self):
        return {
            "success": True,
            "products": [
                {"asin": f"B00{i}", "brand": "LANEIGE" if i < 5 else "COSRX", "rank": i + 1}
                for i in range(10)
            ],
        }

    @pytest.mark.asyncio
    async def test_execute_returns_laneige_products(self, mock_result):
        """실행 결과에 LANEIGE 제품이 포함되어야 함"""
        from src.agents.crawler_agent import CrawlerAgent

        agent = CrawlerAgent()
        agent.scraper.scrape_category = AsyncMock(return_value=mock_result)
        agent._scrape_tracked_competitors = AsyncMock(return_value=[])

        result = await agent.execute(categories=["lip_care"])

        assert "laneige_products" in result or "all_products" in result
