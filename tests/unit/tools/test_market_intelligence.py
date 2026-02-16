"""
MarketIntelligenceEngine 단위 테스트

테스트 대상: src/tools/intelligence/market_intelligence.py
Coverage target: 50%+
"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.intelligence.market_intelligence import (
    DataLayer,
    LayerData,
    MarketIntelligenceEngine,
    create_market_intelligence_engine,
)


class TestDataLayer:
    """DataLayer 상수 테스트"""

    def test_layer_constants(self):
        """레이어 상수가 올바르게 정의되어야 함"""
        assert DataLayer.LAYER_1_AMAZON == 1
        assert DataLayer.LAYER_2_CONSUMER == 2
        assert DataLayer.LAYER_3_INDUSTRY == 3
        assert DataLayer.LAYER_4_MACRO == 4


class TestLayerData:
    """LayerData 데이터클래스 테스트"""

    def test_init_with_defaults(self):
        """기본값으로 초기화 가능해야 함"""
        layer_data = LayerData(layer=1, layer_name="Test Layer")

        assert layer_data.layer == 1
        assert layer_data.layer_name == "Test Layer"
        assert isinstance(layer_data.data, dict)
        assert isinstance(layer_data.sources, list)
        assert layer_data.collected_at != ""

    def test_init_with_custom_values(self):
        """커스텀 값으로 초기화 가능해야 함"""
        data = {"key": "value"}
        sources = [{"title": "Test Source"}]
        collected_at = "2026-01-15T10:00:00"

        layer_data = LayerData(
            layer=2,
            layer_name="Consumer",
            collected_at=collected_at,
            data=data,
            sources=sources,
        )

        assert layer_data.layer == 2
        assert layer_data.layer_name == "Consumer"
        assert layer_data.collected_at == collected_at
        assert layer_data.data == data
        assert layer_data.sources == sources

    def test_post_init_sets_collected_at(self):
        """collected_at이 없으면 자동으로 설정되어야 함"""
        layer_data = LayerData(layer=1, layer_name="Test")

        assert layer_data.collected_at != ""
        # ISO format 확인
        datetime.fromisoformat(layer_data.collected_at)


class TestMarketIntelligenceEngineInit:
    """MarketIntelligenceEngine 초기화 테스트"""

    def test_init_with_defaults(self):
        """기본값으로 초기화 가능해야 함"""
        engine = MarketIntelligenceEngine()

        assert engine is not None
        assert engine.data_dir == Path("./data/market_intelligence")
        assert engine.public_collector is not None
        assert engine.ir_parser is not None
        assert engine.signal_collector is not None
        assert engine.source_manager is not None
        assert engine._initialized is False
        assert isinstance(engine.layer_data, dict)
        assert len(engine.layer_data) == 0

    def test_init_with_custom_data_dir(self, tmp_path):
        """커스텀 데이터 디렉토리로 초기화 가능해야 함"""
        custom_dir = tmp_path / "custom_market_data"
        engine = MarketIntelligenceEngine(data_dir=str(custom_dir))

        assert engine.data_dir == custom_dir
        assert custom_dir.exists()

    def test_init_with_api_key(self):
        """API 키와 함께 초기화 가능해야 함"""
        api_key = "test-api-key-123"  # pragma: allowlist secret
        engine = MarketIntelligenceEngine(public_data_api_key=api_key)

        assert engine.public_collector is not None

    def test_init_creates_subdirectories(self):
        """초기화 시 하위 디렉토리가 생성되어야 함"""
        engine = MarketIntelligenceEngine()

        # 메인 디렉토리 생성 확인
        assert engine.data_dir.exists()


class TestMarketIntelligenceEngineLifecycle:
    """MarketIntelligenceEngine 라이프사이클 테스트"""

    @pytest.mark.asyncio
    async def test_initialize(self):
        """초기화 가능해야 함"""
        engine = MarketIntelligenceEngine()
        engine.public_collector.initialize = AsyncMock()
        engine.ir_parser.initialize = AsyncMock()
        engine.signal_collector.initialize = AsyncMock()

        await engine.initialize()

        assert engine._initialized is True
        engine.public_collector.initialize.assert_called_once()
        engine.ir_parser.initialize.assert_called_once()
        engine.signal_collector.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """중복 초기화는 무시되어야 함"""
        engine = MarketIntelligenceEngine()
        engine.public_collector.initialize = AsyncMock()
        engine.ir_parser.initialize = AsyncMock()
        engine.signal_collector.initialize = AsyncMock()

        await engine.initialize()
        await engine.initialize()

        # 한 번만 호출되어야 함
        engine.public_collector.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self):
        """리소스 정리 가능해야 함"""
        engine = MarketIntelligenceEngine()
        engine.public_collector.close = AsyncMock()
        engine.ir_parser.close = AsyncMock()
        engine.signal_collector.close = AsyncMock()
        engine._initialized = True

        await engine.close()

        assert engine._initialized is False
        engine.public_collector.close.assert_called_once()
        engine.ir_parser.close.assert_called_once()
        engine.signal_collector.close.assert_called_once()


class TestMarketIntelligenceEngineLayer4:
    """Layer 4 (거시경제/무역) 데이터 수집 테스트"""

    @pytest.mark.asyncio
    async def test_collect_layer_4_success(self):
        """Layer 4 데이터 수집 성공 케이스"""
        engine = MarketIntelligenceEngine()

        # Mock 데이터
        mock_us_export = MagicMock()
        mock_us_export.to_dict.return_value = {
            "year": "2026",
            "month": "01",
            "amount_usd": 12300000000,
            "yoy_change": 12.0,
        }

        engine.public_collector.fetch_us_cosmetics_export = AsyncMock(return_value=mock_us_export)
        engine.public_collector.fetch_cosmetics_trade = AsyncMock(return_value=[])
        engine.public_collector.get_trade_summary = MagicMock(return_value=None)
        engine.public_collector.create_source_reference = MagicMock(
            return_value={"title": "관세청"}
        )

        layer_data = await engine.collect_layer_4_macro(year="2026", month="01")

        assert layer_data.layer == DataLayer.LAYER_4_MACRO
        assert layer_data.layer_name == "거시경제/무역"
        assert "us_export" in layer_data.data
        assert layer_data.data["us_export"] is not None
        assert len(layer_data.sources) > 0
        engine.public_collector.fetch_us_cosmetics_export.assert_called_once_with("2026", "01")

    @pytest.mark.asyncio
    async def test_collect_layer_4_with_default_date(self):
        """기본 날짜로 Layer 4 데이터 수집"""
        engine = MarketIntelligenceEngine()
        engine.public_collector.fetch_us_cosmetics_export = AsyncMock(return_value=None)
        engine.public_collector.fetch_cosmetics_trade = AsyncMock(return_value=[])
        engine.public_collector.get_trade_summary = MagicMock(return_value=None)

        layer_data = await engine.collect_layer_4_macro()

        assert layer_data.layer == DataLayer.LAYER_4_MACRO
        assert "us_export" in layer_data.data
        # 현재 연도/월이 사용되어야 함
        assert engine.public_collector.fetch_us_cosmetics_export.called

    @pytest.mark.asyncio
    async def test_collect_layer_4_error_handling(self):
        """Layer 4 수집 중 에러 처리"""
        engine = MarketIntelligenceEngine()
        engine.public_collector.fetch_us_cosmetics_export = AsyncMock(
            side_effect=Exception("API Error")
        )

        layer_data = await engine.collect_layer_4_macro()

        # 에러가 발생해도 LayerData는 반환되어야 함
        assert layer_data.layer == DataLayer.LAYER_4_MACRO
        assert layer_data.data["us_export"] is None

    @pytest.mark.asyncio
    async def test_collect_layer_4_stores_in_layer_data(self):
        """Layer 4 데이터가 engine.layer_data에 저장되어야 함"""
        engine = MarketIntelligenceEngine()
        engine.public_collector.fetch_us_cosmetics_export = AsyncMock(return_value=None)
        engine.public_collector.fetch_cosmetics_trade = AsyncMock(return_value=[])
        engine.public_collector.get_trade_summary = MagicMock(return_value=None)

        await engine.collect_layer_4_macro()

        assert DataLayer.LAYER_4_MACRO in engine.layer_data
        assert engine.layer_data[DataLayer.LAYER_4_MACRO].layer == DataLayer.LAYER_4_MACRO


class TestMarketIntelligenceEngineLayer3:
    """Layer 3 (산업/기업) 데이터 수집 테스트"""

    @pytest.mark.asyncio
    async def test_collect_layer_3_success(self):
        """Layer 3 데이터 수집 성공 케이스"""
        engine = MarketIntelligenceEngine()

        # Mock IR 보고서
        mock_report = MagicMock()
        mock_report.to_dict.return_value = {
            "year": "2025",
            "quarter": "3Q",
            "release_date": "2025-11-06",
        }
        mock_report.year = "2025"
        mock_report.quarter = "3Q"

        engine.ir_parser.get_latest_report = MagicMock(return_value=mock_report)
        engine.ir_parser.create_source_reference = MagicMock(return_value={"title": "IR Report"})
        engine.ir_parser.get_americas_insights = MagicMock(return_value={})
        engine.ir_parser.get_brand_highlights = MagicMock(return_value=[])

        layer_data = await engine.collect_layer_3_industry()

        assert layer_data.layer == DataLayer.LAYER_3_INDUSTRY
        assert layer_data.layer_name == "산업/기업"
        assert "ir_report" in layer_data.data
        assert layer_data.data["ir_report"] is not None

    @pytest.mark.asyncio
    async def test_collect_layer_3_with_specific_quarter(self):
        """특정 분기 IR 데이터 수집"""
        engine = MarketIntelligenceEngine()

        mock_report = MagicMock()
        mock_report.to_dict.return_value = {"year": "2025", "quarter": "2Q"}
        mock_report.year = "2025"
        mock_report.quarter = "2Q"

        engine.ir_parser.get_quarterly_data = MagicMock(return_value=mock_report)
        engine.ir_parser.create_source_reference = MagicMock(return_value={"title": "IR Report"})
        engine.ir_parser.get_americas_insights = MagicMock(return_value={})
        engine.ir_parser.get_brand_highlights = MagicMock(return_value=[])

        layer_data = await engine.collect_layer_3_industry(year="2025", quarter="2Q")

        engine.ir_parser.get_quarterly_data.assert_called_once_with("2025", "2Q")
        assert layer_data.data["ir_report"] is not None

    @pytest.mark.asyncio
    async def test_collect_layer_3_with_brand_highlights(self):
        """브랜드 하이라이트 포함 수집"""
        engine = MarketIntelligenceEngine()

        mock_report = MagicMock()
        mock_report.to_dict.return_value = {"year": "2025", "quarter": "3Q"}
        mock_report.year = "2025"
        mock_report.quarter = "3Q"

        mock_laneige_highlight = MagicMock()
        mock_laneige_highlight.to_dict.return_value = {
            "brand": "LANEIGE",
            "highlights": ["Strong performance in US market"],
        }

        engine.ir_parser.get_latest_report = MagicMock(return_value=mock_report)
        engine.ir_parser.create_source_reference = MagicMock(return_value={"title": "IR Report"})
        engine.ir_parser.get_americas_insights = MagicMock(return_value={})
        engine.ir_parser.get_brand_highlights = MagicMock(
            side_effect=lambda brand, year, quarter: (
                [mock_laneige_highlight] if brand == "LANEIGE" else []
            )
        )

        layer_data = await engine.collect_layer_3_industry()

        assert "brand_highlights" in layer_data.data
        assert "LANEIGE" in layer_data.data["brand_highlights"]

    @pytest.mark.asyncio
    async def test_collect_layer_3_error_handling(self):
        """Layer 3 수집 중 에러 처리"""
        engine = MarketIntelligenceEngine()
        engine.ir_parser.get_latest_report = MagicMock(side_effect=Exception("IR Parser Error"))

        layer_data = await engine.collect_layer_3_industry()

        assert layer_data.layer == DataLayer.LAYER_3_INDUSTRY
        assert layer_data.data["ir_report"] is None


class TestMarketIntelligenceEngineLayer2:
    """Layer 2 (소비자 트렌드) 데이터 수집 테스트"""

    @pytest.mark.asyncio
    async def test_collect_layer_2_success(self):
        """Layer 2 데이터 수집 성공 케이스"""
        engine = MarketIntelligenceEngine()

        # Mock 신호 데이터
        mock_kbeauty_signal = MagicMock()
        mock_kbeauty_signal.to_dict.return_value = {
            "title": "K-Beauty trends in 2026",
            "source": "allure",
            "url": "https://example.com",
        }

        engine.signal_collector.fetch_kbeauty_news = AsyncMock(return_value=[mock_kbeauty_signal])
        engine.signal_collector.fetch_reddit_trends = AsyncMock(return_value=[])
        engine.signal_collector.fetch_industry_signals = AsyncMock(return_value=[])
        engine.signal_collector.create_source_reference = MagicMock(return_value={"title": "News"})

        layer_data = await engine.collect_layer_2_consumer()

        assert layer_data.layer == DataLayer.LAYER_2_CONSUMER
        assert layer_data.layer_name == "소비자 트렌드"
        assert "kbeauty_news" in layer_data.data
        assert len(layer_data.data["kbeauty_news"]) > 0

    @pytest.mark.asyncio
    async def test_collect_layer_2_with_keywords(self):
        """키워드 필터링과 함께 Layer 2 수집"""
        engine = MarketIntelligenceEngine()

        engine.signal_collector.fetch_kbeauty_news = AsyncMock(return_value=[])
        engine.signal_collector.fetch_reddit_trends = AsyncMock(return_value=[])
        engine.signal_collector.fetch_industry_signals = AsyncMock(return_value=[])

        keywords = ["LANEIGE", "lip mask"]
        layer_data = await engine.collect_layer_2_consumer(keywords=keywords)

        engine.signal_collector.fetch_reddit_trends.assert_called_once()
        call_kwargs = engine.signal_collector.fetch_reddit_trends.call_args.kwargs
        assert call_kwargs.get("keywords") == keywords

    @pytest.mark.asyncio
    async def test_collect_layer_2_with_reddit_trends(self):
        """Reddit 트렌드 포함 수집"""
        engine = MarketIntelligenceEngine()

        mock_reddit_signal = MagicMock()
        mock_reddit_signal.to_dict.return_value = {
            "title": "LANEIGE Lip Sleeping Mask review",
            "source": "reddit",
            "metadata": {"upvotes": 500},
        }

        engine.signal_collector.fetch_kbeauty_news = AsyncMock(return_value=[])
        engine.signal_collector.fetch_reddit_trends = AsyncMock(return_value=[mock_reddit_signal])
        engine.signal_collector.fetch_industry_signals = AsyncMock(return_value=[])

        layer_data = await engine.collect_layer_2_consumer()

        assert "reddit_trends" in layer_data.data
        assert len(layer_data.data["reddit_trends"]) > 0

    @pytest.mark.asyncio
    async def test_collect_layer_2_error_handling(self):
        """Layer 2 수집 중 에러 처리"""
        engine = MarketIntelligenceEngine()
        engine.signal_collector.fetch_kbeauty_news = AsyncMock(
            side_effect=Exception("Signal Collector Error")
        )

        layer_data = await engine.collect_layer_2_consumer()

        # 에러가 발생해도 LayerData는 반환되어야 함
        assert layer_data.layer == DataLayer.LAYER_2_CONSUMER
        assert isinstance(layer_data.data, dict)


class TestMarketIntelligenceEngineCollectLayer:
    """특정 레이어 수집 테스트"""

    @pytest.mark.asyncio
    async def test_collect_layer_4(self):
        """Layer 4 수집 라우팅"""
        engine = MarketIntelligenceEngine()
        engine.public_collector.fetch_us_cosmetics_export = AsyncMock(return_value=None)
        engine.public_collector.fetch_cosmetics_trade = AsyncMock(return_value=[])
        engine.public_collector.get_trade_summary = MagicMock(return_value=None)

        layer_data = await engine.collect_layer(4)

        assert layer_data is not None
        assert layer_data.layer == DataLayer.LAYER_4_MACRO

    @pytest.mark.asyncio
    async def test_collect_layer_3(self):
        """Layer 3 수집 라우팅"""
        engine = MarketIntelligenceEngine()
        engine.ir_parser.get_latest_report = MagicMock(return_value=None)

        layer_data = await engine.collect_layer(3)

        assert layer_data is not None
        assert layer_data.layer == DataLayer.LAYER_3_INDUSTRY

    @pytest.mark.asyncio
    async def test_collect_layer_2(self):
        """Layer 2 수집 라우팅"""
        engine = MarketIntelligenceEngine()
        engine.signal_collector.fetch_kbeauty_news = AsyncMock(return_value=[])
        engine.signal_collector.fetch_reddit_trends = AsyncMock(return_value=[])
        engine.signal_collector.fetch_industry_signals = AsyncMock(return_value=[])

        layer_data = await engine.collect_layer(2)

        assert layer_data is not None
        assert layer_data.layer == DataLayer.LAYER_2_CONSUMER

    @pytest.mark.asyncio
    async def test_collect_layer_1_returns_none(self):
        """Layer 1은 기존 시스템에서 처리되므로 None 반환"""
        engine = MarketIntelligenceEngine()

        layer_data = await engine.collect_layer(1)

        assert layer_data is None

    @pytest.mark.asyncio
    async def test_collect_layer_unknown_returns_none(self):
        """알 수 없는 레이어는 None 반환"""
        engine = MarketIntelligenceEngine()

        layer_data = await engine.collect_layer(99)

        assert layer_data is None

    @pytest.mark.asyncio
    async def test_collect_layer_with_kwargs(self):
        """kwargs 전달 테스트"""
        engine = MarketIntelligenceEngine()
        engine.public_collector.fetch_us_cosmetics_export = AsyncMock(return_value=None)
        engine.public_collector.fetch_cosmetics_trade = AsyncMock(return_value=[])
        engine.public_collector.get_trade_summary = MagicMock(return_value=None)

        await engine.collect_layer(4, year="2025", month="12")

        engine.public_collector.fetch_us_cosmetics_export.assert_called_once_with("2025", "12")


class TestMarketIntelligenceEngineCollectAll:
    """전체 레이어 수집 테스트"""

    @pytest.mark.asyncio
    async def test_collect_all_layers(self):
        """모든 레이어(2-4) 병렬 수집"""
        engine = MarketIntelligenceEngine()
        engine._initialized = True

        # Mock 모든 수집기
        engine.public_collector.fetch_us_cosmetics_export = AsyncMock(return_value=None)
        engine.public_collector.fetch_cosmetics_trade = AsyncMock(return_value=[])
        engine.public_collector.get_trade_summary = MagicMock(return_value=None)
        engine.ir_parser.get_latest_report = MagicMock(return_value=None)
        engine.signal_collector.fetch_kbeauty_news = AsyncMock(return_value=[])
        engine.signal_collector.fetch_reddit_trends = AsyncMock(return_value=[])
        engine.signal_collector.fetch_industry_signals = AsyncMock(return_value=[])

        result = await engine.collect_all_layers()

        assert isinstance(result, dict)
        assert DataLayer.LAYER_4_MACRO in result
        assert DataLayer.LAYER_3_INDUSTRY in result
        assert DataLayer.LAYER_2_CONSUMER in result

    @pytest.mark.asyncio
    async def test_collect_all_layers_initializes_if_needed(self):
        """초기화되지 않은 경우 자동 초기화"""
        engine = MarketIntelligenceEngine()
        engine._initialized = False

        engine.public_collector.initialize = AsyncMock()
        engine.ir_parser.initialize = AsyncMock()
        engine.signal_collector.initialize = AsyncMock()
        engine.public_collector.fetch_us_cosmetics_export = AsyncMock(return_value=None)
        engine.public_collector.fetch_cosmetics_trade = AsyncMock(return_value=[])
        engine.public_collector.get_trade_summary = MagicMock(return_value=None)
        engine.ir_parser.get_latest_report = MagicMock(return_value=None)
        engine.signal_collector.fetch_kbeauty_news = AsyncMock(return_value=[])
        engine.signal_collector.fetch_reddit_trends = AsyncMock(return_value=[])
        engine.signal_collector.fetch_industry_signals = AsyncMock(return_value=[])

        await engine.collect_all_layers()

        assert engine._initialized is True


class TestMarketIntelligenceEngineInsightGeneration:
    """인사이트 생성 테스트"""

    def test_generate_layered_insight_basic(self):
        """기본 인사이트 생성"""
        engine = MarketIntelligenceEngine()

        insight = engine.generate_layered_insight()

        assert isinstance(insight, str)
        assert "LANEIGE Amazon US 일일 인사이트" in insight
        assert "생성일:" in insight

    def test_generate_layered_insight_with_layer4(self):
        """Layer 4 데이터 포함 인사이트"""
        engine = MarketIntelligenceEngine()

        # Layer 4 데이터 추가
        engine.layer_data[DataLayer.LAYER_4_MACRO] = LayerData(
            layer=4,
            layer_name="거시경제/무역",
            data={
                "us_export": {
                    "year": "2026",
                    "month": "01",
                    "amount_usd": 12300000000,
                    "yoy_change": 12.0,
                }
            },
        )
        engine.source_manager.add_source = MagicMock(
            return_value=MagicMock(to_citation=MagicMock(return_value="[1]"))
        )

        insight = engine.generate_layered_insight()

        assert "Layer 4: 거시경제/무역" in insight
        assert "화장품 대미 수출" in insight

    def test_generate_layered_insight_with_layer3(self):
        """Layer 3 데이터 포함 인사이트"""
        engine = MarketIntelligenceEngine()

        # Layer 3 데이터 추가
        engine.layer_data[DataLayer.LAYER_3_INDUSTRY] = LayerData(
            layer=3,
            layer_name="산업/기업",
            data={
                "ir_report": {
                    "year": "2025",
                    "quarter": "3Q",
                    "release_date": "2025-11-06",
                },
                "americas_insights": {
                    "regional_performance": [{"revenue_krw": 150.5, "revenue_yoy": 6.9}]
                },
            },
        )
        engine.source_manager.add_source = MagicMock(
            return_value=MagicMock(to_citation=MagicMock(return_value="[1]"))
        )

        insight = engine.generate_layered_insight()

        assert "Layer 3: 산업/기업 동향" in insight
        assert "아모레퍼시픽 Americas" in insight

    def test_generate_layered_insight_with_layer2(self):
        """Layer 2 데이터 포함 인사이트"""
        engine = MarketIntelligenceEngine()

        # Layer 2 데이터 추가
        engine.layer_data[DataLayer.LAYER_2_CONSUMER] = LayerData(
            layer=2,
            layer_name="소비자 트렌드",
            data={
                "kbeauty_news": [
                    {
                        "title": "K-Beauty trends dominate US market in 2026",
                        "source": "allure",
                        "published_at": "2026-01-15",
                        "url": "https://example.com",
                    }
                ],
                "reddit_trends": [
                    {
                        "title": "LANEIGE review",
                        "metadata": {"upvotes": 500},
                    }
                ],
            },
        )
        engine.source_manager.add_source = MagicMock(
            return_value=MagicMock(to_citation=MagicMock(return_value="[1]"))
        )

        insight = engine.generate_layered_insight()

        assert "Layer 2: 소비자 트렌드" in insight

    def test_generate_layered_insight_with_amazon_data(self):
        """Amazon 데이터 포함 인사이트"""
        engine = MarketIntelligenceEngine()

        amazon_data = {"laneige_rank": 3, "sos": 12.5}

        insight = engine.generate_layered_insight(amazon_data=amazon_data)

        assert "Layer 1: Amazon 성과" in insight
        assert "3위" in insight
        assert "12.5%" in insight

    def test_generate_layered_insight_includes_references(self):
        """참고자료 섹션 포함"""
        engine = MarketIntelligenceEngine()
        engine.source_manager.generate_references_section = MagicMock(
            return_value="## 참고자료\n[1] Test Source"
        )

        insight = engine.generate_layered_insight()

        assert "참고자료" in insight

    def test_generate_layer_summary_layer4(self):
        """Layer 4 요약 생성"""
        engine = MarketIntelligenceEngine()
        engine.layer_data[DataLayer.LAYER_4_MACRO] = LayerData(layer=4, layer_name="거시경제/무역")
        engine.public_collector.generate_insight_section = MagicMock(return_value="Layer 4 summary")

        summary = engine.generate_layer_summary(4)

        assert "Layer 4 summary" in summary

    def test_generate_layer_summary_layer3(self):
        """Layer 3 요약 생성"""
        engine = MarketIntelligenceEngine()
        engine.layer_data[DataLayer.LAYER_3_INDUSTRY] = LayerData(layer=3, layer_name="산업/기업")
        engine.ir_parser.generate_insight_section = MagicMock(return_value="Layer 3 summary")

        summary = engine.generate_layer_summary(3)

        assert "Layer 3 summary" in summary

    def test_generate_layer_summary_layer2(self):
        """Layer 2 요약 생성"""
        engine = MarketIntelligenceEngine()
        engine.layer_data[DataLayer.LAYER_2_CONSUMER] = LayerData(
            layer=2, layer_name="소비자 트렌드"
        )
        engine.signal_collector.generate_report_section = MagicMock(return_value="Layer 2 summary")

        summary = engine.generate_layer_summary(2)

        assert "Layer 2 summary" in summary

    def test_generate_layer_summary_no_data(self):
        """데이터 없을 때 요약 생성"""
        engine = MarketIntelligenceEngine()

        summary = engine.generate_layer_summary(4)

        assert "수집되지 않았습니다" in summary

    def test_generate_layer_summary_unknown_layer(self):
        """알 수 없는 레이어 요약"""
        engine = MarketIntelligenceEngine()
        engine.layer_data[99] = LayerData(layer=99, layer_name="Unknown")

        summary = engine.generate_layer_summary(99)

        assert "요약 생성 불가" in summary


class TestMarketIntelligenceEngineDataPersistence:
    """데이터 저장/로드 테스트"""

    def test_save_data(self, tmp_path):
        """데이터 저장 가능해야 함"""
        engine = MarketIntelligenceEngine(data_dir=str(tmp_path / "market_data"))

        # 테스트 데이터 추가
        engine.layer_data[DataLayer.LAYER_4_MACRO] = LayerData(
            layer=4,
            layer_name="거시경제/무역",
            data={"test": "value"},
            sources=[{"title": "Test"}],
        )

        engine.save_data()

        filepath = tmp_path / "market_data" / "layer_data.json"
        assert filepath.exists()

        # JSON 파일 내용 확인
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        assert "layers" in data
        assert "4" in data["layers"]
        assert data["layers"]["4"]["layer"] == 4

    def test_load_data(self, tmp_path):
        """데이터 로드 가능해야 함"""
        data_dir = tmp_path / "market_data"
        data_dir.mkdir()

        # 테스트 데이터 생성
        test_data = {
            "layers": {
                "4": {
                    "layer": 4,
                    "layer_name": "거시경제/무역",
                    "collected_at": "2026-01-15T10:00:00",
                    "data": {"test": "value"},
                    "sources": [{"title": "Test"}],
                }
            },
            "updated_at": "2026-01-15T10:00:00",
        }

        filepath = data_dir / "layer_data.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(test_data, f)

        # 로드 테스트
        engine = MarketIntelligenceEngine(data_dir=str(data_dir))
        engine.load_data()

        assert DataLayer.LAYER_4_MACRO in engine.layer_data
        assert engine.layer_data[DataLayer.LAYER_4_MACRO].layer == 4
        assert engine.layer_data[DataLayer.LAYER_4_MACRO].data == {"test": "value"}

    def test_load_data_file_not_exists(self, tmp_path):
        """파일이 없으면 조용히 무시"""
        engine = MarketIntelligenceEngine(data_dir=str(tmp_path / "market_data"))

        engine.load_data()  # 에러 없이 실행되어야 함

        assert len(engine.layer_data) == 0

    def test_load_data_invalid_json(self, tmp_path):
        """잘못된 JSON 파일 처리"""
        data_dir = tmp_path / "market_data"
        data_dir.mkdir()

        filepath = data_dir / "layer_data.json"
        with open(filepath, "w") as f:
            f.write("invalid json {")

        engine = MarketIntelligenceEngine(data_dir=str(data_dir))
        engine.load_data()  # 에러 없이 실행되어야 함

        assert len(engine.layer_data) == 0


class TestMarketIntelligenceEngineStats:
    """통계 조회 테스트"""

    def test_get_stats(self):
        """통계 반환 가능해야 함"""
        engine = MarketIntelligenceEngine()
        engine.public_collector.get_stats = MagicMock(return_value={"total": 10})
        engine.ir_parser.get_stats = MagicMock(return_value={"reports": 5})
        engine.signal_collector.get_stats = MagicMock(return_value={"signals": 20})
        engine.source_manager.get_stats = MagicMock(return_value={"sources": 15})

        # 레이어 데이터 추가
        engine.layer_data[DataLayer.LAYER_4_MACRO] = LayerData(layer=4, layer_name="거시경제/무역")

        stats = engine.get_stats()

        assert "layers_collected" in stats
        assert "public_data_stats" in stats
        assert "ir_stats" in stats
        assert "signal_stats" in stats
        assert "source_stats" in stats
        assert "initialized" in stats
        assert DataLayer.LAYER_4_MACRO in stats["layers_collected"]


class TestCreateMarketIntelligenceEngine:
    """편의 함수 테스트"""

    @pytest.mark.asyncio
    async def test_create_market_intelligence_engine(self):
        """엔진 생성 및 초기화 함수"""
        with patch.object(MarketIntelligenceEngine, "initialize", new_callable=AsyncMock):
            engine = await create_market_intelligence_engine(
                api_key="test-key"  # pragma: allowlist secret
            )

            assert engine is not None
            assert isinstance(engine, MarketIntelligenceEngine)

    @pytest.mark.asyncio
    async def test_create_market_intelligence_engine_without_key(self):
        """API 키 없이 엔진 생성"""
        with patch.object(MarketIntelligenceEngine, "initialize", new_callable=AsyncMock):
            engine = await create_market_intelligence_engine()

            assert engine is not None
