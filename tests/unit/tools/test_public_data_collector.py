"""
Tests for PublicDataCollector

공공데이터 API 수집기 테스트
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

from src.tools.public_data_collector import (
    PublicDataCollector,
    TradeData,
    CosmeticsProduct,
    DataSourceType,
    HS_CODE_COSMETICS,
    COUNTRY_CODES
)


class TestTradeData:
    """TradeData 데이터클래스 테스트"""

    def test_create_trade_data(self):
        """TradeData 생성 테스트"""
        trade = TradeData(
            data_id="TRADE-20260126-0001",
            hs_code="3304",
            trade_type="export",
            year="2025",
            month="12",
            country_code="US",
            country_name="미국",
            amount_usd=1_500_000_000,
            revenue_yoy=12.5
        )

        assert trade.data_id == "TRADE-20260126-0001"
        assert trade.hs_code == "3304"
        assert trade.trade_type == "export"
        assert trade.country_code == "US"
        assert trade.amount_usd == 1_500_000_000
        assert trade.reliability_score == 0.95  # 기본값

    def test_trade_data_to_dict(self):
        """TradeData to_dict 테스트"""
        trade = TradeData(
            data_id="TRADE-001",
            hs_code="3304",
            trade_type="export",
            year="2025",
            month="01"
        )

        data = trade.to_dict()

        assert isinstance(data, dict)
        assert data["data_id"] == "TRADE-001"
        assert data["hs_code"] == "3304"

    def test_trade_data_to_insight_format(self):
        """인사이트 포맷 변환 테스트"""
        trade = TradeData(
            data_id="TRADE-001",
            hs_code="3304",
            trade_type="export",
            year="2025",
            month="01",
            country_name="미국",
            amount_usd=1_500_000_000,  # $1.5B
            yoy_change=12.5
        )

        insight = trade.to_insight_format()

        assert "화장품" in insight
        assert "미국" in insight
        assert "$1.5B" in insight
        assert "+12.5%" in insight
        assert "관세청" in insight
        assert "2025.01" in insight


class TestCosmeticsProduct:
    """CosmeticsProduct 데이터클래스 테스트"""

    def test_create_cosmetics_product(self):
        """CosmeticsProduct 생성 테스트"""
        product = CosmeticsProduct(
            product_id="COSM-001",
            product_name="테스트 세럼",
            company_name="아모레퍼시픽",
            functional_type="주름개선",
            approval_date="20250101"
        )

        assert product.product_id == "COSM-001"
        assert product.product_name == "테스트 세럼"
        assert product.company_name == "아모레퍼시픽"
        assert product.reliability_score == 0.95


class TestPublicDataCollector:
    """PublicDataCollector 테스트"""

    @pytest.fixture
    def collector(self, tmp_path):
        """테스트용 collector 인스턴스"""
        return PublicDataCollector(
            api_key="test_api_key",
            data_dir=str(tmp_path / "public_data")
        )

    def test_collector_initialization(self, collector):
        """초기화 테스트"""
        assert collector.api_key == "test_api_key"
        assert collector.data_dir.exists()
        assert len(collector.trade_data) == 0
        assert len(collector.cosmetics_products) == 0

    def test_collector_no_api_key(self, tmp_path):
        """API 키 없이 초기화"""
        collector = PublicDataCollector(
            api_key=None,
            data_dir=str(tmp_path / "public_data")
        )

        assert collector.api_key == ""

    @pytest.mark.asyncio
    async def test_initialize(self, collector):
        """비동기 초기화 테스트"""
        await collector.initialize()

        # 세션이 생성되어야 함
        assert collector._session is not None

        await collector.close()

    def test_parse_customs_response_valid(self, collector):
        """관세청 응답 파싱 테스트 - 정상"""
        response = {
            "response": {
                "body": {
                    "items": {
                        "item": [
                            {
                                "natCd": "US",
                                "natEngNm": "United States",
                                "expDlr": "1500000000",
                                "expQty": "500000"
                            }
                        ]
                    }
                }
            }
        }

        items = collector._parse_customs_response(response)

        assert len(items) == 1
        assert items[0]["natCd"] == "US"
        assert items[0]["expDlr"] == "1500000000"

    def test_parse_customs_response_single_item(self, collector):
        """관세청 응답 파싱 테스트 - 단일 아이템"""
        response = {
            "response": {
                "body": {
                    "items": {
                        "item": {
                            "natCd": "CN",
                            "natEngNm": "China",
                            "expDlr": "2000000000"
                        }
                    }
                }
            }
        }

        items = collector._parse_customs_response(response)

        assert len(items) == 1
        assert items[0]["natCd"] == "CN"

    def test_parse_customs_response_empty(self, collector):
        """관세청 응답 파싱 테스트 - 빈 응답"""
        response = {"response": {"body": {"items": {}}}}

        items = collector._parse_customs_response(response)

        assert items == []

    def test_safe_float_valid(self, collector):
        """float 변환 테스트 - 정상"""
        assert collector._safe_float("123.45") == 123.45
        assert collector._safe_float("1000") == 1000.0
        assert collector._safe_float(500) == 500.0

    def test_safe_float_invalid(self, collector):
        """float 변환 테스트 - 비정상"""
        assert collector._safe_float(None) is None
        assert collector._safe_float("") is None
        assert collector._safe_float("not_a_number") is None

    def test_generate_data_id(self, collector):
        """데이터 ID 생성 테스트"""
        id1 = collector._generate_data_id("TRADE")
        id2 = collector._generate_data_id("COSM")

        assert id1.startswith("TRADE-")
        assert id2.startswith("COSM-")
        assert id1 != id2

    def test_get_trade_summary_empty(self, collector):
        """수출입 요약 테스트 - 빈 데이터"""
        summary = collector.get_trade_summary("2025", "export")

        assert "error" in summary

    def test_get_trade_summary_with_data(self, collector):
        """수출입 요약 테스트 - 데이터 있음"""
        # 테스트 데이터 추가
        collector.trade_data = [
            TradeData(
                data_id="T1",
                hs_code="3304",
                trade_type="export",
                year="2025",
                month="01",
                country_code="US",
                country_name="미국",
                amount_usd=1_000_000_000
            ),
            TradeData(
                data_id="T2",
                hs_code="3304",
                trade_type="export",
                year="2025",
                month="01",
                country_code="CN",
                country_name="중국",
                amount_usd=500_000_000
            )
        ]

        summary = collector.get_trade_summary("2025", "export")

        assert summary["year"] == "2025"
        assert summary["trade_type"] == "export"
        assert summary["total_amount_usd"] == 1_500_000_000
        assert summary["total_records"] == 2
        assert len(summary["top_countries"]) == 2

    def test_generate_insight_section_empty(self, collector):
        """인사이트 섹션 생성 - 빈 데이터"""
        result = collector.generate_insight_section()

        assert "수집된 공공데이터가 없습니다" in result

    def test_generate_insight_section_with_data(self, collector):
        """인사이트 섹션 생성 - 데이터 있음"""
        collector.trade_data = [
            TradeData(
                data_id="T1",
                hs_code="3304",
                trade_type="export",
                year="2025",
                month="01",
                country_name="미국",
                amount_usd=1_500_000_000,
                yoy_change=12.5
            )
        ]

        result = collector.generate_insight_section()

        assert "거시경제/무역 데이터" in result
        assert "미국" in result

    def test_create_source_reference_trade(self, collector):
        """출처 참조 생성 - 수출입 데이터"""
        trade = TradeData(
            data_id="T1",
            hs_code="3304",
            trade_type="export",
            year="2025",
            month="01",
            source_url="https://apis.data.go.kr/..."
        )

        ref = collector.create_source_reference(trade_data=trade)

        assert ref["publisher"] == "관세청"
        assert ref["source_type"] == "government"
        assert ref["reliability_score"] == 0.95

    def test_get_stats(self, collector):
        """통계 반환 테스트"""
        stats = collector.get_stats()

        assert "trade_records" in stats
        assert "cosmetics_products" in stats
        assert "api_key_configured" in stats
        assert stats["api_key_configured"] is True


class TestCountryCodes:
    """국가 코드 상수 테스트"""

    def test_us_country_code(self):
        """미국 코드 테스트"""
        assert "US" in COUNTRY_CODES
        assert COUNTRY_CODES["US"]["name"] == "미국"

    def test_hs_code_cosmetics(self):
        """화장품 HS 코드 테스트"""
        assert HS_CODE_COSMETICS == "3304"


class TestDataSourceType:
    """DataSourceType Enum 테스트"""

    def test_data_source_types(self):
        """데이터 소스 유형 테스트"""
        assert DataSourceType.CUSTOMS_TRADE.value == "customs_trade"
        assert DataSourceType.MFDS_COSMETICS.value == "mfds_cosmetics"
        assert DataSourceType.KOSIS_STATS.value == "kosis_stats"
