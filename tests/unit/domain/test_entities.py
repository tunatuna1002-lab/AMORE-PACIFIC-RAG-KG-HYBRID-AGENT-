"""
Domain Entities Tests (TDD - RED Phase)
========================================
테스트를 먼저 작성하여 src/domain/entities의 구조를 정의합니다.

이 테스트는 새로운 domain layer에서 import할 때 실패합니다 (RED).
구현 후 통과해야 합니다 (GREEN).
"""

from datetime import date

import pytest


class TestRankRecord:
    """RankRecord Entity 테스트"""

    def test_rank_record_creation(self):
        """RankRecord가 필수 필드로 올바르게 생성되는지 검증"""
        # 새 domain layer에서 import (현재는 실패해야 함)
        from src.domain.entities.product import RankRecord

        record = RankRecord(
            snapshot_date=date(2026, 1, 22),
            category_id="lip_care",
            asin="B08XYZ1234",
            product_name="LANEIGE Lip Sleeping Mask - Berry",
            brand="LANEIGE",
            rank=1,
            product_url="https://www.amazon.com/dp/B08XYZ1234",
        )

        assert record.asin == "B08XYZ1234"
        assert record.brand == "LANEIGE"
        assert record.rank == 1
        assert record.category_id == "lip_care"

    def test_rank_record_with_optional_fields(self):
        """RankRecord의 선택적 필드가 올바르게 처리되는지 검증"""
        from src.domain.entities.product import RankRecord

        record = RankRecord(
            snapshot_date=date(2026, 1, 22),
            category_id="lip_care",
            asin="B08XYZ1234",
            product_name="LANEIGE Lip Sleeping Mask",
            brand="LANEIGE",
            rank=1,
            price=24.00,
            list_price=30.00,
            discount_percent=20.0,
            rating=4.7,
            reviews_count=89234,
            badge="Best Seller",
            product_url="https://www.amazon.com/dp/B08XYZ1234",
        )

        assert record.price == 24.00
        assert record.discount_percent == 20.0
        assert record.rating == 4.7
        assert record.badge == "Best Seller"

    def test_rank_record_rank_validation(self):
        """RankRecord의 rank 범위 검증 (1-100)"""
        from src.domain.entities.product import RankRecord

        # rank가 0이면 에러
        with pytest.raises(ValueError):
            RankRecord(
                snapshot_date=date(2026, 1, 22),
                category_id="lip_care",
                asin="B08XYZ1234",
                product_name="Test",
                brand="Test",
                rank=0,  # Invalid
                product_url="https://www.amazon.com/dp/B08XYZ1234",
            )

        # rank가 101이면 에러
        with pytest.raises(ValueError):
            RankRecord(
                snapshot_date=date(2026, 1, 22),
                category_id="lip_care",
                asin="B08XYZ1234",
                product_name="Test",
                brand="Test",
                rank=101,  # Invalid
                product_url="https://www.amazon.com/dp/B08XYZ1234",
            )


class TestProduct:
    """Product Entity 테스트"""

    def test_product_creation(self):
        """Product가 올바르게 생성되는지 검증"""
        from src.domain.entities.product import Product

        product = Product(
            asin="B08XYZ1234",
            product_name="LANEIGE Lip Sleeping Mask - Berry",
            brand="LANEIGE",
            product_url="https://www.amazon.com/dp/B08XYZ1234",
        )

        assert product.asin == "B08XYZ1234"
        assert product.brand == "LANEIGE"
        assert "Lip Sleeping Mask" in product.product_name

    def test_product_with_dates(self):
        """Product의 날짜 필드가 올바르게 처리되는지 검증"""
        from src.domain.entities.product import Product

        product = Product(
            asin="B08XYZ1234",
            product_name="LANEIGE Lip Sleeping Mask",
            brand="LANEIGE",
            product_url="https://www.amazon.com/dp/B08XYZ1234",
            first_seen_date=date(2025, 1, 1),
            launch_date=date(2024, 11, 15),
        )

        assert product.first_seen_date == date(2025, 1, 1)
        assert product.launch_date == date(2024, 11, 15)


class TestBrand:
    """Brand Entity 테스트"""

    def test_brand_creation(self):
        """Brand가 올바르게 생성되는지 검증"""
        from src.domain.entities.brand import Brand

        brand = Brand(name="LANEIGE", is_target=True)

        assert brand.name == "LANEIGE"
        assert brand.is_target is True

    def test_brand_default_is_target(self):
        """Brand의 기본 is_target 값이 False인지 검증"""
        from src.domain.entities.brand import Brand

        brand = Brand(name="Competitor")

        assert brand.is_target is False


class TestBrandMetrics:
    """BrandMetrics Entity 테스트"""

    def test_brand_metrics_creation(self):
        """BrandMetrics가 올바르게 생성되는지 검증"""
        from src.domain.entities.brand import BrandMetrics

        metrics = BrandMetrics(
            brand="LANEIGE", category_id="lip_care", sos=15.5, brand_avg_rank=5.2, product_count=3
        )

        assert metrics.brand == "LANEIGE"
        assert metrics.sos == 15.5
        assert metrics.brand_avg_rank == 5.2
        assert metrics.product_count == 3

    def test_brand_metrics_with_cpi(self):
        """BrandMetrics의 CPI와 rating_gap 필드 검증"""
        from src.domain.entities.brand import BrandMetrics

        metrics = BrandMetrics(
            brand="LANEIGE", category_id="lip_care", sos=15.5, cpi=105.2, avg_rating_gap=0.3
        )

        assert metrics.cpi == 105.2
        assert metrics.avg_rating_gap == 0.3


class TestMarketMetrics:
    """MarketMetrics Entity 테스트"""

    def test_market_metrics_creation(self):
        """MarketMetrics가 올바르게 생성되는지 검증"""
        from src.domain.entities.market import MarketMetrics

        metrics = MarketMetrics(category_id="lip_care", snapshot_date=date(2026, 1, 22), hhi=0.08)

        assert metrics.category_id == "lip_care"
        assert metrics.hhi == 0.08

    def test_market_metrics_with_optional_fields(self):
        """MarketMetrics의 선택적 필드 검증"""
        from src.domain.entities.market import MarketMetrics

        metrics = MarketMetrics(
            category_id="lip_care",
            snapshot_date=date(2026, 1, 22),
            hhi=0.08,
            churn_rate=12.5,
            category_avg_price=25.99,
            category_avg_rating=4.3,
        )

        assert metrics.churn_rate == 12.5
        assert metrics.category_avg_price == 25.99
        assert metrics.category_avg_rating == 4.3


class TestCategory:
    """Category Entity 테스트"""

    def test_category_creation(self):
        """Category가 올바르게 생성되는지 검증"""
        from src.domain.entities.market import Category

        category = Category(
            id="lip_care",
            name="Lip Care",
            url="https://www.amazon.com/Best-Sellers-Beauty/zgbs/beauty/3761351",
        )

        assert category.id == "lip_care"
        assert category.name == "Lip Care"

    def test_category_hierarchy(self):
        """Category의 계층 구조 필드 검증"""
        from src.domain.entities.market import Category

        category = Category(
            id="lip_care",
            name="Lip Care",
            url="https://www.amazon.com/zgbs/beauty/3761351",
            parent_id="beauty",
            level=2,
            path=["beauty", "skin_care", "lip_care"],
            children=[],
        )

        assert category.parent_id == "beauty"
        assert category.level == 2
        assert len(category.path) == 3


class TestSnapshot:
    """Snapshot Entity 테스트"""

    def test_snapshot_creation(self):
        """Snapshot이 올바르게 생성되는지 검증"""
        from src.domain.entities.market import Snapshot

        snapshot = Snapshot(
            snapshot_date=date(2026, 1, 22),
            category_id="lip_care",
            total_products=100,
            success=True,
        )

        assert snapshot.category_id == "lip_care"
        assert snapshot.total_products == 100
        assert snapshot.success is True

    def test_snapshot_with_error(self):
        """실패한 Snapshot 생성 검증"""
        from src.domain.entities.market import Snapshot

        snapshot = Snapshot(
            snapshot_date=date(2026, 1, 22),
            category_id="lip_care",
            total_products=0,
            success=False,
            error_message="Connection timeout",
        )

        assert snapshot.success is False
        assert snapshot.error_message == "Connection timeout"


class TestBadgeType:
    """BadgeType Enum 테스트"""

    def test_badge_types(self):
        """BadgeType Enum 값 검증"""
        from src.domain.entities.product import BadgeType

        assert BadgeType.BEST_SELLER.value == "Best Seller"
        assert BadgeType.AMAZONS_CHOICE.value == "Amazon's Choice"
        assert BadgeType.CLIMATE_PLEDGE.value == "Climate Pledge Friendly"
        assert BadgeType.NONE.value == ""
