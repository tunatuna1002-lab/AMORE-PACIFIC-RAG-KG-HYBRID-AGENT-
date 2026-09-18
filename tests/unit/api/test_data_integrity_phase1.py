"""
Phase 1 데이터 정합성 회귀 테스트 (§1.3 SoS 분모 / §1.4 가짜 지표 / §1.5 거짓 신선도)
"""

from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.dependencies import compute_freshness
from src.api.routes import data as data_routes
from src.api.routes import market_intelligence as mi_routes
from src.tools.calculators.metric_calculator import SOS_MIN_SAMPLE, calculate_sos_pct

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"


# =============================================================================
# §1.3 SoS 실분모
# =============================================================================


class TestCalculateSosPct:
    def test_uses_real_denominator(self):
        # 60개만 수집된 부분 수집일: 바닥 100을 쓰면 3.0%로 과소 계산됐다
        assert calculate_sos_pct(3, 60) == 5.0

    def test_full_sample(self):
        assert calculate_sos_pct(5, 100) == 5.0

    def test_below_min_sample_returns_none(self):
        assert calculate_sos_pct(3, 20) is None

    def test_boundary_at_min_sample(self):
        assert calculate_sos_pct(1, SOS_MIN_SAMPLE) is not None
        assert calculate_sos_pct(1, SOS_MIN_SAMPLE - 1) is None

    def test_zero_total(self):
        assert calculate_sos_pct(0, 0) is None

    def test_no_floor_left_in_data_routes(self):
        """`max(total, 100)` 분모 바닥이 data.py에 재등장하지 않는다"""
        text = (SRC_ROOT / "api" / "routes" / "data.py").read_text(encoding="utf-8")
        offenders = [
            f"{i}: {line.strip()}"
            for i, line in enumerate(text.splitlines(), 1)
            if "max(total" in line and "100)" in line
        ]
        assert not offenders, "SoS 분모 바닥이 되살아났습니다:\n" + "\n".join(offenders)


class TestSqliteFallbackSos:
    """부분 수집 시나리오: SQLite 폴백 응답"""

    @staticmethod
    def _records(n_total: int, n_laneige: int, snapshot_date: str):
        records = [
            {
                "brand": "LANEIGE",
                "product_name": "Lip Sleeping Mask",
                "asin": f"L{i}",
                "rank": i + 1,
                "snapshot_date": snapshot_date,
            }
            for i in range(n_laneige)
        ]
        records += [
            {
                "brand": f"Other{i}",
                "product_name": f"P{i}",
                "asin": f"O{i}",
                "rank": n_laneige + i + 1,
                "snapshot_date": snapshot_date,
            }
            for i in range(n_total - n_laneige)
        ]
        return records

    async def _run(self, records):
        storage = MagicMock()
        storage.initialize = AsyncMock()
        storage.get_latest_data = AsyncMock(return_value=records)
        with patch.object(data_routes, "get_sqlite_storage", return_value=storage):
            return await data_routes._generate_dashboard_from_sqlite()

    @pytest.mark.asyncio
    async def test_partial_collection_uses_real_denominator(self):
        today = datetime.now().strftime("%Y-%m-%d")
        result = await self._run(self._records(60, 3, today))
        # 3/60 = 5.0% (바닥 100이면 3.0%)
        assert result["brand"]["kpis"]["sos"] == 5.0
        assert result["metadata"]["insufficient_sample"] is False

    @pytest.mark.asyncio
    async def test_below_min_sample_flags_and_nulls(self):
        today = datetime.now().strftime("%Y-%m-%d")
        result = await self._run(self._records(20, 2, today))
        assert result["brand"]["kpis"]["sos"] is None
        assert result["metadata"]["insufficient_sample"] is True

    @pytest.mark.asyncio
    async def test_freshness_is_computed_not_hardcoded(self):
        """§1.5: 3일 전 스냅샷은 stale로 표시된다"""
        old_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        result = await self._run(self._records(100, 5, old_date))
        assert result["metadata"]["_is_stale"] is True
        assert result["metadata"]["_cache_age_hours"] > 24

    @pytest.mark.asyncio
    async def test_fresh_snapshot_not_stale(self):
        today = datetime.now().strftime("%Y-%m-%d")
        result = await self._run(self._records(100, 5, today))
        assert result["metadata"]["_is_stale"] is False


# =============================================================================
# §1.5 신선도 헬퍼
# =============================================================================


class TestComputeFreshness:
    def test_today_is_fresh(self):
        age, stale = compute_freshness(datetime.now().strftime("%Y-%m-%d"))
        assert stale is False
        assert age is not None and age < 24

    def test_three_days_ago_is_stale(self):
        old = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        age, stale = compute_freshness(old)
        assert stale is True
        assert age > 24

    def test_missing_date_is_stale(self):
        assert compute_freshness(None) == (None, True)
        assert compute_freshness("") == (None, True)

    def test_malformed_date_is_stale(self):
        assert compute_freshness("not-a-date") == (None, True)


# =============================================================================
# §1.4 market_intelligence 가짜 지표
# =============================================================================


class TestMarketIntelligenceLayer1:
    def test_no_placeholder_literals_in_source(self):
        """placeholder 상수(sos 5.2 / rank 15) 회귀 방지"""
        text = (SRC_ROOT / "api" / "routes" / "market_intelligence.py").read_text(encoding="utf-8")
        assert "placeholder" not in text.lower()
        assert '"sos": 5.2' not in text
        assert '"laneige_rank": 15' not in text

    @pytest.mark.asyncio
    async def test_returns_real_values(self):
        records = [{"brand": "LANEIGE", "rank": 7, "snapshot_date": "2026-08-31"}] + [
            {"brand": f"B{i}", "rank": i + 8, "snapshot_date": "2026-08-31"} for i in range(99)
        ]
        storage = MagicMock()
        storage.initialize = AsyncMock()
        storage.get_latest_data = AsyncMock(return_value=records)
        with patch.object(mi_routes, "get_sqlite_storage", return_value=storage):
            result = await mi_routes._fetch_amazon_layer1()
        assert result["laneige_rank"] == 7
        assert result["sos"] == 1.0
        assert result["total_products"] == 100

    @pytest.mark.asyncio
    async def test_returns_none_when_no_data(self):
        storage = MagicMock()
        storage.initialize = AsyncMock()
        storage.get_latest_data = AsyncMock(return_value=[])
        with patch.object(mi_routes, "get_sqlite_storage", return_value=storage):
            assert await mi_routes._fetch_amazon_layer1() is None

    @pytest.mark.asyncio
    async def test_returns_none_when_brand_absent(self):
        storage = MagicMock()
        storage.initialize = AsyncMock()
        storage.get_latest_data = AsyncMock(
            return_value=[{"brand": "COSRX", "rank": 1, "snapshot_date": "2026-08-31"}]
        )
        with patch.object(mi_routes, "get_sqlite_storage", return_value=storage):
            assert await mi_routes._fetch_amazon_layer1() is None

    @pytest.mark.asyncio
    async def test_returns_none_on_storage_error(self):
        storage = MagicMock()
        storage.initialize = AsyncMock(side_effect=RuntimeError("db down"))
        with patch.object(mi_routes, "get_sqlite_storage", return_value=storage):
            assert await mi_routes._fetch_amazon_layer1() is None


# =============================================================================
# §1.6 가짜 성장 목표
# =============================================================================


class TestNoFakeGrowthTarget:
    def test_target_sos_removed(self):
        text = (SRC_ROOT / "agents" / "period_insight_agent.py").read_text(encoding="utf-8")
        assert '"target_sos"' not in text
        assert "* 1.1" not in text
