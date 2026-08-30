"""
Phase 2 대시보드 KPI 델타 테스트 (§2.4)

Brand View의 Top10/평균순위 배지와 HHI 밴드가 실계산 값으로 채워지는지 검증한다.
(과거에는 "+1 증가", "▼ 2.3위 개선", "중간 집중도"가 HTML에 고정돼 있었다.)
"""

from pathlib import Path

import pytest

from src.tools.exporters.dashboard_exporter import DashboardExporter

DASHBOARD_HTML = (
    Path(__file__).resolve().parents[3] / "dashboard" / "amore_unified_dashboard_v4.html"
)


@pytest.fixture
def exporter():
    return DashboardExporter.__new__(DashboardExporter)


def _records(date: str, laneige_ranks: list[int], other_count: int = 10):
    rows = [
        {"brand": "LANEIGE", "product_name": "Lip Sleeping Mask", "rank": r, "snapshot_date": date}
        for r in laneige_ranks
    ]
    rows += [
        {"brand": f"Other{i}", "product_name": f"P{i}", "rank": 50 + i, "snapshot_date": date}
        for i in range(other_count)
    ]
    return rows


class TestTop10Delta:
    def test_increase(self, exporter):
        raw = _records("2026-08-30", [5, 40]) + _records("2026-08-31", [3, 8, 40])
        assert exporter._calculate_top10_delta(raw, "2026-08-31") == "+1개"

    def test_decrease(self, exporter):
        raw = _records("2026-08-30", [3, 8]) + _records("2026-08-31", [8])
        assert exporter._calculate_top10_delta(raw, "2026-08-31") == "-1개"

    def test_no_change(self, exporter):
        raw = _records("2026-08-30", [3]) + _records("2026-08-31", [4])
        assert exporter._calculate_top10_delta(raw, "2026-08-31") == "+0개"

    def test_none_without_prior_snapshot(self, exporter):
        raw = _records("2026-08-31", [3])
        assert exporter._calculate_top10_delta(raw, "2026-08-31") is None


class TestAvgRankDelta:
    def test_improvement_is_negative(self, exporter):
        """순위가 낮아지면(상승하면) 음수 델타"""
        raw = _records("2026-08-30", [20, 30]) + _records("2026-08-31", [10, 20])
        assert exporter._calculate_avg_rank_delta(raw, "2026-08-31") == "-10.0위"

    def test_decline_is_positive(self, exporter):
        raw = _records("2026-08-30", [10]) + _records("2026-08-31", [15])
        assert exporter._calculate_avg_rank_delta(raw, "2026-08-31") == "+5.0위"

    def test_no_change(self, exporter):
        raw = _records("2026-08-30", [10]) + _records("2026-08-31", [10])
        assert exporter._calculate_avg_rank_delta(raw, "2026-08-31") == "0.0위"

    def test_none_without_prior_snapshot(self, exporter):
        assert exporter._calculate_avg_rank_delta(_records("2026-08-31", [3]), "2026-08-31") is None


class TestHhiBand:
    @pytest.mark.parametrize(
        "hhi,expected",
        [
            (0.0, "분산 시장"),
            (0.1499, "분산 시장"),
            (0.15, "중간 집중도"),
            (0.2499, "중간 집중도"),
            (0.25, "고집중 시장"),
            (1.0, "고집중 시장"),
        ],
    )
    def test_bands(self, hhi, expected):
        assert DashboardExporter._hhi_band(hhi) == expected


class TestBrandKpiPayload:
    def test_contains_delta_keys(self, exporter):
        raw = _records("2026-08-30", [5, 40]) + _records("2026-08-31", [3, 8, 40])
        kpis = exporter._generate_brand_data(raw)["kpis"]
        for key in ("sos_delta", "top10_delta", "avg_rank_delta", "hhi", "hhi_band"):
            assert key in kpis, f"{key} 누락"
        assert 0.0 <= kpis["hhi"] <= 1.0


class TestDashboardHtmlNoHardcodedBadges:
    """HTML에 정적 배지 문구가 되살아나지 않는다"""

    FORBIDDEN = [
        ">+1 증가<",
        ">▼ 2.3위 개선<",
        '<span class="kpi-delta">중간 집중도</span>',
        ">18.5<",
        ">Category Leader<",
        ">1위 근접<",
        ">가격 경쟁력 하락<",
        ">경쟁 심화<",
        "lip-mask-berry",
        "https://api.frankfurter.app",
        "loadAlertSettings();",
    ]

    def test_no_seed_values(self):
        html = DASHBOARD_HTML.read_text(encoding="utf-8")
        found = [needle for needle in self.FORBIDDEN if needle in html]
        assert not found, f"하드코딩 시드/외부호출이 되살아났습니다: {found}"

    def test_badge_ids_present(self):
        html = DASHBOARD_HTML.read_text(encoding="utf-8")
        for badge_id in (
            "brand-sos-delta",
            "brand-top10-delta",
            "brand-rank-delta",
            "brand-hhi-band",
        ):
            assert f'id="{badge_id}"' in html, f"{badge_id} 배지 id 누락"

    def test_api_base_uses_origin(self):
        html = DASHBOARD_HTML.read_text(encoding="utf-8")
        assert "const API_BASE = SERVER_URL;" in html
        assert "window.location.origin" in html

    def test_switch_page_binds_category_and_product(self):
        html = DASHBOARD_HTML.read_text(encoding="utf-8")
        assert "if (pageId === 'category') {" in html
        assert "updateProductList();" in html
