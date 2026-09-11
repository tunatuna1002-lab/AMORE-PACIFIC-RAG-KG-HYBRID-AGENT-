"""
Phase 6 위생 테스트 (§6.1 config 단일화 / §6.2 로더 통일 / §6.3 기본값 / §6.4~6.6)
"""

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
THRESHOLDS = json.loads((REPO_ROOT / "config" / "thresholds.json").read_text(encoding="utf-8"))


# =============================================================================
# §6.1 죽은 config 키 제거 + 배지 임계값 단일화
# =============================================================================


class TestConfigKeys:
    DEAD_KEYS = [
        "eviction_policy",
        "max_chunks",
        "rerank_enabled",
        "crawl_hour_utc",
        "new_product_watch_days",
        "gap_alert",
        "laneige_market_share_warning",
        "hhi_concentrated",
        "cpi_premium",
        "rating_gap_warning",
    ]

    def _all_keys(self, node, out=None):
        out = out if out is not None else set()
        if isinstance(node, dict):
            for k, v in node.items():
                out.add(k)
                self._all_keys(v, out)
        return out

    def test_dead_keys_removed(self):
        keys = self._all_keys(THRESHOLDS)
        remaining = [k for k in self.DEAD_KEYS if k in keys]
        assert not remaining, f"리더 0건 config 키가 남아있습니다: {remaining}"

    def test_badge_thresholds_block_exists(self):
        badges = THRESHOLDS.get("dashboard_badges")
        assert badges, "dashboard_badges 블록 누락"
        for key in (
            "sos_leader",
            "sos_strong",
            "rank_near_top",
            "rank_top10",
            "cpi_score_strong",
            "cpi_score_ok",
            "brand_count_high",
            "brand_count_mid",
        ):
            assert key in badges, f"{key} 누락"

    def test_api_serves_badge_thresholds(self):
        from src.api.routes.data import load_badge_thresholds

        served = load_badge_thresholds()
        assert served["sos_leader"] == THRESHOLDS["dashboard_badges"]["sos_leader"]

    def test_dashboard_reads_served_thresholds(self):
        html = (REPO_ROOT / "dashboard" / "amore_unified_dashboard_v4.html").read_text(
            encoding="utf-8"
        )
        assert "badge_thresholds" in html
        assert "const T = badgeThresholds();" in html
        # 매직넘버 재구현 회귀 방지
        assert "if (sos >= 15) {" not in html
        assert "if (cpiScore >= 90) {" not in html


class TestAlertThresholdsFromConfig:
    """§6.1: alert_manager 상수와 config 값이 갈라지지 않는다"""

    def test_manager_reads_config(self):
        from src.core.alert_manager import AlertManager

        manager = AlertManager()
        assert manager.sos_change_threshold == abs(THRESHOLDS["brand_health"]["sos_change_up"])
        assert manager.rank_change_threshold == THRESHOLDS["ranking"]["significant_rise"]

    def test_falls_back_when_config_missing(self, monkeypatch):
        from src.core import alert_manager as am

        monkeypatch.setattr(am.AlertManager, "_load_thresholds", staticmethod(lambda: (10, 2.0)))
        manager = am.AlertManager()
        assert manager.rank_change_threshold == 10
        assert manager.sos_change_threshold == 2.0


# =============================================================================
# §6.2 config 로더 이중화 해소
# =============================================================================


class TestSingleThresholdGetter:
    def test_app_config_getter_removed(self):
        from src.infrastructure.config.config_manager import AppConfig

        assert not hasattr(AppConfig, "get_threshold")

    def test_rules_engine_getter_kept(self):
        from src.core.rules_engine import RulesEngine

        assert hasattr(RulesEngine, "get_threshold")


# =============================================================================
# §6.3 retrieval_weights 코드 기본값
# =============================================================================


class TestRetrievalWeightDefaults:
    def test_code_default_matches_config_file(self):
        file_cfg = json.loads(
            (REPO_ROOT / "config" / "retrieval_weights.json").read_text(encoding="utf-8")
        )
        src = (SRC_ROOT / "rag" / "hybrid_retriever.py").read_text(encoding="utf-8")
        expected = file_cfg["max_context_items"]["rag_chunks"]
        assert f'"rag_chunks": {expected}' in src
        assert '"rag_chunks": 3' not in src, "사이클 2 버그 값(3)이 코드 기본값에 남아있습니다"

    def test_deep_merge_keeps_unspecified_defaults(self, tmp_path, monkeypatch):
        """설정 파일이 일부 키만 담아도 나머지 기본값이 살아남는다"""
        from src.rag.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever.__new__(HybridRetriever)
        weights = retriever._load_retrieval_weights()
        items = weights["max_context_items"]
        # 파일에 없는 키까지 전부 존재해야 한다
        assert {"ontology_facts", "inferences", "rag_chunks"} <= set(items)
        assert items["rag_chunks"] == 8


# =============================================================================
# §6.4 죽은 코드 삭제
# =============================================================================


class TestDeadCodeRemoved:
    def test_context_bonus_removed(self):
        from src.core.confidence import ConfidenceAssessor

        assert not hasattr(ConfidenceAssessor, "_context_bonus")

    def test_dead_api_router_removed(self):
        import src.api

        assert not hasattr(src.api, "api_router")
        assert not hasattr(src.api, "include_routers")

    def test_competitors_json_removed(self):
        assert not (REPO_ROOT / "config" / "competitors.json").exists()

    def test_v3_alert_routes_marked_deprecated(self):
        """§6.4 vs §6.7 충돌: 삭제 대신 deprecated 표시 (FUTURE_WORK 기록)"""
        src = (SRC_ROOT / "api" / "routes" / "alerts.py").read_text(encoding="utf-8")
        assert src.count("deprecated=True") == 4


# =============================================================================
# §6.5 상수 필드 제거
# =============================================================================


class TestNoConstantMetricFields:
    def test_exporter_emits_none(self):
        src = (SRC_ROOT / "tools" / "exporters" / "dashboard_exporter.py").read_text(
            encoding="utf-8"
        )
        assert '"streak_days": 7' not in src
        assert '"rating_gap": 0.1' not in src
        assert '"streak_days": None' in src

    @pytest.mark.parametrize(
        "condition_factory,key",
        [
            ("streak_days_above", "streak_days"),
            ("rating_gap_negative", "rating_gap"),
            ("rating_gap_positive", "rating_gap"),
            ("rank_improving", "rank_change_7d"),
            ("rank_declining", "rank_change_7d"),
            # 사이클 10: SoS·HHI·CPI 조건이 이 규칙에서 빠져 있었다. HHI 결측이 0으로
            # 읽혀 "분산 시장" 규칙이 발화했고 답변에 "HHI: 0.000"이 나왔다.
            ("sos_above", "sos"),
            ("sos_below", "sos"),
            ("hhi_above", "hhi"),
            ("hhi_below", "hhi"),
            ("cpi_above", "cpi"),
            ("cpi_below", "cpi"),
        ],
    )
    def test_missing_value_does_not_trigger_rule(self, condition_factory, key):
        """결측(None)은 0으로 취급되지 않고 조건이 False가 된다"""
        from src.ontology.reasoner import StandardConditions

        threshold_args = {
            "streak_days_above": 30,
            "sos_above": 0.15,
            "sos_below": 0.05,
            "hhi_above": 0.25,
            "hhi_below": 0.15,
            "cpi_above": 90,
            "cpi_below": 110,
        }
        factory = getattr(StandardConditions, condition_factory)
        condition = (
            factory(threshold_args[condition_factory])
            if condition_factory in threshold_args
            else factory()
        )

        assert condition.check({}) is False, "키 없음인데 규칙이 발화했습니다"
        assert condition.check({key: None}) is False, "None인데 규칙이 발화했습니다"

    def test_real_value_still_triggers(self):
        from src.ontology.reasoner import StandardConditions

        assert StandardConditions.streak_days_above(30).check({"streak_days": 45}) is True
        assert StandardConditions.rating_gap_positive().check({"rating_gap": 0.2}) is True
        assert StandardConditions.rank_improving().check({"rank_change_7d": -3}) is True


# =============================================================================
# §6.6 예외 침묵 구체화
# =============================================================================


class TestExceptionMessages:
    def test_no_generic_suppressed_exception(self):
        offenders = []
        for path in SRC_ROOT.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if "Suppressed Exception" in text:
                offenders.append(str(path.relative_to(REPO_ROOT)))
        assert not offenders, f"동일 문구 예외 로그가 남아있습니다: {offenders}"

    def test_intent_classification_failure_is_logged(self):
        src = (SRC_ROOT / "rag" / "hybrid_retriever.py").read_text(encoding="utf-8")
        assert "인텐트 분류 실패" in src
        assert (
            "fusion_strategy_name = config.fusion_strategy\n        except Exception:\n            pass"
            not in src
        )
