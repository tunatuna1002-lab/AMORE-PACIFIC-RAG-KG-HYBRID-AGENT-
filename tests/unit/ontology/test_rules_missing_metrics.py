"""결측 지표가 추론 규칙을 발화시키지 않는다 (사이클 10, 2026-09-12)

배경: 평가 실행에서 챗봇 답변 여러 건이 "시장이 분산되어 있고(HHI: 0.000)"와
"존재감이 낮아(0.0%)"를 단정했다. HHI·SoS가 컨텍스트에 없었는데
StandardConditions.hhi_below/sos_below가 결측을 0으로 읽어 조건을 통과시켰고,
결론 템플릿이 그 0을 출력했다. 데이터가 없다는 사실이 "값이 0"이라는 주장으로
바뀌어 사용자에게 전달된 것이다.
(docs/experiments/eval_cycle10_2026-09-12.md §2-d)
"""

from src.ontology.rules.growth_rules import RULE_CATEGORY_OPPORTUNITY
from src.ontology.rules.market_rules import RULE_FRAGMENTED_COMPETITION
from src.ontology.rules.price_rules import RULE_BESTSELLER_BADGE_EFFECT, RULE_PREMIUM_POSITION
from src.rag.hybrid_retriever import HybridRetriever

COMPETITORS = [{"brand": f"brand_{i}"} for i in range(13)]


class TestFabricatedMetricsAreNotEmitted:
    def test_fragmented_competition_needs_a_real_hhi(self):
        context = {"competitor_count": 13, "competitors": COMPETITORS}

        assert RULE_FRAGMENTED_COMPETITION.apply(context) is None

    def test_fragmented_competition_reports_the_real_hhi(self):
        context = {"hhi": 0.0681, "competitor_count": 13, "competitors": COMPETITORS}

        result = RULE_FRAGMENTED_COMPETITION.apply(context)

        assert result is not None
        assert "0.068" in result.insight
        assert "0.000" not in result.insight

    def test_category_opportunity_needs_real_hhi_and_sos(self):
        context = {"is_target": True, "brand": "laneige", "category": "lip_care"}

        assert RULE_CATEGORY_OPPORTUNITY.apply(context) is None
        assert RULE_CATEGORY_OPPORTUNITY.apply({**context, "hhi": 0.0681}) is None

    def test_category_opportunity_fires_on_real_values(self):
        context = {
            "is_target": True,
            "brand": "laneige",
            "category": "skin_care",
            "hhi": 0.067,
            "sos": 0.0,  # 실제로 관측된 0%는 결측과 다르다 — 발화해야 한다
        }

        result = RULE_CATEGORY_OPPORTUNITY.apply(context)

        assert result is not None
        assert "0.067" in result.insight

    def test_premium_position_needs_a_real_rating_gap(self):
        assert RULE_PREMIUM_POSITION.apply({"cpi": 170}) is None
        assert RULE_PREMIUM_POSITION.apply({"cpi": 170, "rating_gap": 0.1}) is not None

    def test_badge_stability_needs_a_real_rank_change(self):
        assert RULE_BESTSELLER_BADGE_EFFECT.apply({"badge": "Best Seller"}) is None


class TestInferenceContextDoesNotInjectDefaults:
    @staticmethod
    def _retriever() -> HybridRetriever:
        # 브랜드 엔티티가 없으면 KG에 접근하지 않으므로 초기화 없이 쓸 수 있다
        return HybridRetriever.__new__(HybridRetriever)

    def test_null_market_fields_stay_absent(self):
        metrics = {
            "market_metrics": [
                {"category_id": "lip_care", "hhi": None, "cpi": None, "churn_rate_7d": None}
            ]
        }

        context = self._retriever()._build_inference_context({"categories": ["lip_care"]}, metrics)

        for key in ("hhi", "cpi", "churn_rate", "rating_gap"):
            assert key not in context, f"{key}가 결측인데 기본값으로 채워졌다"

    def test_present_market_fields_are_kept(self):
        metrics = {"market_metrics": [{"category_id": "lip_care", "hhi": 0.0681}]}

        context = self._retriever()._build_inference_context({"categories": ["lip_care"]}, metrics)

        assert context["hhi"] == 0.0681

    def test_missing_brand_share_is_not_zero(self):
        metrics = {"brand_metrics": [{"is_laneige": True, "avg_rank": 8.5}]}

        context = self._retriever()._build_inference_context({"categories": ["lip_care"]}, metrics)

        assert "sos" not in context
