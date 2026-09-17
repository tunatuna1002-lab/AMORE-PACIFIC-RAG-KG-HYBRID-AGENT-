"""
인텐트 기반 검색 설정 단위 테스트
=================================
IntentRetrievalConfig / get_intent_retrieval_config 와 컨텍스트 예산 드리프트 방지.

2026-09: OWLRetrievalStrategy·RetrievalStrategy 프로토콜 삭제로 해당 테스트도 제거했다.
"""

import pytest

from src.core.intent import UnifiedIntent
from src.rag.retrieval_strategy import IntentRetrievalConfig, get_intent_retrieval_config


class TestIntentRetrievalConfig:
    """IntentRetrievalConfig 데이터클래스 테스트"""

    def test_default_values(self):
        """기본값 확인"""
        config = IntentRetrievalConfig()
        assert config.weights == {"kg": 0.4, "rag": 0.4, "inference": 0.2}
        assert config.top_k == 5
        assert config.doc_type_filter is None
        assert config.description == "default"

    def test_frozen_dataclass(self):
        """frozen=True이므로 수정 불가"""
        config = IntentRetrievalConfig()
        with pytest.raises(AttributeError):
            config.top_k = 10  # type: ignore[misc]

    def test_custom_values(self):
        """커스텀 값으로 생성"""
        config = IntentRetrievalConfig(
            weights={"kg": 0.6, "rag": 0.2, "inference": 0.2},
            top_k=10,
            doc_type_filter=["playbook"],
            description="test",
        )
        assert config.weights["kg"] == 0.6
        assert config.top_k == 10
        assert config.doc_type_filter == ["playbook"]
        assert config.description == "test"


class TestGetIntentRetrievalConfig:
    """get_intent_retrieval_config 함수 테스트"""

    def test_all_intents_mapped(self):
        """모든 UnifiedIntent 값에 대해 config 반환"""
        for intent in UnifiedIntent:
            config = get_intent_retrieval_config(intent)
            assert isinstance(config, IntentRetrievalConfig)
            assert isinstance(config.weights, dict)
            assert "kg" in config.weights
            assert "rag" in config.weights
            assert "inference" in config.weights

    def test_weights_sum_to_one(self):
        """모든 인텐트의 가중치 합이 1.0"""
        for intent in UnifiedIntent:
            config = get_intent_retrieval_config(intent)
            total = sum(config.weights.values())
            assert abs(total - 1.0) < 0.01, f"{intent.value}: weights sum to {total}"

    def test_diagnosis_is_graph_heavy(self):
        """DIAGNOSIS → KG 가중치가 RAG보다 높음"""
        config = get_intent_retrieval_config(UnifiedIntent.DIAGNOSIS)
        assert config.weights["kg"] > config.weights["rag"]
        assert config.description.startswith("graph-heavy")

    def test_competitive_is_graph_heavy(self):
        """COMPETITIVE → KG 가중치가 높음"""
        config = get_intent_retrieval_config(UnifiedIntent.COMPETITIVE)
        assert config.weights["kg"] >= config.weights["rag"]
        assert config.description.startswith("graph-heavy")

    def test_general_is_vector_heavy(self):
        """GENERAL → RAG 가중치가 KG보다 높음"""
        config = get_intent_retrieval_config(UnifiedIntent.GENERAL)
        assert config.weights["rag"] > config.weights["kg"]
        assert config.description.startswith("vector-heavy")

    def test_definition_is_vector_heavy(self):
        """DEFINITION → RAG 문서 가중치가 높음"""
        config = get_intent_retrieval_config(UnifiedIntent.DEFINITION)
        assert config.weights["rag"] > config.weights["kg"]
        assert "metric_guide" in (config.doc_type_filter or [])

    def test_analysis_is_inference_heavy(self):
        """ANALYSIS → inference 가중치가 가장 높음"""
        config = get_intent_retrieval_config(UnifiedIntent.ANALYSIS)
        assert config.weights["inference"] >= config.weights["kg"]
        assert config.weights["inference"] >= config.weights["rag"]

    def test_insight_rule_is_inference_heavy(self):
        """INSIGHT_RULE → inference 가중치가 높음"""
        config = get_intent_retrieval_config(UnifiedIntent.INSIGHT_RULE)
        assert config.weights["inference"] >= config.weights["kg"]

    def test_trend_is_hybrid(self):
        """TREND → 균형 잡힌 가중치"""
        config = get_intent_retrieval_config(UnifiedIntent.TREND)
        assert config.description.startswith("hybrid")
        # top_k가 기본보다 높음 (더 넓은 검색)
        assert config.top_k >= 5

    def test_crisis_has_response_guide_priority(self):
        """CRISIS → response_guide 문서 우선"""
        config = get_intent_retrieval_config(UnifiedIntent.CRISIS)
        assert config.doc_type_filter is not None
        assert "response_guide" in config.doc_type_filter

    def test_metric_has_metric_guide_priority(self):
        """METRIC → metric_guide 문서 우선"""
        config = get_intent_retrieval_config(UnifiedIntent.METRIC)
        assert config.doc_type_filter is not None
        assert "metric_guide" in config.doc_type_filter

    def test_general_has_no_doc_filter(self):
        """GENERAL → 전체 문서 검색 (필터 없음)"""
        config = get_intent_retrieval_config(UnifiedIntent.GENERAL)
        assert config.doc_type_filter is None

    def test_data_query_has_no_doc_filter(self):
        """DATA_QUERY → 전체 문서 검색"""
        config = get_intent_retrieval_config(UnifiedIntent.DATA_QUERY)
        assert config.doc_type_filter is None

    def test_top_k_positive(self):
        """모든 인텐트의 top_k가 양수"""
        for intent in UnifiedIntent:
            config = get_intent_retrieval_config(intent)
            assert config.top_k > 0


class TestContextBudgetAlignment:
    """인텐트별 검색 top_k는 선언된 컨텍스트 예산과 일치해야 한다 (사이클 8).

    사이클 2에서 컨텍스트 상한을 3→8로 올렸는데 인텐트별 top_k는 5로 남아
    검색이 예산보다 적게 공급했고, 상한 8은 한 번도 걸리지 않았다.
    같은 드리프트가 다시 생기면 이 테스트가 잡는다.
    """

    @staticmethod
    def _declared_budget() -> int:
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parents[3] / "config" / "retrieval_weights.json"
        return json.loads(path.read_text(encoding="utf-8"))["max_context_items"]["rag_chunks"]

    def test_every_intent_supplies_the_declared_budget(self):
        from src.rag.retrieval_strategy import _INTENT_STRATEGY_MAP

        budget = self._declared_budget()
        offenders = {
            intent.name: cfg.top_k
            for intent, cfg in _INTENT_STRATEGY_MAP.items()
            if cfg.top_k != budget
        }

        assert not offenders, f"컨텍스트 예산({budget})과 어긋난 인텐트: {offenders}"

    def test_intents_still_differentiate_by_weights(self):
        """차등은 weights로 표현한다 — top_k를 줄여서 표현하지 않는다."""
        from src.rag.retrieval_strategy import _INTENT_STRATEGY_MAP

        kg_weights = {cfg.weights["kg"] for cfg in _INTENT_STRATEGY_MAP.values()}

        assert len(kg_weights) > 1
