"""
OWL 검색 전략 제거 회귀 테스트 (S4-1, 트랙 4-C)
===============================================
E3 결론: OWL은 검색 전략이 아니다. `OWLRetrievalStrategy`·`create_owl_strategy`·
`RetrievalStrategy` 프로토콜과 플래그 `retriever.use_owl_strategy`를 삭제했다.
되살아나면 이 테스트가 잡는다.

유지되는 것: `IntentRetrievalConfig`/`get_intent_retrieval_config`(hybrid_retriever·
container가 사용), `reasoner.*` 플래그(규칙 추론 on/off), `src/ontology/owl_reasoner.py`
(어휘 용도).
"""

import inspect

import pytest

import src.rag as rag_pkg
from src.rag import retrieval_strategy


class TestDeletedSymbols:
    @pytest.mark.parametrize(
        "symbol", ["OWLRetrievalStrategy", "create_owl_strategy", "RetrievalStrategy"]
    )
    def test_retrieval_strategy_module_no_longer_exports(self, symbol: str) -> None:
        assert not hasattr(retrieval_strategy, symbol)

    @pytest.mark.parametrize(
        "symbol", ["OWLRetrievalStrategy", "create_owl_strategy", "RetrievalStrategy"]
    )
    def test_rag_package_no_longer_exports(self, symbol: str) -> None:
        assert not hasattr(rag_pkg, symbol)
        assert symbol not in rag_pkg.__all__

    def test_intent_config_is_kept(self) -> None:
        from src.core.intent import UnifiedIntent

        config = retrieval_strategy.get_intent_retrieval_config(UnifiedIntent.GENERAL)
        assert config.top_k == 8
        assert rag_pkg.IntentRetrievalConfig is retrieval_strategy.IntentRetrievalConfig

    def test_owl_reasoner_module_is_kept_as_vocabulary(self) -> None:
        from src.ontology.owl_reasoner import OWLReasoner

        assert OWLReasoner is not None


class TestHybridRetrieverHasNoStrategyHook:
    def test_constructor_has_no_owl_strategy_parameter(self) -> None:
        from src.rag.hybrid_retriever import HybridRetriever

        params = inspect.signature(HybridRetriever.__init__).parameters
        assert "owl_strategy" not in params

    def test_instance_has_no_owl_strategy_attribute(self) -> None:
        from unittest.mock import MagicMock

        from src.rag.hybrid_retriever import HybridRetriever

        retriever = HybridRetriever(
            knowledge_graph=MagicMock(),
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )
        assert not hasattr(retriever, "owl_strategy")


class TestBrainStatusShape:
    @pytest.mark.asyncio
    async def test_component_status_has_no_owl_entry(self) -> None:
        from src.core.brain import UnifiedBrain

        status = UnifiedBrain().get_component_status()

        assert "owl_strategy" not in status
        assert set(status) == {"react_agent"}
