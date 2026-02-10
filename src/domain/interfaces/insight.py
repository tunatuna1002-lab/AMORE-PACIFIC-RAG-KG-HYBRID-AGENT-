"""
Insight Agent Protocol
======================
HybridInsightAgent에 대한 추상 인터페이스

구현체:
- HybridInsightAgent (src/agents/hybrid_insight_agent.py)
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class InsightAgentProtocol(Protocol):
    """
    Hybrid Insight Agent Protocol

    Ontology-RAG 하이브리드 인사이트 생성 에이전트 인터페이스.
    Knowledge Graph 업데이트, Ontology 추론, RAG 검색을 통합하여
    전략적 인사이트를 생성합니다.

    Methods:
        execute: 메트릭 데이터 기반 인사이트 생성
        get_results: 마지막 실행 결과 조회
        get_last_hybrid_context: 마지막 하이브리드 컨텍스트 조회
        get_knowledge_graph: Knowledge Graph 인스턴스 반환
        get_reasoner: Ontology Reasoner 인스턴스 반환
    """

    async def execute(
        self,
        metrics_data: dict[str, Any],
        target_brand: str = "LANEIGE",
        category_id: str | None = None,
    ) -> dict[str, Any]:
        """
        메트릭 데이터를 분석하고 전략적 인사이트를 생성합니다.

        Flow:
        1. Knowledge Graph 업데이트
        2. Ontology Reasoner로 규칙 기반 추론
        3. RAG로 관련 가이드라인 검색
        4. 외부 신호 수집 (뉴스, 트렌드, 소셜 미디어)
        5. 통합 컨텍스트로 LLM 인사이트 생성

        Args:
            metrics_data: 메트릭 데이터 딕셔너리
                {
                    "brand_metrics": {...},
                    "market_metrics": {...},
                    "product_metrics": {...},
                    ...
                }
            target_brand: 분석 대상 브랜드 (기본: "LANEIGE")
            category_id: 카테고리 ID (선택)

        Returns:
            인사이트 결과 딕셔너리
            {
                "insights": [
                    {
                        "type": "opportunity" | "threat" | "recommendation",
                        "priority": "high" | "medium" | "low",
                        "title": str,
                        "content": str,
                        "action_items": [...],
                        "data_sources": [...],
                        "confidence": float,
                        ...
                    },
                    ...
                ],
                "summary": str,
                "highlights": [...],
                "external_signals": {...},
                "execution_time": float,
                "cost": float,
                ...
            }
        """
        ...

    def get_results(self) -> dict[str, Any]:
        """
        마지막 실행 결과를 반환합니다.

        Returns:
            실행 결과 딕셔너리 (execute()의 반환값과 동일)
        """
        ...

    def get_last_hybrid_context(self) -> Any:
        """
        마지막으로 사용한 하이브리드 컨텍스트를 반환합니다.

        Returns:
            HybridContext 객체 또는 None
        """
        ...

    def get_knowledge_graph(self) -> Any:
        """
        Knowledge Graph 인스턴스를 반환합니다.

        Returns:
            KnowledgeGraph 인스턴스
        """
        ...

    def get_reasoner(self) -> Any:
        """
        Ontology Reasoner 인스턴스를 반환합니다.

        Returns:
            OntologyReasoner 인스턴스
        """
        ...
