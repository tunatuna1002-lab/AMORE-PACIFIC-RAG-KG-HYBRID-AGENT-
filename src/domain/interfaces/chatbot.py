"""
Chatbot Agent Protocol
======================
HybridChatbotAgent에 대한 추상 인터페이스

구현체:
- HybridChatbotAgent (src/agents/hybrid_chatbot_agent.py)
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ChatbotAgentProtocol(Protocol):
    """
    Hybrid Chatbot Agent Protocol

    Ontology-RAG 하이브리드 챗봇 에이전트 인터페이스.
    Knowledge Graph, Ontology Reasoner, RAG를 통합하여
    사용자 질문에 답변합니다.

    Methods:
        chat: 사용자 질문 처리 및 응답 생성
        set_data_context: 데이터 컨텍스트 설정
        get_conversation_history: 대화 기록 조회
        clear_conversation: 대화 기록 초기화
        get_last_hybrid_context: 마지막 하이브리드 컨텍스트 조회
        get_knowledge_graph: Knowledge Graph 인스턴스 반환
        get_reasoner: Ontology Reasoner 인스턴스 반환
        explain_last_response: 마지막 응답 설명 생성
    """

    async def chat(
        self,
        query: str,
        session_id: str | None = None,
        current_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        사용자 질문을 처리하고 응답을 생성합니다.

        Args:
            query: 사용자 질문
            session_id: 세션 ID (선택)
            current_metrics: 현재 메트릭 데이터 (선택)

        Returns:
            응답 딕셔너리
            {
                "response": str,              # 생성된 응답
                "sources": [...],             # 참조한 소스 목록
                "confidence": float,          # 신뢰도 (0~1)
                "suggestions": [...],         # 추천 질문
                "entities": {...},            # 추출된 엔티티
                "inferences": [...],          # 추론 결과
                "query_type": str,            # 질문 유형
                "rewrite": {...},             # 질문 재작성 정보
                ...
            }
        """
        ...

    def set_data_context(self, data: dict[str, Any]) -> None:
        """
        데이터 컨텍스트를 설정합니다.

        최신 메트릭 데이터를 챗봇에 전달하여
        실시간 정보를 기반으로 답변할 수 있도록 합니다.

        Args:
            data: 컨텍스트 데이터 (메트릭, 제품 정보 등)
        """
        ...

    def get_conversation_history(self, limit: int = 10) -> list[dict]:
        """
        대화 기록을 조회합니다.

        Args:
            limit: 반환할 최대 대화 수

        Returns:
            대화 기록 리스트 [{"role": "user", "content": "..."}, ...]
        """
        ...

    def clear_conversation(self) -> None:
        """
        대화 기록을 초기화합니다.
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

    async def explain_last_response(self) -> str:
        """
        마지막 응답에 대한 설명을 생성합니다.

        추론 과정, 사용한 데이터 소스 등을 설명합니다.

        Returns:
            설명 텍스트
        """
        ...
