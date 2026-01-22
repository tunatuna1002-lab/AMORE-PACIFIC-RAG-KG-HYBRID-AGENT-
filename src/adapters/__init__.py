"""
Adapters Layer
==============
Clean Architecture의 Interface Adapters Layer

이 레이어는 외부 인터페이스를 내부 Use Case에 맞게 변환합니다.
Domain Protocol들의 구체적인 구현체를 포함합니다.

구조:
- agents/: Agent 구현체 (CrawlerAgent, StorageAgent 등)
- rag/: RAG 컴포넌트 (HybridRetriever, ContextBuilder 등)
- presenters/: 데이터 변환/포맷팅 (DashboardPresenter 등)

호환성:
- 기존 src/agents/는 이 레이어에서 re-export됩니다
- 기존 src/rag/도 이 레이어에서 re-export됩니다
"""

# Re-export for convenience (actual implementations are in src/agents, src/rag)
# These will be migrated here in a future phase
