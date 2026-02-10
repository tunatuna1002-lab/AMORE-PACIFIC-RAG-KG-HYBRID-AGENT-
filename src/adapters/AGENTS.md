# src/adapters/ - Interface Adapters

## 개요

Clean Architecture Layer 3 - Interface Adapters 레이어입니다.
Application Layer와 Infrastructure Layer 사이의 데이터 변환을 담당합니다.

## 역할

- Use Case 입출력을 외부 프레임워크/드라이버에 맞게 변환
- Domain 모델 ↔ DTO/API Response 변환
- Controller/Presenter 패턴 구현

## 디렉토리 구조

| 디렉토리 | 상태 | 설명 |
|----------|------|------|
| `agents/` | 구현 예정 | Agent 어댑터 (LLM 호출 래핑) |
| `rag/` | 구현 예정 | RAG 시스템 어댑터 |
| `presenters/` | 구현 예정 | 응답 데이터 포맷팅 |

## 파일 목록

| 파일 | 설명 |
|------|------|
| `__init__.py` | 패키지 초기화 (빈 파일) |
| `AGENTS.md` | 현재 문서 |

## 의존성 규칙

### 허용되는 Import
- `src.domain.*` - Domain entities, interfaces
- `src.application.*` - Use cases, workflows
- 표준 라이브러리

### 금지된 Import
- `src.infrastructure.*` - Infrastructure는 Adapter를 참조해야 함 (역방향 금지)

## 구현 예정 사항

### 1. Agent Adapters
```python
# agents/chatbot_adapter.py
class ChatbotAdapter:
    def __init__(self, chatbot_agent: ChatbotAgentProtocol):
        self.agent = chatbot_agent

    async def process(self, request: ChatRequest) -> ChatResponse:
        # DTO → Domain → Agent → Domain → DTO
        pass
```

### 2. RAG Adapters
```python
# rag/retriever_adapter.py
class RetrieverAdapter:
    def __init__(self, retriever: RetrieverProtocol):
        self.retriever = retriever

    async def retrieve(self, query: QueryDTO) -> List[DocumentDTO]:
        pass
```

### 3. Presenters
```python
# presenters/dashboard_presenter.py
class DashboardPresenter:
    def present(self, data: DomainData) -> DashboardResponse:
        # Domain → API Response 변환
        pass
```

## 주의사항

1. **단방향 의존성**: Adapter는 Application과 Domain만 참조
2. **구현체 분리**: Infrastructure 구현체 직접 참조 금지
3. **DTO 변환**: 외부 형식 ↔ Domain 모델 변환 책임
4. **Thin Layer**: 비즈니스 로직 포함 금지 (변환 로직만)

## 현재 상태

- 현재 Agent/RAG 구현은 `src/agents/`, `src/rag/`에 직접 존재
- Clean Architecture 완전 전환은 점진적 진행 예정
- `__init__.py`만 존재하며 실제 구현체는 아직 없음

## 참고

- [Clean Architecture - Uncle Bob](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html)
- `src/domain/interfaces/` - Protocol 정의
- `src/application/workflows/` - Use Case 구현
