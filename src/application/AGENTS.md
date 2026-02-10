# src/application/ - Use Cases

## 개요

Clean Architecture Layer 2 - Application Business Rules 레이어입니다.
시스템의 Use Case와 Application-specific 비즈니스 로직을 구현합니다.

## 역할

- Use Case 구현 (workflows)
- Application 서비스 조정
- Domain 인터페이스만 참조 (구현체 직접 import 금지)
- 외부 의존성으로부터 Domain 격리

## 디렉토리 구조

| 디렉토리 | 파일 수 | 설명 |
|----------|---------|------|
| `workflows/` | 5개 | Use Case 구현체 |
| `services/` | 1개 | Application 서비스 |
| `orchestrators/` | 0개 | 워크플로우 조정 (예정) |

## 파일 목록

### workflows/

| 파일 | 설명 | 주요 클래스 |
|------|------|------------|
| `chat_workflow.py` | 챗봇 대화 처리 | `ChatWorkflow` |
| `crawl_workflow.py` | Amazon 크롤링 워크플로우 | `CrawlWorkflow` |
| `insight_workflow.py` | 인사이트 생성 워크플로우 | `InsightWorkflow` |
| `alert_workflow.py` | 알림 발송 워크플로우 | `AlertWorkflow` |
| `batch_workflow.py` | 배치 작업 조정 | `BatchWorkflow` |
| `__init__.py` | 패키지 초기화 | - |

### services/

| 파일 | 설명 | 주요 클래스 |
|------|------|------------|
| `query_analyzer.py` | 쿼리 분석 서비스 | `QueryAnalyzer` |
| `__init__.py` | 패키지 초기화 | - |

## 의존성 규칙

### ✅ 허용되는 Import

```python
# Domain Layer만 참조
from src.domain.entities.product import Product
from src.domain.interfaces.agent import ChatbotAgentProtocol
from src.domain.interfaces.storage import StorageProtocol

# 표준 라이브러리
import asyncio
from typing import List, Dict
```

### ❌ 금지된 Import

```python
# 구현체 직접 참조 금지
from src.agents.hybrid_chatbot_agent import HybridChatbotAgent  # ❌
from src.rag.hybrid_retriever import HybridRetriever  # ❌
from src.infrastructure.database import SQLiteDatabase  # ❌

# 대신 Protocol 사용
from src.domain.interfaces.agent import ChatbotAgentProtocol  # ✅
```

## Workflow 패턴

### 1. Dependency Injection

```python
# workflows/chat_workflow.py
class ChatWorkflow:
    def __init__(
        self,
        chatbot: ChatbotAgentProtocol,  # Protocol 주입
        storage: StorageProtocol
    ):
        self.chatbot = chatbot
        self.storage = storage

    async def execute(self, query: str) -> str:
        # Use Case 로직
        pass
```

### 2. Use Case 실행

```python
# workflows/crawl_workflow.py
class CrawlWorkflow:
    async def execute(self, category: str) -> List[Product]:
        # 1. 크롤링 실행
        products = await self.crawler.crawl(category)

        # 2. DB 저장
        await self.storage.save_products(products)

        # 3. KG 업데이트
        await self.kg.add_products(products)

        return products
```

## 주요 Workflows

### ChatWorkflow
- 사용자 쿼리 → AI 챗봇 응답
- RAG + KG + Ontology 하이브리드 검색
- 대화 컨텍스트 관리

### CrawlWorkflow
- Amazon 카테고리 크롤링
- 제품 데이터 파싱 및 저장
- KG 자동 업데이트

### InsightWorkflow
- 데이터 분석 → 전략적 인사이트 생성
- 외부 신호 통합 (뉴스, 트렌드)
- LLM 기반 텍스트 생성

### AlertWorkflow
- KPI 변동 감지 (순위 ±10, SoS 변동)
- 이메일 알림 발송
- 알림 이력 기록

### BatchWorkflow
- 일일 배치 작업 조정
- Crawl → Metric → Insight → Alert 순차 실행
- 실패 시 재시도 로직

## 주의사항

1. **Protocol 우선**: 구체 클래스 대신 Protocol 사용
2. **순수성 유지**: HTTP/DB/외부 API 직접 호출 금지
3. **Single Responsibility**: 하나의 Workflow = 하나의 Use Case
4. **Error Handling**: 외부 실패는 재시도 또는 graceful degradation
5. **Testing**: Mock Protocol로 단위 테스트 가능해야 함

## DI 컨테이너

실제 구현체 주입은 상위 레이어에서:

```python
# dashboard_api.py (Infrastructure Layer)
from src.agents.hybrid_chatbot_agent import HybridChatbotAgent
from src.application.workflows.chat_workflow import ChatWorkflow

# 구현체 생성
chatbot = HybridChatbotAgent(...)

# Workflow에 주입
chat_workflow = ChatWorkflow(chatbot=chatbot)
```

## 테스트

```bash
# Application Layer 테스트
python3 -m pytest tests/unit/application/ -v
```

## 참고

- `src/domain/interfaces/` - Protocol 정의
- `src/domain/entities/` - Domain 모델
- `tests/unit/application/` - Workflow 테스트
- `dashboard_api.py` - DI 컨테이너 역할
