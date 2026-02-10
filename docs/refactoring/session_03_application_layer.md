# Session 3: Application 레이어 구축 (Use Cases)

> ⏱ 예상 시간: 50~70분 | 위험도: 🟡 중간 | 선행 조건: Session 2 완료

---

## 프롬프트 (아래를 복사해서 새 Claude Code 세션에 붙여넣기)

```
너는 20년 베테랑 Python 개발자이자 Clean Architecture 전문가야. AMORE RAG-KG Hybrid Agent의 Application 레이어를 구축하는 세션이야.

## 이번 세션 목표
현재 거의 비어있는 `src/application/` (120줄)에 Use Case를 구축해서 비즈니스 로직의 중심지를 만들어.
현재 `src/core/brain.py`와 `src/agents/`에 흩어져 있는 워크플로우 로직을 Application 레이어로 이동시킬 거야.

## 컨텍스트
- 프로젝트: `/Users/leedongwon/Desktop/AMORE-RAG-ONTOLOGY-HYBRID AGENT/`
- 전체 마스터 플랜: `docs/refactoring/00_MASTER_PLAN.md` 참조
- Session 2에서 `src/domain/interfaces/`에 Protocol들이 추가되었음
- Python 3.13.7 (`python3` 사용)

## 현재 문제
- `src/application/`이 120줄밖에 안 됨 (정상이면 8,000~10,000줄)
- 비즈니스 워크플로우가 brain.py(1787줄)에 집중되어 있음
- API 라우트가 직접 core/agents를 호출함

## 수행할 작업

### 1. 현재 워크플로우 파악
먼저 다음 파일들을 읽고, 어떤 워크플로우(비즈니스 프로세스)가 존재하는지 파악해줘:
- `src/core/brain.py` — process_query, schedule_crawl, generate_insight 등
- `src/core/batch_workflow.py` — 배치 처리
- `src/agents/hybrid_chatbot_agent.py` — 채팅 흐름
- `src/agents/hybrid_insight_agent.py` — 인사이트 생성 흐름
- `src/agents/crawler_agent.py` — 크롤링 흐름
- `dashboard_api.py` — API에서 직접 호출하는 로직

### 2. Use Case 설계 및 구현 (TDD)

각 Use Case에 대해 **테스트를 먼저 작성**하고, 그 다음 구현해:

#### a) `src/application/workflows/chat_workflow.py`
```python
class ChatWorkflow:
    """사용자 질문 → 답변 생성 워크플로우"""
    def __init__(self, retriever: RetrieverProtocol, chatbot: ChatbotAgentProtocol, ...):
        ...
    async def execute(self, query: str, session_id: str) -> ChatResponse:
        # 1. 쿼리 분석 (복잡도 판단)
        # 2. 검색 (RAG + KG)
        # 3. 답변 생성
        # 4. 세션 기록
```

#### b) `src/application/workflows/crawl_workflow.py`
```python
class CrawlWorkflow:
    """Amazon BSR 크롤링 → 저장 → KPI 계산 워크플로우"""
    def __init__(self, scraper: ScraperProtocol, storage: StorageProtocol, ...):
        ...
    async def execute(self, categories: List[str]) -> CrawlResult:
        # 1. 카테고리별 크롤링
        # 2. 데이터 저장
        # 3. KPI 계산
        # 4. KG 업데이트
```

#### c) `src/application/workflows/insight_workflow.py`
```python
class InsightWorkflow:
    """데이터 분석 → 인사이트 생성 워크플로우"""
    def __init__(self, insight_agent: InsightAgentProtocol, ...):
        ...
    async def execute(self, data_range: DateRange) -> InsightResult:
        # 1. 데이터 수집
        # 2. 분석
        # 3. 인사이트 생성
        # 4. 저장
```

#### d) `src/application/workflows/alert_workflow.py`
```python
class AlertWorkflow:
    """변동 감지 → 알림 발송 워크플로우"""
```

### 3. 공통 서비스
#### `src/application/services/query_analyzer.py`
- 쿼리 복잡도 판단 로직 (현재 brain.py에 있는 것)을 분리

### 4. 기존 코드와의 연결
- 이번 세션에서는 Use Case를 **만들기만** 해. 기존 코드(brain.py 등)는 아직 수정하지 마.
- Session 7-8에서 brain.py가 이 Use Case를 호출하도록 변경할 거야.
- 단, Use Case 내부에서는 Session 2에서 만든 Protocol을 사용해.

### 5. 테스트
- `tests/unit/application/` 디렉토리에 테스트 작성
- 모든 의존성은 mock (Protocol 기반)
- 각 워크플로우의 정상/에러 경로 테스트

### 6. 검증
- `python3 -m pytest tests/unit/application/ -v` — application 테스트 통과
- `python3 -m pytest tests/ -v --tb=short` — 전체 테스트 통과

## 주의사항
- Application 레이어는 **domain 레이어만** import 가능 (인터페이스 통해)
- 직접 src/agents/, src/tools/ 등을 import하면 안 됨
- 비즈니스 로직의 "흐름"만 정의, 구체적 구현은 Protocol 뒤에 숨겨
- Context7 MCP로 Python Protocol, DI 패턴 관련 문서 참조 가능
```

---

## 이 세션의 체크리스트

- [ ] 기존 워크플로우 파악 완료
- [ ] ChatWorkflow 테스트 + 구현
- [ ] CrawlWorkflow 테스트 + 구현
- [ ] InsightWorkflow 테스트 + 구현
- [ ] AlertWorkflow 테스트 + 구현
- [ ] QueryAnalyzer 서비스 분리
- [ ] 전체 테스트 통과 확인
