# Session 2: Domain 레이어 강화 (인터페이스 확충)

> ⏱ 예상 시간: 30~40분 | 위험도: 🟢 낮음 | 선행 조건: Session 1 완료

---

## 프롬프트 (아래를 복사해서 새 Claude Code 세션에 붙여넣기)

```
너는 20년 베테랑 Python 개발자이자 Clean Architecture 전문가야. AMORE RAG-KG Hybrid Agent의 Domain 레이어를 강화하는 세션이야.

## 이번 세션 목표
`src/domain/` 레이어에 Protocol 인터페이스를 확충해서, 이후 세션에서 순환 의존성을 끊을 수 있는 기반을 만들어.

## 컨텍스트
- 프로젝트: `/Users/leedongwon/Desktop/AMORE-RAG-ONTOLOGY-HYBRID AGENT/`
- 전체 마스터 플랜: `docs/refactoring/00_MASTER_PLAN.md` 참조
- 의존성 그래프: `DEPENDENCY_GRAPH.txt` 참조
- Python 3.13.7 (`python3` 사용)
- **핵심 문제**: src.core ↔ src.agents ↔ src.tools ↔ src.api 간 순환 의존성 23개

## 현재 src/domain/ 구조
```
src/domain/
├── entities/
│   ├── brand.py       # Brand 엔티티
│   ├── market.py      # Product, Category 등
│   └── relations.py   # 관계 정의
├── interfaces/
│   ├── agent.py       # CrawlerAgentProtocol 등 (일부만 있음)
│   ├── knowledge_graph.py
│   ├── llm_client.py
│   ├── repository.py
│   ├── retriever.py
│   └── scraper.py
├── value_objects/
│   └── __init__.py    # 비어있음
└── exceptions.py
```

## 수행할 작업

### 1. 기존 인터페이스 감사
먼저 `src/domain/interfaces/` 안의 모든 Protocol을 읽고, 어떤 메서드가 정의되어 있는지 확인해줘.
그다음 실제 구현체(src/agents/, src/tools/, src/core/ 등)가 이 Protocol을 따르는지 확인해줘.

### 2. 누락된 인터페이스 추가
순환 의존성을 끊으려면 다음 Protocol이 필요해:

#### a) `src/domain/interfaces/brain.py` (NEW)
- `BrainProtocol`: src/core/brain.py의 핵심 메서드를 Protocol로 정의
  - `process_query()`, `get_status()` 등

#### b) `src/domain/interfaces/insight.py` (NEW)
- `InsightAgentProtocol`: 인사이트 생성 에이전트
  - `generate_insight()` 등

#### c) `src/domain/interfaces/chatbot.py` (NEW)
- `ChatbotAgentProtocol`: 챗봇 에이전트
  - `chat()`, `process_message()` 등

#### d) `src/domain/interfaces/storage.py` (NEW)
- `StorageProtocol`: SQLite/DB 스토리지
  - `save_products()`, `get_products()`, `get_dashboard_data()` 등

#### e) `src/domain/interfaces/metric.py` (NEW)
- `MetricCalculatorProtocol`: KPI 계산기
  - `calculate_sos()`, `calculate_hhi()`, `calculate_cpi()` 등

#### f) `src/domain/interfaces/signal.py` (NEW)
- `SignalCollectorProtocol`: 외부 신호 수집기
  - `collect_signals()` 등

### 3. 기존 인터페이스 보강
- `agent.py` — 현재 CrawlerAgentProtocol만 있으면, AlertAgentProtocol 등 추가
- `retriever.py` — HybridRetrieverProtocol이 실제 구현체와 맞는지 확인/수정
- `scraper.py` — AmazonScraperProtocol이 실제 구현체와 맞는지 확인/수정

### 4. Value Objects 정의 (선택)
비어있는 `value_objects/`에 필요한 VO 추가:
- `CategoryId`, `BrandName` 등 (있으면 좋지만 필수는 아님)

### 5. TDD 방식
- 각 Protocol에 대한 테스트를 먼저 작성: "이 Protocol을 구현한 클래스가 필요한 메서드를 갖는가"
- `tests/unit/domain/test_interfaces.py`에 추가

### 6. 검증
- `python3 -m pytest tests/unit/domain/ -v` — domain 테스트 통과
- `python3 -m pytest tests/ -v --tb=short` — 전체 테스트 통과
- 기존 코드의 import가 깨지지 않는지 확인

## 주의사항
- Domain 레이어는 외부 패키지 의존 최소화 (pydantic, typing, abc만)
- Protocol 정의 시 실제 구현체의 시그니처를 반드시 확인
- 이번 세션에서는 Protocol만 만들고, 구현체에 적용하는 것은 이후 세션에서
- Context7 MCP를 활용해서 pydantic, Python Protocol 관련 최신 문서를 참조해도 좋아
```

---

## 이 세션의 체크리스트

- [ ] 기존 인터페이스 감사 완료
- [ ] BrainProtocol 추가
- [ ] InsightAgentProtocol 추가
- [ ] ChatbotAgentProtocol 추가
- [ ] StorageProtocol 추가
- [ ] MetricCalculatorProtocol 추가
- [ ] SignalCollectorProtocol 추가
- [ ] 기존 인터페이스 보강
- [ ] 테스트 작성 및 통과
- [ ] 전체 테스트 통과 확인
