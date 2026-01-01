# LLM Orchestrator 재설계 설계서

## 문서 정보
- **버전**: 1.0
- **작성일**: 2025-12-31
- **작성자**: AI Agent 설계 전문가
- **상태**: 설계 완료, 구현 대기

---

## 1. 개요

### 1.1 배경
현재 AMORE-RAG-ONTOLOGY-HYBRID AGENT 시스템은 하드코딩된 파이프라인 방식으로 동작하며, 사용자 입력에 따른 동적 라우팅 기능이 없다. 이로 인해:
- 매번 전체 워크플로우가 실행됨 (비효율)
- 사용자 의도에 맞는 유연한 응답 불가
- 진정한 AI Agent로서의 역할 부재

### 1.2 목표
- LLM이 오케스트레이터 역할을 수행하는 진정한 AI Agent 시스템 구축
- RAG + Knowledge Graph 기반의 정확한 판단
- 신뢰도 기반 2단계 라우팅으로 비용 최적화
- 모든 답변에 RAG + KG 컨텍스트 반영

### 1.3 범위
| 포함 | 제외 |
|------|------|
| LLM Orchestrator 신규 개발 | 기존 에이전트 로직 변경 |
| RAG+KG 기반 판단 로직 | 크롤링/저장 로직 변경 |
| 응답 파이프라인 통합 | UI/대시보드 변경 |
| 캐싱 및 상태 관리 | 외부 API 변경 |

---

## 2. 시스템 아키텍처

### 2.1 현재 아키텍처 (AS-IS)

```
┌─────────────────────────────────────────────────────────────┐
│                        main.py                               │
├─────────────────────────────────────────────────────────────┤
│  --workflow 모드          │  --chat 모드                    │
│         │                 │         │                        │
│         ▼                 │         ▼                        │
│  Orchestrator             │  HybridChatbotAgent             │
│  (하드코딩 파이프라인)      │  (독립적 동작)                   │
│         │                 │         │                        │
│  CRAWL→STORE→CALC→INSIGHT │  RAG+KG→LLM                     │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    dashboard_api.py                          │
│  (별도 FastAPI 서버, Orchestrator 미사용)                    │
│         │                                                    │
│  RAGRouter → LLM 직접 호출                                   │
└─────────────────────────────────────────────────────────────┘
```

**문제점**:
1. 두 개의 분리된 진입점 (main.py, dashboard_api.py)
2. 오케스트레이터가 LLM이 아닌 if-else 분기
3. 동적 라우팅 없음
4. dashboard_api가 KnowledgeGraph 미활용

### 2.2 목표 아키텍처 (TO-BE)

```
┌─────────────────────────────────────────────────────────────┐
│              사용자 입력 (CLI / API / Dashboard)             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     LLMOrchestrator                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ 1. Rule-based 1차 분류 (RAGRouter)                      ││
│  │    - 키워드/패턴 매칭으로 빠른 분류                      ││
│  │    - 신뢰도 점수 산출                                   ││
│  └─────────────────────────────────────────────────────────┘│
│                              │                               │
│                              ▼                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ 2. 컨텍스트 수집 (ContextGatherer)                      ││
│  │    - RAG: 관련 문서 검색                                ││
│  │    - KG: 엔티티/관계 조회                               ││
│  │    - State: 데이터 신선도 확인                          ││
│  └─────────────────────────────────────────────────────────┘│
│                              │                               │
│                              ▼                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ 3. 신뢰도 기반 분기                                     ││
│  │    - HIGH (5.0+): Rule 결과 신뢰, 바로 응답             ││
│  │    - MEDIUM (3.0~4.9): LLM 도구 선택                    ││
│  │    - LOW (1.5~2.9): LLM 전체 판단                       ││
│  │    - UNKNOWN (<1.5): 명확화 요청                        ││
│  └─────────────────────────────────────────────────────────┘│
│                              │                               │
│                              ▼                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ 4. 도구 실행 (필요시)                                   ││
│  │    - crawl_amazon: CrawlerAgent                         ││
│  │    - store_data: StorageAgent                           ││
│  │    - calculate_metrics: MetricsAgent                    ││
│  │    - query_data: SheetsReader/KG 조회                   ││
│  │    - run_full_workflow: LegacyOrchestrator              ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ResponsePipeline                          │
│  (모든 응답에 적용)                                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ RAG Context + KG Facts + Tool Results → LLM → 응답      ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        사용자 응답                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 컴포넌트 상세 설계

### 3.1 LLMOrchestrator

**파일 위치**: `core/llm_orchestrator.py`

**책임**:
- 사용자 입력 수신 및 분석
- RAG + KG 컨텍스트 수집 조율
- 신뢰도 기반 처리 전략 결정
- 도구 실행 및 결과 수집
- ResponsePipeline으로 최종 응답 생성 위임

**주요 메서드**:

```python
class LLMOrchestrator:
    async def process(self, user_input: str, session_id: str = None) -> Response
    async def _gather_context(self, query: str, entities: Dict) -> Context
    def _assess_confidence(self, rule_result: Dict) -> ConfidenceLevel
    async def _execute_tools(self, plan: ExecutionPlan) -> ToolResults
    async def _llm_decide(self, query: str, context: Context) -> Decision
```

**의존성**:
- RAGRouter (기존)
- HybridRetriever (기존)
- KnowledgeGraph (기존)
- OrchestratorState (신규)
- ResponseCache (신규)
- ResponsePipeline (신규)

### 3.2 ContextGatherer

**파일 위치**: `core/context_gatherer.py`

**책임**:
- RAG 문서 검색
- KG 엔티티/관계 조회
- 시스템 상태 수집
- 컨텍스트 통합 및 요약

**주요 메서드**:

```python
class ContextGatherer:
    async def gather(self, query: str, entities: Dict) -> Context
    async def _search_rag(self, query: str) -> List[Document]
    def _query_kg(self, entities: Dict) -> List[KGFact]
    def _get_system_state(self) -> SystemState
    def _build_summary(self, context: Context) -> str
```

**컨텍스트 구조**:

```python
@dataclass
class Context:
    query: str
    entities: Dict[str, List[str]]
    rag_docs: List[Document]
    kg_facts: List[KGFact]
    kg_inferences: List[InferenceResult]
    system_state: SystemState
    summary: str
    gathered_at: datetime
```

### 3.3 신뢰도 평가 (ConfidenceAssessor)

**파일 위치**: `core/confidence.py`

**신뢰도 레벨 정의**:

| 레벨 | 절대 점수 | 의미 | 처리 방식 |
|------|----------|------|----------|
| HIGH | 5.0+ | 매우 확실 | Rule 결과 신뢰, LLM 판단 스킵 |
| MEDIUM | 3.0~4.9 | 확실 | LLM에게 도구 선택만 위임 |
| LOW | 1.5~2.9 | 불확실 | LLM에게 전체 판단 위임 |
| UNKNOWN | <1.5 | 알 수 없음 | 사용자에게 명확화 요청 |

**점수 계산 기준** (RAGRouter 기존 로직):

| 매칭 유형 | 점수 | 예시 |
|----------|------|------|
| 키워드 | +2.0 | "뭐야", "해석", "분석" |
| 지표명 | +1.5 | "sos", "hhi", "cpi" |
| 엔티티 | +1.5 | "라네즈", "laneige" |
| 패턴 | +1.0 | "높으면", "낮으면" |

**예시**:

```
"SoS가 뭐야?"
  = 키워드("뭐야") 2.0 + 지표("sos") 1.5
  = 3.5점 → MEDIUM

"SoS 정의와 해석 알려줘"
  = 키워드("정의") 2.0 + 키워드("해석") 2.0 + 지표("sos") 1.5
  = 5.5점 → HIGH

"잘 되고 있어?"
  = 패턴 매칭 없음
  = 0점 → UNKNOWN
```

### 3.4 도구 정의 (AgentTools)

**파일 위치**: `core/tools.py`

**도구 목록**:

| 도구명 | 설명 | 래핑 대상 | 호출 조건 |
|--------|------|----------|----------|
| `crawl_amazon` | Amazon 크롤링 | CrawlerAgent | 데이터 오래됨/명시적 요청 |
| `store_data` | 데이터 저장 | StorageAgent | 크롤링 후 저장 필요 |
| `calculate_metrics` | 지표 계산 | MetricsAgent | 새 데이터로 재계산 |
| `query_data` | 데이터 조회 | SheetsReader/KG | 팩트 확인 필요 |
| `run_full_workflow` | 전체 워크플로우 | LegacyOrchestrator | 전체 실행 요청 |

**OpenAI Function Calling 형식**:

```python
AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "crawl_amazon",
            "description": "Amazon 베스트셀러 데이터를 크롤링합니다. 데이터가 오래되었거나 사용자가 명시적으로 요청할 때 사용합니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "크롤링할 카테고리 목록. 비어있으면 전체 카테고리."
                    }
                },
                "required": []
            }
        }
    },
    # ... 나머지 도구들
]
```

### 3.5 ResponsePipeline

**파일 위치**: `core/response_pipeline.py`

**책임**:
- 모든 응답에 RAG + KG 컨텍스트 적용
- 도구 실행 결과 통합
- 최종 LLM 응답 생성

**주요 메서드**:

```python
class ResponsePipeline:
    async def generate(
        self,
        query: str,
        context: Context,
        tool_results: Optional[ToolResults] = None,
        decision: Optional[Decision] = None
    ) -> Response

    def _build_system_prompt(self) -> str
    def _build_user_prompt(self, query: str, context: Context, ...) -> str
    async def _call_llm(self, messages: List[Dict]) -> str
```

**프롬프트 구조**:

```
System Prompt:
- 역할 정의 (LANEIGE Amazon 분석 전문가)
- 온톨로지 엔티티 설명
- 응답 가이드라인

User Prompt:
- 사용자 질문
- 추출된 엔티티
- RAG 참조 문서
- KG 사실/추론
- 도구 실행 결과 (있으면)
- 이전 대화 기록 (있으면)
```

### 3.6 OrchestratorState

**파일 위치**: `core/state.py`

**관리 상태**:

```python
@dataclass
class OrchestratorState:
    last_crawl_time: Optional[datetime] = None
    data_freshness: str = "unknown"  # fresh, stale, unknown
    kg_initialized: bool = False
    kg_triple_count: int = 0
    current_session_id: Optional[str] = None
    active_tools: List[str] = field(default_factory=list)

    def is_crawl_needed(self) -> bool:
        """오늘 크롤링 필요 여부"""
        if not self.last_crawl_time:
            return True
        return self.last_crawl_time.date() < datetime.now().date()

    def mark_crawled(self):
        """크롤링 완료 표시"""
        self.last_crawl_time = datetime.now()
        self.data_freshness = "fresh"
```

### 3.7 ResponseCache

**파일 위치**: `core/cache.py`

**캐싱 전략**:

| 캐시 유형 | TTL | 키 생성 방식 |
|----------|-----|-------------|
| query | 24시간 | MD5(query.lower().strip()) |
| crawl | 당일 | "crawl_{date}" |
| kg | 1시간 | "kg_{entity}_{query_type}" |

**구현**:

```python
class ResponseCache:
    def __init__(self):
        self._cache: Dict[str, CacheEntry] = {}
        self._ttl = {
            "query": timedelta(hours=24),
            "crawl": timedelta(days=1),
            "kg": timedelta(hours=1)
        }

    def get(self, key: str, cache_type: str = "query") -> Optional[Any]
    def set(self, key: str, value: Any, cache_type: str = "query")
    def invalidate(self, pattern: str = None)

    @staticmethod
    def hash_query(query: str) -> str:
        return hashlib.md5(query.lower().strip().encode()).hexdigest()
```

---

## 4. 데이터 흐름

### 4.1 시나리오 1: 단순 질문 (HIGH 신뢰도)

```
사용자: "SoS 정의 알려줘"
         │
         ▼
┌─ LLMOrchestrator ─────────────────────────────────────────┐
│  1. RAGRouter.route() → DEFINITION, score=5.5           │
│  2. ConfidenceAssessor → HIGH                           │
│  3. ContextGatherer.gather()                            │
│     - RAG: "Strategic Indicators Definition.md" 검색    │
│     - KG: 관련 엔티티 없음 (개념 질문)                   │
│  4. 도구 호출 스킵 (HIGH 신뢰도)                        │
└──────────────────────────────────────────────────────────┘
         │
         ▼
┌─ ResponsePipeline ────────────────────────────────────────┐
│  RAG 컨텍스트 + 질문 → LLM → "SoS(Share of Shelf)는..."  │
└──────────────────────────────────────────────────────────┘
```

### 4.2 시나리오 2: 분석 질문 (MEDIUM 신뢰도)

```
사용자: "LANEIGE 경쟁사 대비 어떤 상황이야?"
         │
         ▼
┌─ LLMOrchestrator ─────────────────────────────────────────┐
│  1. RAGRouter.route() → ANALYSIS, score=3.5             │
│  2. ConfidenceAssessor → MEDIUM                         │
│  3. ContextGatherer.gather()                            │
│     - RAG: "경쟁 분석 가이드" 검색                       │
│     - KG: LANEIGE 메타데이터 (SoS 12%, 순위 25)         │
│     - KG: 경쟁사 관계 (COSRX, TIRTIR)                   │
│     - State: last_crawl=오늘, freshness=fresh           │
│  4. LLM 판단 요청 (컨텍스트 포함)                       │
│     → LLM: "KG에 데이터 있음, direct_answer"            │
│  5. 도구 호출 없음                                       │
└──────────────────────────────────────────────────────────┘
         │
         ▼
┌─ ResponsePipeline ────────────────────────────────────────┐
│  RAG + KG 컨텍스트 → LLM → "LANEIGE는 경쟁사 대비..."   │
└──────────────────────────────────────────────────────────┘
```

### 4.3 시나리오 3: 크롤링 요청

```
사용자: "오늘 데이터 크롤링 해줘"
         │
         ▼
┌─ LLMOrchestrator ─────────────────────────────────────────┐
│  1. RAGRouter.route() → DATA_QUERY, score=4.0           │
│  2. ConfidenceAssessor → MEDIUM                         │
│  3. ContextGatherer.gather()                            │
│     - State: last_crawl=어제, freshness=stale           │
│  4. LLM 판단 요청                                       │
│     → LLM: "크롤링 요청, crawl_amazon 호출"             │
│  5. 도구 실행: crawl_amazon()                           │
│     → CrawlerAgent.execute()                            │
│     → 500개 제품 수집                                   │
│  6. State.mark_crawled()                                │
└──────────────────────────────────────────────────────────┘
         │
         ▼
┌─ ResponsePipeline ────────────────────────────────────────┐
│  도구 결과 포함 → LLM → "크롤링 완료. 500개 제품..."    │
└──────────────────────────────────────────────────────────┘
```

### 4.4 시나리오 4: 불명확한 질문 (UNKNOWN)

```
사용자: "어떻게 생각해?"
         │
         ▼
┌─ LLMOrchestrator ─────────────────────────────────────────┐
│  1. RAGRouter.route() → UNKNOWN, score=0.5              │
│  2. ConfidenceAssessor → UNKNOWN                        │
│  3. 명확화 요청 반환                                     │
└──────────────────────────────────────────────────────────┘
         │
         ▼
"무엇에 대해 분석해 드릴까요?
 예: LANEIGE 순위, SoS 해석, 경쟁사 비교 등"
```

---

## 5. 파일 구조

### 5.1 신규 생성 파일

```
core/
├── __init__.py
├── llm_orchestrator.py      # 메인 오케스트레이터
├── context_gatherer.py      # RAG+KG 컨텍스트 수집
├── confidence.py            # 신뢰도 평가
├── tools.py                 # 에이전트 도구 정의
├── response_pipeline.py     # 응답 생성 파이프라인
├── state.py                 # 상태 관리
├── cache.py                 # 응답 캐싱
└── models.py                # 데이터 모델 (Context, Response 등)
```

### 5.2 수정 파일

```
orchestrator.py              # LegacyOrchestrator로 이름 변경
main.py                      # LLMOrchestrator 사용으로 변경
dashboard_api.py             # LLMOrchestrator 연동
```

### 5.3 신규 테스트 파일

```
tests/
├── test_llm_orchestrator.py
├── test_context_gatherer.py
├── test_confidence.py
└── test_response_pipeline.py
```

### 5.4 문서 파일 이동

```
docs/
├── guides/
│   ├── Strategic Indicators Definition.md
│   ├── Metric Interpretation Guide.md
│   ├── Indicator Combination Playbook.md
│   └── Home Page Insight Rules.md
├── reference/
│   ├── _Introduction to Agents_ 핵심 정리.docx
│   └── ... (기타 docx 파일)
└── design/
    └── LLM_ORCHESTRATOR_DESIGN.md  # 이 문서
```

---

## 6. API 변경

### 6.1 기존 API 유지 (하위 호환)

```python
# dashboard_api.py

@app.post("/api/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    # 내부적으로 LLMOrchestrator 사용
    response = await orchestrator.process(
        user_input=request.message,
        session_id=request.session_id
    )
    return ChatResponse(
        response=response.text,
        query_type=response.query_type,
        confidence=response.confidence,
        sources=response.sources,
        suggestions=response.suggestions,
        entities=response.entities
    )
```

### 6.2 신규 내부 API

```python
# LLMOrchestrator 공개 메서드

class LLMOrchestrator:
    async def process(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        context_override: Optional[Context] = None
    ) -> Response

    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict
    ) -> ToolResult

    def get_state(self) -> OrchestratorState

    def clear_cache(self, pattern: Optional[str] = None)
```

---

## 7. 에러 처리 및 Fallback

### 7.1 에러 유형별 처리

| 에러 유형 | 처리 방식 |
|----------|----------|
| LLM API 오류 | Rule-based 결과로 Fallback |
| RAG 검색 실패 | KG 컨텍스트만으로 진행 |
| KG 조회 실패 | RAG 컨텍스트만으로 진행 |
| 도구 실행 실패 | 에러 메시지 포함 응답 |
| 전체 실패 | 기본 Fallback 메시지 |

### 7.2 Fallback 응답

```python
FALLBACK_RESPONSES = {
    "llm_error": "죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요.",
    "no_data": "현재 데이터가 없습니다. 크롤링을 먼저 실행해주세요.",
    "unknown_query": "질문을 이해하지 못했습니다. 다르게 표현해 주시겠어요?",
    "tool_error": "작업 중 오류가 발생했습니다: {error_message}"
}
```

---

## 8. 성능 고려사항

### 8.1 응답 시간 목표

| 시나리오 | 목표 응답 시간 |
|----------|---------------|
| 캐시 히트 | < 100ms |
| HIGH 신뢰도 (도구 없음) | < 2초 |
| MEDIUM 신뢰도 (도구 없음) | < 3초 |
| 도구 실행 포함 | < 30초 (크롤링 제외) |

### 8.2 비용 최적화

| 전략 | 예상 절감 |
|------|----------|
| 캐싱 | 동일 질문 100% 절감 |
| HIGH 신뢰도 LLM 스킵 | ~30% 호출 절감 |
| 컨텍스트 압축 | ~20% 토큰 절감 |

### 8.3 토큰 사용량 예상

| 컴포넌트 | 입력 토큰 | 출력 토큰 |
|----------|----------|----------|
| LLM 판단 | ~800 | ~100 |
| 응답 생성 | ~1200 | ~500 |
| **합계** | ~2000 | ~600 |

---

## 9. 테스트 계획

### 9.1 단위 테스트

| 테스트 대상 | 테스트 케이스 |
|------------|--------------|
| ConfidenceAssessor | 점수별 신뢰도 레벨 매핑 |
| ContextGatherer | RAG/KG 결과 통합 |
| ResponseCache | TTL, 캐시 히트/미스 |
| OrchestratorState | 상태 전이, 신선도 판단 |

### 9.2 통합 테스트

| 시나리오 | 검증 항목 |
|----------|----------|
| 단순 질문 | HIGH 신뢰도 → 도구 스킵 → 응답 |
| 분석 질문 | MEDIUM → LLM 판단 → 응답 |
| 크롤링 요청 | 도구 실행 → 상태 업데이트 |
| 불명확 질문 | UNKNOWN → 명확화 요청 |
| LLM 오류 | Fallback 응답 |

### 9.3 성능 테스트

- 응답 시간 측정
- 동시 요청 처리
- 캐시 효과 검증

---

## 10. 구현 일정

### Phase 1: 핵심 컴포넌트 (1단계)

| 순서 | 파일 | 설명 |
|------|------|------|
| 1 | `core/models.py` | 데이터 모델 정의 |
| 2 | `core/confidence.py` | 신뢰도 평가 |
| 3 | `core/cache.py` | 캐싱 레이어 |
| 4 | `core/state.py` | 상태 관리 |

### Phase 2: 컨텍스트 및 도구 (2단계)

| 순서 | 파일 | 설명 |
|------|------|------|
| 5 | `core/context_gatherer.py` | RAG+KG 컨텍스트 수집 |
| 6 | `core/tools.py` | 에이전트 도구 래핑 |
| 7 | `core/response_pipeline.py` | 응답 생성 |

### Phase 3: 오케스트레이터 및 통합 (3단계)

| 순서 | 파일 | 설명 |
|------|------|------|
| 8 | `core/llm_orchestrator.py` | 메인 오케스트레이터 |
| 9 | `orchestrator.py` | LegacyOrchestrator 수정 |
| 10 | `main.py` | 진입점 수정 |
| 11 | `dashboard_api.py` | API 연동 |

### Phase 4: 테스트 및 문서화 (4단계)

| 순서 | 파일 | 설명 |
|------|------|------|
| 12 | `tests/test_*.py` | 테스트 작성 |
| 13 | 문서 파일 이동 | `docs/` 폴더 정리 |

---

## 11. 리스크 및 완화 방안

| 리스크 | 완화 방안 | 구현 여부 |
|--------|----------|----------|
| LLM 판단 오류 | 2단계 라우팅 (Rule → LLM) | ✅ 설계 반영 |
| API 비용 증가 | 캐싱, HIGH 신뢰도 LLM 스킵 | ✅ 설계 반영 |
| 기존 코드 호환성 | LegacyOrchestrator 보존 | ✅ 설계 반영 |
| 응답 품질 저하 | 항상 RAG+KG 컨텍스트 사용 | ✅ 설계 반영 |
| 테스트 복잡도 | 단위/통합 테스트 계획 | ✅ 설계 반영 |

---

## 12. 승인 및 서명

- **설계자**: AI Agent 설계 전문가
- **검토자**: (사용자 승인 대기)
- **승인일**: (승인 후 기록)

---

## 부록 A: 용어 정의

| 용어 | 정의 |
|------|------|
| RAG | Retrieval-Augmented Generation, 검색 증강 생성 |
| KG | Knowledge Graph, 지식 그래프 |
| SoS | Share of Shelf, 진열 점유율 |
| HHI | Herfindahl-Hirschman Index, 시장 집중도 지수 |
| CPI | Category Price Index, 카테고리 가격 지수 |

## 부록 B: 참조 문서

- `PROJECT_PLAN.md`: 기존 프로젝트 계획
- `Strategic Indicators Definition.md`: 지표 정의
- `Metric Interpretation Guide.md`: 지표 해석 가이드
