# Session 1: 안전망 테스트 작성

> ⏱ 예상 시간: 40~60분 | 위험도: 🟢 낮음 | 선행 조건: Session 0 완료

---

## 프롬프트 (아래를 복사해서 새 Claude Code 세션에 붙여넣기)

```
너는 20년 베테랑 Python 개발자이자 TDD 전문가야. AMORE RAG-KG Hybrid Agent의 대대적인 리팩토링을 앞두고 "안전망 테스트"를 작성하는 세션이야.

## 이번 세션 목표
리팩토링 전에 핵심 기능이 동작함을 보증하는 안전망(regression) 테스트를 작성해.
이 테스트들은 이후 세션에서 리팩토링할 때 "기존 기능이 깨지지 않았다"를 증명하는 데 쓰여.

## 컨텍스트
- 프로젝트: `/Users/leedongwon/Desktop/AMORE-RAG-ONTOLOGY-HYBRID AGENT/`
- 전체 마스터 플랜: `docs/refactoring/00_MASTER_PLAN.md` 참조
- Python 3.13.7 (`python3` 사용)
- 현재 커버리지: 10.11% (이 세션에서 최소 25%까지 올리는 것이 목표)
- 테스트 실행: `python3 -m pytest tests/ -v`
- LLM/외부 API 호출하는 부분은 반드시 mock 처리

## 작성할 안전망 테스트 (우선순위 순)

### 1. Domain 레이어 (src/domain/) — 가장 쉬움
- `src/domain/entities/market.py` — Product, Category 등 엔티티 생성/검증
- `src/domain/entities/brand.py` — Brand 엔티티
- `src/domain/entities/relations.py` — 관계 정의
- `src/domain/exceptions.py` — 커스텀 예외
- `src/domain/interfaces/` — Protocol 클래스 (인스턴스화 테스트는 불필요, 서브클래스 검증)
- 위치: `tests/unit/domain/`

### 2. Ontology 레이어 (src/ontology/)
- `knowledge_graph.py` — KG 로드, 트리플 추가/검색, 저장 (파일 I/O mock)
- `reasoner.py` — 추론 결과 검증
- `business_rules.py` — 규칙 평가
- 위치: `tests/unit/ontology/`

### 3. RAG 레이어 (src/rag/)
- `hybrid_retriever.py` — 검색 파이프라인 (ChromaDB + KG mock)
- `retriever.py` — 기본 검색
- `reranker.py` — 재순위화
- `chunker.py` — 텍스트 분할
- 위치: `tests/unit/rag/`

### 4. Core 기능 (src/core/)
- `brain.py` — 스케줄러의 핵심 메서드들 (query routing, complexity detection)
- `react_agent.py` — ReAct 루프 (LLM mock)
- 위치: `tests/unit/core/`

### 5. 핵심 Tools (src/tools/)
- `metric_calculator.py` — SoS, HHI, CPI 계산 (순수 함수, mock 불필요)
- `sqlite_storage.py` — DB CRUD (in-memory SQLite)
- 위치: `tests/unit/tools/`

### 6. API 엔드포인트 (최소한)
- `GET /api/health` → 200 반환
- `GET /api/data` → 정상 응답
- `POST /api/v3/chat` → mock LLM으로 응답 구조 검증
- 위치: `tests/unit/api/`

## 테스트 작성 원칙
1. **외부 I/O 전부 mock**: OpenAI, ChromaDB, Playwright, Google Sheets, SQLite(파일), HTTP
2. **각 테스트 독립적**: 순서 의존 없음
3. **한 테스트 한 관심사**: 하나만 검증
4. **네이밍**: `test_{기능}_{시나리오}_{기대결과}` 형식
5. **fixture 활용**: `conftest.py`에 공통 fixture 정의

## 검증
- `python3 -m pytest tests/ -v --tb=short` — 전체 통과
- `python3 -m pytest tests/ --cov=src --cov-report=term-missing` — 커버리지 확인
- 목표: 최소 25% 커버리지

## 주의사항
- 기존 코드를 수정하지 마. 테스트만 추가해.
- 기존 `tests/` 안의 테스트와 충돌하지 않게 해.
- 이미 존재하는 테스트 파일은 보강만 해 (덮어쓰지 마).
```

---

## 이 세션의 체크리스트

- [ ] Domain 엔티티 테스트 작성
- [ ] Ontology 핵심 테스트 작성
- [ ] RAG 파이프라인 테스트 작성
- [ ] Core (brain, react) 테스트 작성
- [ ] Tools (metric_calculator) 테스트 작성
- [ ] API 엔드포인트 테스트 작성
- [ ] 전체 테스트 통과 확인
- [ ] 커버리지 25%+ 달성

## 예상 결과
- 테스트 파일 10~15개 추가
- 커버리지 10% → 25%+
- 리팩토링 안전망 확보
