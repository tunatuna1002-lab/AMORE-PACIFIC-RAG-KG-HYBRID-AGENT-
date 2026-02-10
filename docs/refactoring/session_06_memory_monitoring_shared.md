# Session 6: Memory + Monitoring + Shared 정리

> ⏱ 예상 시간: 20~30분 | 위험도: 🟢 낮음 | 선행 조건: Session 2 완료

---

## 프롬프트 (아래를 복사해서 새 Claude Code 세션에 붙여넣기)

```
너는 20년 베테랑 Python 개발자야. AMORE RAG-KG Hybrid Agent의 유틸리티 모듈(memory, monitoring, shared)을 정리하는 세션이야.

## 이번 세션 목표
가장 고립된 3개 모듈(memory, monitoring, shared)을 정리해. 이 모듈들은 내부 의존성이 없거나 최소여서 안전하게 작업 가능.

## 컨텍스트
- 프로젝트: `/Users/leedongwon/Desktop/AMORE-RAG-ONTOLOGY-HYBRID AGENT/`
- Python 3.13.7 (`python3` 사용)
- 의존성:
  - `src/memory/` → 내부 의존 없음. `src/agents/`, `src/core/`가 사용
  - `src/monitoring/` → 내부 의존 없음. `src/agents/`, `src/core/`, `src/shared/`가 사용
  - `src/shared/` → `src/monitoring/`만 의존. `src/agents/`가 사용

## 현재 구조
```
src/memory/          # 636줄
├── __init__.py
├── context.py       # 컨텍스트 관리
├── history.py       # 대화 이력
└── session.py       # 세션 관리

src/monitoring/      # 865줄
├── __init__.py
├── logger.py        # 로깅
├── metrics.py       # 성능 메트릭 수집
└── tracer.py        # 트레이싱

src/shared/          # 387줄
├── __init__.py
├── constants.py     # 상수 정의
└── llm_client.py    # LiteLLM 클라이언트
```

## 수행할 작업

### 1. Memory 모듈 검토
- `context.py`, `history.py`, `session.py`를 읽고:
  - 실제 사용되는지 확인 (grep으로 import 추적)
  - 중복/미사용 코드 제거
  - 타입 힌트 보강
  - `src/domain/interfaces/`에 대응하는 Protocol이 필요하면 메모 (추가는 하지 마)

### 2. Monitoring 모듈 검토
- `logger.py` — 로깅 설정이 적절한지 확인
- `metrics.py` — 메트릭 수집 구조 확인
- `tracer.py` — 트레이싱 구조 확인
- 불필요한 코드 제거, 타입 힌트 보강

### 3. Shared 모듈 검토
- `constants.py` — 상수가 적절한 위치에 있는지 확인
  - domain 관련 상수 → `src/domain/`으로 이동해야 할 수 있음
  - 설정 관련 상수 → `src/infrastructure/config/`로 이동해야 할 수 있음
- `llm_client.py` — LiteLLM 래퍼
  - `src/domain/interfaces/llm_client.py`의 Protocol을 따르는지 확인
  - infrastructure 레이어로 이동해야 하는지 검토 (외부 서비스 래퍼이므로)

### 4. 테스트 작성
- `tests/unit/memory/` — 세션/컨텍스트 관리 테스트
- `tests/unit/monitoring/` — 로거 설정 테스트
- `tests/unit/shared/` — LLM 클라이언트 테스트 (LiteLLM mock)

### 5. 검증
- `python3 -m pytest tests/ -v --tb=short` — 전체 테스트 통과

## 주의사항
- 이 모듈들은 다른 모듈이 의존하므로, 인터페이스(public API)를 바꾸면 안 됨
- 내부 구현만 정리
- llm_client.py의 위치 이동은 이번 세션에서 하지 말 것 (메모만)
```

---

## 체크리스트

- [ ] Memory 모듈 검토 및 정리
- [ ] Monitoring 모듈 검토 및 정리
- [ ] Shared 모듈 검토 (상수 위치, LLM 클라이언트)
- [ ] 테스트 추가
- [ ] 전체 테스트 통과
