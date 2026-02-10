# Session 8: Agents 리팩토링 + 순환 의존성 해소

> ⏱ 예상 시간: 50~70분 | 위험도: 🔴 높음 | 선행 조건: Session 7 완료

---

## 프롬프트 (아래를 복사해서 새 Claude Code 세션에 붙여넣기)

```
너는 20년 베테랑 Python 개발자이자 AI Agent 아키텍처 전문가야. AMORE RAG-KG Hybrid Agent의 agents 모듈을 리팩토링하는 세션이야.

## 이번 세션 목표
1. `src/agents/` 중복 파일 통합
2. `src/agents/ ↔ src/core/` 순환 의존성 해소
3. Agent들이 Domain Protocol을 구현하도록 정리

## 컨텍스트
- 프로젝트: `/Users/leedongwon/Desktop/AMORE-RAG-ONTOLOGY-HYBRID AGENT/`
- 전체 마스터 플랜: `docs/refactoring/00_MASTER_PLAN.md` 참조
- Session 7에서 tools/ 분할 및 일부 순환 의존성이 해소되었음
- Python 3.13.7 (`python3` 사용)

## 현재 구조 & 문제점
```
src/agents/
├── hybrid_chatbot_agent.py       # 1,624줄 — 챗봇
├── hybrid_insight_agent.py       # 1,341줄 — 인사이트
├── true_hybrid_insight_agent.py  # 905줄 — ↑와 중복!
├── period_insight_agent.py       # 1,113줄 — 기간별 인사이트
├── crawler_agent.py              # 크롤러
├── metrics_agent.py              # 메트릭
├── storage_agent.py              # 저장소
├── alert_agent.py                # 알림
└── __init__.py
```

총 5,684줄. 문제:
1. `hybrid_insight_agent.py` vs `true_hybrid_insight_agent.py` — **반드시 통합**
2. `hybrid_chatbot_agent.py` (1624줄) — God Object
3. `agents ↔ core` 순환: brain이 agents를 import, agents가 core를 import

## 수행할 작업 (TDD 방식)

### 1. 중복 통합: insight agents
- `hybrid_insight_agent.py`와 `true_hybrid_insight_agent.py`를 비교
- 어디서 import되는지 추적
- 하나로 통합. 통합된 파일 이름은 `insight_agent.py`로.
- `period_insight_agent.py`와의 관계도 확인 — 별도 유지할지, 통합할지

### 2. Protocol 구현 적용
Session 2에서 만든 Protocol을 각 Agent가 구현하도록:

```python
# src/agents/insight_agent.py
from src.domain.interfaces.insight import InsightAgentProtocol

class InsightAgent(InsightAgentProtocol):
    """InsightAgentProtocol 구현체"""
    ...
```

각 Agent에 대해:
- `hybrid_chatbot_agent.py` → `ChatbotAgentProtocol` 구현
- `insight_agent.py` (통합) → `InsightAgentProtocol` 구현
- `crawler_agent.py` → `CrawlerAgentProtocol` 구현
- `alert_agent.py` → `AlertAgentProtocol` 구현
- `metrics_agent.py` → 역할 확인 후 Protocol 매핑
- `storage_agent.py` → `StorageProtocol`과 관계 확인

### 3. 순환 의존성 해소: agents ↔ core

현재:
```python
# src/core/brain.py (현재)
from src.agents.hybrid_chatbot_agent import HybridChatbotAgent  # 순환!
```

변경 후:
```python
# src/core/brain.py (변경 후)
from src.domain.interfaces.chatbot import ChatbotAgentProtocol  # Protocol 사용

class UnifiedBrain:
    def __init__(self, chatbot: ChatbotAgentProtocol, ...):
        self.chatbot = chatbot  # DI로 주입
```

- `src/core/`에서 `src/agents/`를 직접 import하는 모든 곳을 찾아서 Protocol로 대체
- `src/agents/`에서 `src/core/`를 import하는 곳도 찾아서 필요하면 Protocol로 대체

### 4. chatbot_agent.py 분할 검토
1624줄이면 분할 고려:
- 쿼리 분석 로직 → Session 3의 `application/services/query_analyzer.py`로 이미 분리됨
- 프롬프트 구성 로직 → `prompts/`로 분리 가능
- 대화 이력 관리 → `src/memory/`와 연동
- 핵심 채팅 로직만 남기기

### 5. 테스트
- `tests/unit/agents/` 보강
- 통합된 insight_agent 테스트
- Protocol 구현 검증 테스트:
  ```python
  def test_insight_agent_implements_protocol():
      assert isinstance(InsightAgent(...), InsightAgentProtocol)
  ```
- 순환 의존성 검증

### 6. 검증
- `python3 -m pytest tests/unit/agents/ -v` — agents 테스트 통과
- `python3 -m pytest tests/ -v --tb=short` — 전체 테스트 통과
- 순환 의존성 검증 스크립트 실행

## 주의사항
- `hybrid_insight_agent.py` 삭제 전에 어디서 import하는지 반드시 확인
- 통합 후 기존 import 경로 호환: `__init__.py`에서 re-export
- brain.py 수정은 최소한으로 (Protocol import 변경만)
- LLM 호출 부분은 반드시 mock
```

---

## 체크리스트

- [ ] insight agent 중복 분석 및 통합
- [ ] 각 Agent에 Protocol 구현 적용
- [ ] agents ↔ core 순환 의존성 해소
- [ ] chatbot_agent.py 분할 검토
- [ ] 테스트 보강
- [ ] 순환 의존성 검증 통과
- [ ] 전체 테스트 통과
