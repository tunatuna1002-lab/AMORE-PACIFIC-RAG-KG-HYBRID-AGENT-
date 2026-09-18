# ReAct Self-Reflection Agent Guide

> **[2026-09 사후] 이 가이드의 도구명·활성화 조건·반복 기본값은 트랙 5-C 리팩터 이전 기준이다.**
> `src/core/react_agent.py`(`ReActAgent`/`get_react_agent`/`set_tool_executor` 등 API 자체는
> 여전히 존재한다) 아래에서 실제로 바뀐 부분:
> - **도구**: 아래 문서 곳곳의 `query_data`·`query_knowledge_graph`·`calculate_metrics`는
>   더 이상 존재하지 않는다. ReAct 전용 도구 3종·대시보드 JSON 도구 5종이 폐지되고,
>   DecisionMaker와 공유하는 5종 레지스트리(`resolve_entity`·`kg_neighbors`·`get_metrics`·
>   `apply_rules`·`search_docs`, `src/core/tool_registry.py`) + 루프 제어 2종
>   (`final_answer`·`refine_search`)만 남았다.
> - **활성화**: "분석/비교/왜" 같은 키워드 기반 복잡도 판단이 아니라, 홉 카운트 라우터
>   (`src/core/router.py`, `hops >= HOP_THRESHOLD(2)`)가 먼저 판단하고 플래그
>   `agents.use_react_agent`(기본 OFF)가 켜져 있어야 실제로 ReAct 경로를 탄다(OFF 상태에서는
>   `agents.react_shadow_mode`로 그림자 실행만 가능). 단, 신뢰도가 HIGH면 홉 수와 무관하게 파이프라인이 답한다 — ReAct는 신뢰도 MEDIUM/LOW + 2홉 이상일 때만 탄다(측정용 플래그 `agents.react_bypass_confidence`, 기본 OFF, 로 이 관문을 건너뛸 수 있다). [2026-09-18 사후] 6단계 비교 결과 세 플래그 모두 기본 OFF 유지(`docs/plans/evidence-react-ontology-decisions-2026-09.md` S6-3).
> - **호출 방식**: 프롬프트에 JSON을 요구해 직접 파싱하던 방식에서 네이티브 function
>   calling(`tools=`/`tool_choice="auto"`)으로 바뀌었다.
> - **반복·예산**: `max_iterations` 기본값은 3이 아니라 5이며, 질문당 토큰 예산
>   `max_total_tokens`(기본 `DEFAULT_TOKEN_BUDGET=12000`)이 새로 추가됐다.
>
> 아래 본문은 개념(Thought→Action→Observation→Reflection 루프, Self-Reflection)은 여전히
> 유효하지만, 도구명·활성화 조건·기본 반복 횟수가 나오는 예시는 위 내용을 기준으로 읽어야 한다.

## 개요

ReAct (Reasoning + Acting) Self-Reflection 패턴을 구현한 AI 에이전트로, 복잡한 질문에 대해 단계적 사고와 자체 평가를 통해 고품질 응답을 생성합니다.

## 동작 원리

### ReAct Loop

```
1. Thought (사고)
   ↓
2. Action (행동)
   ↓
3. Observation (관찰)
   ↓
4. Reflection (반성)
   ↓
   반복 (최대 3회)
```

### Self-Reflection

응답 생성 후 자체 품질 평가:
- 질문에 완전히 답변했는가?
- 누락된 중요 정보가 있는가?
- 데이터/근거가 충분한가?

## 사용 시나리오

### 자동 활성화 (UnifiedBrain)

`UnifiedBrain.process_query()`에서 복잡한 질문 자동 감지:

```python
from src.core.brain import get_initialized_brain

brain = await get_initialized_brain()
response = await brain.process_query("LANEIGE가 경쟁사 대비 어떤 위치에 있는지 분석해줘")

# 복잡도 판단 → ReAct 모드 자동 활성화
print(f"Mode: {response.metadata.get('mode')}")  # "react"
print(f"Iterations: {response.metadata.get('iterations')}")
```

### 수동 사용 (Standalone)

```python
from src.core.react_agent import get_react_agent

agent = get_react_agent()
agent.set_tool_executor(tool_executor)

result = await agent.run(
    query="LANEIGE 순위 추이는?",
    context="최근 30일 데이터 있음"
)

print(f"답변: {result.final_answer}")
print(f"신뢰도: {result.confidence}")
print(f"반복 횟수: {result.iterations}")
```

## 복잡도 판단 기준

UnifiedBrain은 다음 조건으로 복잡한 질문 판단:

| 조건 | 예시 |
|------|------|
| 분석적 키워드 | "왜", "어떻게", "비교", "분석", "추천" |
| 컨텍스트 부족 | RAG 문서 < 2개, KG 트리플 없음 |
| 다단계 질문 | "?"가 2개 이상, 접속사 ("그리고", "하지만") |

**복잡한 질문 예시:**
- "LANEIGE가 CeraVe 대비 왜 순위가 낮은지 분석해줘"
- "경쟁사와 비교했을 때 LANEIGE의 전략은 어떻게 개선할 수 있을까?"
- "SoS가 하락한 원인과 해결 방안은?"

**단순한 질문 예시:**
- "LANEIGE 순위 알려줘"
- "오늘 데이터 있어?"
- "LANEIGE ASIN은?"

## ReAct Step 구조

```python
@dataclass
class ReActStep:
    thought: str                       # 현재 상황 분석
    action: str | None = None          # 선택한 도구 (query_data, query_knowledge_graph 등)
    action_input: dict | None = None   # 도구 파라미터
    observation: str | None = None     # 도구 실행 결과
    reflection: str | None = None      # 결과 평가
```

## 설정 파라미터

```python
agent = ReActAgent(
    model="gpt-4o-mini",          # LLM 모델
    max_iterations=3,             # 최대 반복 횟수
    min_confidence=0.7            # 최소 신뢰도 임계값
)
```

## 도구 통합

ReActAgent는 `ToolExecutor`와 연결하여 다음 도구 사용:

| 도구 | 설명 |
|------|------|
| `query_data` | 데이터베이스 조회 |
| `query_knowledge_graph` | 지식 그래프 조회 |
| `calculate_metrics` | 지표 계산 |
| `final_answer` | 최종 답변 (루프 종료) |

```python
# UnifiedBrain의 ToolExecutor 연결
agent.set_tool_executor(brain.tool_executor)
```

## 응답 구조

```python
@dataclass
class ReActResult:
    final_answer: str                  # 최종 답변
    steps: list[ReActStep]             # 실행 단계
    iterations: int                    # 반복 횟수
    confidence: float                  # 신뢰도 (0.0~1.0)
    needs_improvement: bool            # 개선 필요 여부
```

## 로깅 및 디버깅

```python
import logging

# ReAct 실행 로그 확인
logging.getLogger("src.core.react_agent").setLevel(logging.DEBUG)

# 복잡도 판단 로그
logging.getLogger("src.core.brain").setLevel(logging.INFO)
```

**로그 예시:**
```
[INFO] Complex query detected, using ReAct mode: LANEIGE가 경쟁사 대비...
[INFO] 🔧 Tool: query_data | Params: {"query_type": "brand_metrics"}
[INFO] 👁️ Observation: {"brand": "LANEIGE", "rank": 5, "sos": 12.5}
[WARNING] ReAct result needs improvement (confidence: 0.65)
```

## 성능 최적화

### 1. 캐싱

복잡한 질문도 캐싱 가능:

```python
response = await brain.process_query(
    query="LANEIGE 경쟁 분석",
    skip_cache=False  # 캐시 활성화
)
```

### 2. 반복 횟수 조정

간단한 복잡한 질문:
```python
agent = ReActAgent(max_iterations=2)  # 빠른 응답
```

매우 복잡한 질문:
```python
agent = ReActAgent(max_iterations=5)  # 깊은 분석
```

### 3. 모델 선택

빠른 응답:
```python
agent = ReActAgent(model="gpt-4o-mini")  # 기본값
```

고품질 응답:
```python
agent = ReActAgent(model="gpt-4o")  # 더 정확
```

## 에러 처리

```python
try:
    result = await agent.run(query, context)

    if result.needs_improvement:
        logger.warning("낮은 품질 응답 감지")

    if result.confidence < 0.5:
        logger.warning("낮은 신뢰도 - 추가 정보 필요")

except Exception as e:
    logger.error(f"ReAct 실행 실패: {e}")
    # Fallback 로직
```

## 실전 예시

### 예시 1: 순위 분석

```python
query = "LANEIGE 순위가 왜 하락했는지 분석해줘"

# UnifiedBrain이 자동으로 ReAct 모드 활성화
response = await brain.process_query(query)

# Step 1: Thought - "순위 하락 원인 분석 필요"
# Step 2: Action - query_data(query_type="brand_metrics")
# Step 3: Observation - {"rank": 8, "rank_delta": "+3"}
# Step 4: Thought - "경쟁사 동향 확인 필요"
# Step 5: Action - query_data(query_type="competitor_analysis")
# Step 6: Observation - {"CeraVe": {"rank": 2, "deals": "Lightning Deal"}}
# Step 7: Action - final_answer
# Self-Reflection: confidence=0.85, needs_improvement=False
```

### 예시 2: 전략 추천

```python
query = "LANEIGE의 SoS를 높이기 위한 전략을 추천해줘"

response = await brain.process_query(query)

# Step 1: Thought - "현재 SoS 확인"
# Step 2: Action - query_data(query_type="brand_metrics")
# Step 3: Observation - {"sos": 12.5}
# Step 4: Thought - "경쟁사 SoS와 비교 필요"
# Step 5: Action - query_knowledge_graph(entity="LANEIGE", relation_type="competitors")
# Step 6: Observation - {"competitors": ["CeraVe", "Neutrogena"]}
# Step 7: Action - final_answer
# Self-Reflection: confidence=0.72, needs_improvement=True
```

## 제약사항

1. **최대 반복 횟수 제한**: 무한 루프 방지 (기본 5회, `max_iterations` — [2026-09 사후] 정정) + 질문당 토큰 예산(기본 12,000)
2. **도구 의존성**: ToolExecutor에 등록된 도구만 사용 가능
3. **컨텍스트 길이**: 과도한 step 누적 시 토큰 제한 주의
4. **LLM 파싱 오류**: JSON 파싱 실패 시 fallback

## 향후 개선 방향

- [ ] Multi-Agent ReAct (병렬 사고)
- [ ] 학습 기반 반복 횟수 조정
- [ ] 도구 선택 우선순위 학습
- [ ] Reflection 기반 자동 재시도

## 참고 자료

- [ReAct Paper (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
- [Self-Reflection in LLMs](https://arxiv.org/abs/2303.11366)
- `src/core/react_agent.py`: 구현 코드
- `tests/unit/core/test_react_agent.py`: 단위 테스트
- `examples/react_agent_demo.py`: 실행 가능한 데모
