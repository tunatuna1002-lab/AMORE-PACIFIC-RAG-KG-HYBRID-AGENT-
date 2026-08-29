# prompts/ — 프롬프트 중앙 관리

> AI 에이전트용 모듈 가이드 (2026-08 실제 구조 기준으로 재작성)

## 구조

| 파일 | 역할 |
|------|------|
| `registry.py` | **PromptRegistry** — 시스템 프롬프트 로딩/치환의 단일 진입점 (`get_instance()` 싱글턴) |
| `agents/chatbot_system.txt` | 챗봇 시스템 프롬프트 (registry가 로드하는 실제 파일) |
| `agents/insight_system.txt` | 인사이트 생성 시스템 프롬프트 |
| `agents/period_insight_system.txt` | 기간별 인사이트 프롬프트 |
| `agents/react_system.txt` | ReAct 에이전트 프롬프트 |
| `agents/variants/chatbot_system_v0~v4.txt` | 프롬프트 실험 변형 (`docs/experiments/prompt_exp_2026-08.md` 참조) |
| `components/__init__.py` | 공유 컴포넌트: 날짜 컨텍스트, 가드레일(보안 규칙 + 환각 방지) |
| `version_manager.py` | 프롬프트 버전 관리 |
| `metrics.json` | 지표 설명 데이터 |

## 사용 방법 (실제 코드 경로)

```python
from prompts.registry import PromptRegistry

registry = PromptRegistry.get_instance()
system_prompt = registry.get_system_prompt(
    "chatbot", include_guardrails=True, data_date="2026-08-30"
)
```

- 호출처: `src/agents/hybrid_chatbot_agent.py`, `src/agents/hybrid_insight_agent.py`
  (`ContextBuilder.build_system_prompt()` → feature flag `prompts.use_centralized_prompts` 분기)
- `{current_date}` / `{guardrails}` 플레이스홀더는 registry가 치환하고,
  남은 `{placeholder}`는 자동 제거됨 (`registry.py`)

## 주의

- **v4 챗 경로(`/api/v4/chat*`)는 registry를 거치지 않음** —
  `src/core/response_pipeline.py`의 `SYSTEM_PROMPT` 클래스 상수를 사용 (통합 예정 항목).
- 과거 문서에 있던 `chat_system.txt`, `insight_generation.txt`, `query_router.txt`는
  어떤 코드에서도 참조되지 않아 2026-08-30 삭제됨. 챗봇 프롬프트를 수정하려면
  `agents/chatbot_system.txt`를 수정할 것.
