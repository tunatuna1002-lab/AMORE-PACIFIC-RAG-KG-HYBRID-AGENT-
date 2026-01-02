# Deprecated Modules

이 폴더에는 더 이상 사용되지 않는 모듈들이 보관되어 있습니다.

## 이동된 모듈 및 이유

### core/brain.py
- **상태**: Abandoned (Level 4 Autonomous Design)
- **이유**: `unified_orchestrator.py`로 기능 대체됨
- **원래 목적**: 단일 "뇌" 기반 자율 에이전트 시스템
- **대체**: `src/core/unified_orchestrator.py`

### core/decision_maker.py
- **상태**: Replaced
- **이유**: `unified_orchestrator.py`에 기능 통합됨
- **원래 목적**: LLM + 규칙 기반 의사결정 엔진
- **대체**: `src/core/unified_orchestrator.py`

### agents/query_agent.py
- **상태**: Not Used (No imports)
- **이유**: Hybrid agents에 기능 통합됨
- **원래 목적**: 쿼리 처리 에이전트
- **대체**: `src/agents/hybrid_chatbot_agent.py`

### agents/workflow_agent.py
- **상태**: Not Used (No imports)
- **이유**: `orchestrator.py`와 기능 중복
- **원래 목적**: 워크플로우 조율
- **대체**: `orchestrator.py`

## 복원 방법
필요시 각 모듈을 원래 위치로 이동하면 됩니다:
```bash
mv _deprecated/core/brain.py src/core/
mv _deprecated/core/decision_maker.py src/core/
mv _deprecated/agents/query_agent.py src/agents/
mv _deprecated/agents/workflow_agent.py src/agents/
```

## 아키텍처 변경 히스토리
- 2025-01: Level 4 자율 시스템 설계 (brain.py, decision_maker.py)
- 2025-01: Unified Orchestrator로 통합 및 단순화
- 2025-01: 미사용 모듈 정리 및 이동
