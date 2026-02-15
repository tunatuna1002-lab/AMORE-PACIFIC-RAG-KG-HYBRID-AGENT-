# HANDOFF.md — Phase 1-2 Worktree (jovial-bell)

> **워크트리**: `.claude/worktrees/jovial-bell`
> **브랜치**: `claude/jovial-bell`
> **범위**: Phase 1 (Dead Code 제거) + Phase 2 (Retriever 통합)
> **마지막 업데이트**: 2026-02-15

---

## 1. 완료한 작업

### 계획 수립 (Plan Mode)
- [x] 코드베이스 탐색 (3개 Explore 에이전트 병렬 실행)
- [x] TOP 10 리팩토링 대상 식별
- [x] 6-Phase 리팩토링 계획 작성 (`~/.claude/plans/snoopy-floating-pie.md`)
- [x] Claude Code 최신 기능 딥 리서치 (tmux, Agent Teams, Worktrees, CLAUDE.md, Skills, Subagents)
- [x] Phase 간 게이트 조건 + 롤백 기준 추가
- [x] Memory 엔드포인트 분리 검토 (결론: chat.py에 합침)
- [x] 기능 → Phase 매핑 테이블 작성

### 아직 코드 변경 없음
- Plan 모드에서 계획 수립만 완료
- 실제 리팩토링 구현은 아직 시작하지 않음

---

## 2. 다음에 해야 할 작업

### Phase 1: Dead Code 제거 (P0, 1일, LOW risk)

| # | 대상 | 경로 | 작업 |
|---|------|------|------|
| 1-1 | TrueHybridInsightAgent | `src/agents/true_hybrid_insight_agent.py` | 파일 삭제 |
| 1-2 | SuggestionGenerator | `src/agents/suggestion_generator.py` | 파일 삭제 |
| 1-3 | ContextBuilder Old | `src/rag/context_builder_old.py` | 파일 삭제 |
| 1-4 | Import 업데이트 | 삭제 모듈의 모든 import | grep → 새 경로로 교체 |

**검증**: `ruff check src/` → 0 err + `python3 -m pytest tests/ -x --tb=short` → 95%+

### Phase 2: Retriever 통합 (P0, 3일, HIGH risk)

1. `UnifiedRetrievalResult`를 `src/domain/value_objects/`로 이동
2. `HybridRetriever`에 Strategy 패턴: `RetrievalStrategy` (legacy/owl)
3. `TrueHybridRetriever` 로직 → `OWLRetrievalStrategy`로 추출
4. `UnifiedRetriever` facade 제거
5. 모든 consumer 업데이트 (chatbot, insight agent 등)
6. 통합 테스트 추가

**수정 대상**: `src/rag/hybrid_retriever.py`, `src/rag/true_hybrid_retriever.py` (삭제), `src/rag/unified_retriever.py` (삭제), `src/domain/interfaces/retriever.py`, `src/agents/hybrid_chatbot_agent.py`, `src/agents/hybrid_insight_agent.py`, `src/infrastructure/container.py`

**게이트**: Phase 2 완료 → PR main 머지 → Phase 3 시작 가능

---

## 3. 시도했지만 실패한 접근법

- (아직 구현 단계가 아니므로 해당 없음)

---

## 4. 수정한 파일 목록

- (아직 코드 변경 없음 — Plan 모드에서 계획 파일만 편집)

| 파일 | 변경 유형 | 요약 |
|------|-----------|------|
| `~/.claude/plans/snoopy-floating-pie.md` | 생성 + 편집 | 전체 리팩토링 계획 (Part A-C) |
| `~/.claude/plans/snoopy-floating-pie-agent-ad228bc.md` | 생성 | tmux + Agent Teams 리서치 |
| `~/.claude/plans/snoopy-floating-pie-agent-a678a07.md` | 생성 | Worktrees + CLAUDE.md/Skills 리서치 |
| `HANDOFF.md` | 생성 | 이 파일 |

---

## 5. 참조 문서

- **전체 계획**: `~/.claude/plans/snoopy-floating-pie.md`
- **성공 기준**: ruff 0 errors + pytest 통과율 95%+ + 커버리지 60%+
- **롤백 기준**: 검증 실패시 worktree 브랜치 삭제 + 원인 분석 + Plan 수정
