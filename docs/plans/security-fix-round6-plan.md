# Round 6: 챗봇 보안 취약점 수정 실행 계획

> **이 파일은 Context Engineering용입니다.**
> 컨텍스트가 초과되어도 이 파일을 읽으면 전체 작업 상태를 파악할 수 있습니다.
> 새 세션에서는 아래 프롬프트를 입력하세요:
> `이 파일을 읽고 이어서 작업하세요: docs/plans/security-fix-round6-plan.md`

---

## 작업 진행 상태 체크리스트

- [x] **Fix 1**: 추천 질문 위험 명령 제거 (`response_pipeline.py`) — 301b0db
- [x] **Fix 2**: SSE 엔드포인트 인증 추가 (`chat.py`) — 75d32ff
- [x] **Fix 3**: PromptGuard 시스템 명령 차단 (`prompt_guard.py`) — 6bef603
- [x] **Fix 4**: LLMOrchestrator 크롤링 자동 실행 제거 (`llm_orchestrator.py`) — 7601628
- [x] **Fix 5**: 테스트 업데이트 (`test_response_pipeline.py`, `test_llm_orchestrator.py`) — 5e1ff21
- [x] **교차 검증**: Fix 3 패턴이 Fix 1 교체 질문을 차단하지 않는지 확인 — 7/7 차단, 15/15 통과
- [x] **통합 테스트**: `python3 -m pytest tests/ -x -v` — 5195 passed, 0 failed, 72.62% coverage
- [x] **E2E 검증**: 대시보드 데이터 연동 + 챗봇 보안 확인 — health ✅, data(5 cat, 5 prod) ✅, SSE 503 ✅
- [x] **커밋**: Fix별 분리 커밋 완료 (5개 커밋)

---

## 배경 (왜 이 작업을 하는가)

대시보드 Round 5 수정 후 챗봇이 정상 작동하지만, 두 가지 보안 문제 발견:
1. 추천 질문에 "지식 그래프 초기화해줘", "최신 데이터 크롤링해줘" 같은 **위험한 시스템 명령**이 표시됨
2. 사용자가 "크롤링해줘"를 채팅으로 입력하면 **실제로 Amazon 크롤링이 실행**됨
3. SSE 스트리밍 엔드포인트(`/api/v4/chat/stream`)에 **인증이 누락**되어 누구나 실행 가능

### 이전 사고 이력
- CSP 헤더 수정 후 대시보드 데이터 연동이 깨진 적 있음
- 따라서 이번 수정에서 **데이터 연동 무결성을 최우선으로 보호**해야 함

---

## 보안 취약점 상세

### 취약점 1: 챗봇으로 크롤링 실행 가능 (🔴 HIGH)

**공격 경로**:
```
POST /api/v4/chat/stream (인증 없음!)
→ brain.process_query_stream()
→ PromptGuard.check_input() → PASS (크롤링은 차단 안 됨)
→ DecisionMaker / LLMOrchestrator._default_decision()
→ "크롤링" 키워드 감지 → Decision(tool="crawl_amazon")
→ ToolCoordinator.execute("crawl_amazon")
→ CrawlerAgent → 실제 Amazon 크롤링 시작!
```

**근거 코드**:
- `src/core/llm_orchestrator.py:454` — 키워드 매칭으로 crawl_amazon 자동 선택
- `src/core/tools.py:350` — crawl_amazon이 AGENT_TOOLS에 실제 등록됨

**보안 계층 분석**:
| 계층 | 차단? | 이유 |
|------|-------|------|
| InputValidator | ❌ | 인젝션 패턴만 차단 |
| PromptGuard | ❌ | 시스템 명령은 비즈니스 요청으로 통과 |
| DecisionMaker | ❌ | 오히려 크롤링 도구를 선택 |
| API Key 인증 | ⚠️ | `/api/v4/chat/stream`만 미적용 |
| Rate Limiter | ✅ | 10/min — 남용 제한은 되지만 단발 실행은 못 막음 |

### 취약점 2: SSE 엔드포인트 인증 누락 (🔴 HIGH)

- `src/api/routes/chat.py:219` — `/api/v4/chat/stream`에 `verify_api_key` 없음
- `/api/chat`(line 39)과 `/api/v4/chat`(line 147)에는 있음

### 취약점 3: 추천 질문에 위험한 시스템 명령 (🟡 MEDIUM)

- `src/core/response_pipeline.py:517` — "지식 그래프 초기화해줘"
- `src/core/response_pipeline.py:519` — "최신 데이터 크롤링해줘"
- `src/core/response_pipeline.py:524` — 폴백에 "오늘 크롤링 해줘"

---

## Fix 1: 추천 질문 위험 명령 제거

**파일**: `src/core/response_pipeline.py`
**라인**: 515-524
**담당**: Developer C

### 변경 전 (line 515-524):
```python
        # 시스템 상태 기반 제안
        if context.system_state:
            if not context.system_state.kg_initialized:
                suggestions.append("지식 그래프 초기화해줘")
            if context.system_state.data_freshness != "fresh":
                suggestions.append("최신 데이터 크롤링해줘")

        # 기본 제안
        if not suggestions:
            suggestions = ["라네즈 현재 순위 알려줘", "SoS가 뭐야?", "오늘 크롤링 해줘"]
```

### 변경 후:
```python
        # 시스템 상태 기반 제안 (분석 질문만 — 시스템 명령 제외)
        if context.system_state:
            if not context.system_state.kg_initialized:
                suggestions.append("LANEIGE 브랜드 현황 요약해줘")
            if context.system_state.data_freshness != "fresh":
                suggestions.append("최근 시장 점유율 변화 분석해줘")

        # 기본 제안 (분석 질문만)
        if not suggestions:
            suggestions = ["라네즈 현재 순위 알려줘", "SoS가 뭐야?", "오늘 주요 인사이트는?"]
```

### 검증:
```bash
python3 -m pytest tests/unit/core/test_response_pipeline.py -v -k "suggestion"
```

---

## Fix 2: SSE 스트리밍 엔드포인트 인증 추가

**파일**: `src/api/routes/chat.py`
**라인**: 219
**담당**: Developer A

### 변경 전 (line 219):
```python
@router.post("/api/v4/chat/stream")
@limiter.limit("10/minute")
async def chat_v4_stream(request: Request, body: BrainChatRequest):
```

### 변경 후:
```python
@router.post("/api/v4/chat/stream", dependencies=[Depends(verify_api_key)])
@limiter.limit("10/minute")
async def chat_v4_stream(request: Request, body: BrainChatRequest):
```

### 프론트엔드 호환성 (확인 완료):
- `dashboard/amore_unified_dashboard_v4.html:9351` — `headers: getApiHeaders()` 사용
- `line 8497-8503` — `getApiHeaders()`는 API_KEY 있으면 `X-API-Key` 헤더 포함
- `verify_api_key` 동작 (`src/api/dependencies.py:41-59`):
  - Production: API_KEY 필수 → 인증 정상 동작
  - Development: API_KEY=None → 503 (기존 /api/chat, /api/v4/chat과 동일)
- **결론: 기존 동작에 영향 없음** ✅

### ⚠️ 데이터 연동 체크:
- `/api/data` (대시보드 데이터)는 수정하지 않음
- `/api/health` (헬스체크)는 수정하지 않음
- `src/api/routes/data.py` **절대 수정 금지**
- `src/api/routes/health.py` **절대 수정 금지**

### 검증:
```bash
python3 -m pytest tests/unit/api/ -v -k "chat"
# 데이터 연동 무결성:
python3 -m pytest tests/unit/api/ -v -k "data or health"
```

---

## Fix 3: PromptGuard 시스템 명령 차단 패턴 추가

**파일**: `src/core/prompt_guard.py`
**담당**: Developer B

### 3-1: SYSTEM_COMMAND_PATTERNS 리스트 추가

`INJECTION_PATTERNS` 다음(line 65 근처)에 새 클래스 변수 추가:

```python
    # 시스템 명령 차단 패턴 (크롤링, 초기화 등 — 챗봇에서 실행 방지)
    SYSTEM_COMMAND_PATTERNS = [
        r"(?i)(크롤링|crawl|scrape|스크래핑)\s*(해줘|시작|실행|해\b|하자|해봐|go|start|run)",
        r"(?i)(초기화|reset|clear|삭제|delete|drop)\s*(해줘|시작|실행|해\b|하자|해봐|go|start|run)",
        r"(?i)(지식\s*그래프|knowledge\s*graph|KG)\s*(초기화|리셋|reset|clear|삭제)",
        r"(?i)(데이터\s*)(수집|업데이트|갱신|refresh)\s*(해줘|시작|실행|해\b|하자)",
    ]
```

### 3-2: check_input() 수정

`check_input()` 메서드(line 136)에서 기존 INJECTION_PATTERNS 검사 후
SYSTEM_COMMAND_PATTERNS 검사 추가:

```python
        # 1. 명백한 인젝션 패턴 검사
        for pattern in cls.INJECTION_PATTERNS:
            if re.search(pattern, text):
                logger.warning(f"Injection attempt blocked: pattern matched - {pattern[:50]}")
                return False, "injection_detected", ""

        # 1.5 시스템 명령 차단 (크롤링, 초기화 등)
        for pattern in cls.SYSTEM_COMMAND_PATTERNS:
            if re.search(pattern, text):
                logger.warning(f"System command blocked: {pattern[:50]}")
                return False, "system_command_blocked", ""
```

### 3-3: get_rejection_message() 추가

`messages` 딕셔너리(line 220)에 추가:

```python
            "system_command_blocked": (
                "시스템 관리 명령은 챗봇에서 실행할 수 없습니다.\n\n"
                "크롤링, 초기화 등 시스템 작업은 관리자 API를 통해 실행해주세요.\n"
                "저는 LANEIGE 마켓 분석 질문에 답변드릴 수 있습니다:\n"
                "• 브랜드 순위 및 점유율 분석\n"
                "• 경쟁사 비교\n"
                "• 카테고리 트렌드"
            ),
```

### ⚠️ 과잉 차단 방지 (데이터 연동 보호 — 가장 중요!)

**반드시 차단해야 하는 것**: "시스템 동사 + 실행 동사" 조합
**절대 차단하면 안 되는 것**: 분석 요청, 데이터 조회, 일반 질문

```bash
python3 -c "
from src.core.prompt_guard import PromptGuard
print('=== 차단 테스트 ===')
for q in ['크롤링 해줘','크롤링 시작','지식 그래프 초기화해줘','crawl start','데이터 수집 실행해','데이터 업데이트 해줘','최신 데이터 크롤링해줘']:
    safe, reason, _ = PromptGuard.check_input(q)
    status = 'BLOCK ✅' if not safe else 'PASS ❌ FAIL!'
    print(f'  [{status}] {q}')
print()
print('=== 통과 테스트 (하나라도 ❌면 패턴 수정!) ===')
for q in ['크롤링 결과 분석해줘','LANEIGE 순위 알려줘','경쟁사 비교해줘','최근 데이터 트렌드는?','업데이트된 순위 보여줘','데이터 분석 결과 알려줘','SoS 점유율 변화 분석','오늘 주요 인사이트는?','시장 점유율 변화 분석해줘','LANEIGE 브랜드 현황 요약해줘','카테고리별 순위 변동 추이','Lip Care 경쟁사 분석','크롤링된 데이터 기반으로 분석해줘','수집된 데이터에서 트렌드 찾아줘','업데이트 현황 알려줘']:
    safe, reason, _ = PromptGuard.check_input(q)
    status = 'PASS ✅' if safe else 'BLOCK ❌ FAIL!'
    print(f'  [{status}] {q}')
"
```

### 검증:
```bash
python3 -m pytest tests/unit/core/ -v -k "prompt_guard"
python3 -m pytest tests/adversarial/ -v
```

---

## Fix 4: LLMOrchestrator 크롤링 자동 실행 제거

**파일**: `src/core/llm_orchestrator.py`
**라인**: 453-455
**담당**: Developer A

### 변경 전 (line 453-455):
```python
        # 크롤링 요청 감지
        if any(kw in query.lower() for kw in ["크롤링", "수집", "업데이트", "refresh"]):
            return Decision(tool="crawl_amazon", reason="크롤링 요청 감지", confidence=0.9)
```

### 변경 후:
```python
        # 크롤링 요청 → 시스템 안내 (챗봇에서 직접 실행 차단)
        if any(kw in query.lower() for kw in ["크롤링", "수집", "업데이트", "refresh"]):
            return Decision(
                tool="direct_answer",
                reason="크롤링은 스케줄러를 통해 자동 실행됩니다",
                confidence=0.8,
                key_points=[
                    "크롤링은 매일 22:00 KST에 자동 실행됩니다",
                    "수동 크롤링은 관리자 API(/api/crawl/start)를 이용하세요",
                ],
            )
```

### ⚠️ 데이터 연동 체크:

**이 Fix는 llm_orchestrator.py만 수정합니다. 아래 파일은 절대 수정하지 마세요:**

| 절대 수정 금지 파일 | 이유 |
|-------------------|------|
| `src/core/brain.py` | 스케줄러 크롤링 경로 (`_execute_scheduled_task → crawl_workflow`) |
| `src/core/tool_coordinator.py` | 도구 실행 인프라 |
| `src/core/tools.py` | 도구 등록 (crawl_amazon 도구 자체는 유지) |
| `src/api/routes/crawl.py` | 관리자 크롤링 API (별도 인증) |
| `src/core/batch_workflow.py` | 배치 워크플로우 |

**스케줄러 크롤링은 영향받지 않음** — 확인된 별도 경로:
```
brain.py:1631 → _execute_scheduled_task()
→ action == "crawl_workflow"
→ BatchWorkflow().run_daily_workflow()
```
이 경로는 `llm_orchestrator.py`를 거치지 않으므로 Fix 4와 무관.

**관리자 API도 영향받지 않음**:
```
src/api/routes/crawl.py → verify_api_key → CrawlManager.start_crawl()
```
이 경로도 `llm_orchestrator.py`를 거치지 않음.

---

## Fix 5: 테스트 업데이트

**파일**: `tests/unit/core/test_response_pipeline.py`
**담당**: Developer C

### 변경:
- line 637: `"지식 그래프 초기화해줘"` → `"LANEIGE 브랜드 현황 요약해줘"`
- line 648: `"최신 데이터 크롤링해줘"` → `"최근 시장 점유율 변화 분석해줘"`

### 검증:
```bash
python3 -m pytest tests/unit/core/test_response_pipeline.py -v -k "suggestion"
```

---

## 팀 구성

| 역할 | 담당 Fix | 수정 파일 | 데이터 위험 |
|------|---------|-----------|------------|
| **Dev A** (API 보안) | Fix 2 + Fix 4 | `chat.py`, `llm_orchestrator.py` | ⚠️ SSE 인증 |
| **Dev B** (PromptGuard) | Fix 3 | `prompt_guard.py` | ⚠️ 과잉 차단 |
| **Dev C** (추천 질문) | Fix 1 + Fix 5 | `response_pipeline.py`, `test_*.py` | 🟢 낮음 |

**3명 모두 서로 다른 파일을 수정하므로 충돌 없이 병렬 작업 가능**

---

## 작업 순서

```
1단계 (병렬):   Dev A + Dev B + Dev C 동시 작업
2단계 (교차):   Fix 3 패턴이 Fix 1 교체 질문을 차단하지 않는지 확인
3단계 (테스트):  python3 -m pytest tests/ -x -v
4단계 (E2E):    대시보드 접속 → 데이터 연동 확인 → 챗봇 보안 확인
5단계 (커밋):   Fix별 분리 커밋
```

---

## 통합 검증

```bash
# 전체 테스트
python3 -m pytest tests/ -x -v 2>&1 | tail -30

# 데이터 연동 무결성
python3 -m pytest tests/unit/api/ -v -k "data"
python3 -m pytest tests/unit/api/ -v -k "health"

# 보안 회귀
python3 -m pytest tests/adversarial/ -v
python3 -m pytest tests/unit/core/ -v -k "prompt_guard"
```

---

## E2E 수동 검증

```bash
# 서버 시작
uvicorn src.api.dashboard_api:app --host 0.0.0.0 --port 8001 --reload

# 1. 데이터 연동 (가장 먼저!)
# http://localhost:8001/dashboard
# → 차트 데이터 정상 로딩?
# → Action Board 정상?
# → 브라우저 콘솔에 에러 없음?

# 2. 챗봇 보안
# → "크롤링 해줘" → 차단 메시지
# → "LANEIGE 순위 알려줘" → 정상 응답
# → 추천 질문에 시스템 명령 없음

# 3. API 인증
curl -X POST http://localhost:8001/api/v4/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "test"}' -w "\n%{http_code}"
# 기대: 401 또는 503
```

---

## 롤백 계획

Fix별 분리 커밋하여 개별 롤백 가능:

```bash
git add src/core/response_pipeline.py && git commit -m "fix: 추천 질문에서 위험한 시스템 명령 제거 (Fix 1)"
git add src/api/routes/chat.py && git commit -m "fix: SSE 엔드포인트에 API Key 인증 추가 (Fix 2)"
git add src/core/prompt_guard.py && git commit -m "fix: PromptGuard에 시스템 명령 차단 패턴 추가 (Fix 3)"
git add src/core/llm_orchestrator.py && git commit -m "fix: 챗봇에서 크롤링 자동 실행 제거 (Fix 4)"
git add tests/ && git commit -m "test: 보안 수정 반영 테스트 업데이트 (Fix 5)"
```

**데이터 연동 문제 발생 시**:
```bash
# 가장 의심되는 Fix부터 롤백
git revert <Fix 2 commit>  # SSE 인증이 문제면
git revert <Fix 3 commit>  # PromptGuard 과잉 차단이면
git revert <Fix 4 commit>  # 크롤링 경로 영향이면
```
