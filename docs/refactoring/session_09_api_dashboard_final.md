# Session 9: API 라우트 정리 + dashboard_api.py 분할 + 최종 검증

> ⏱ 예상 시간: 50~70분 | 위험도: 🟡 중간 | 선행 조건: Session 7, 8 완료

---

## 프롬프트 (아래를 복사해서 새 Claude Code 세션에 붙여넣기)

```
너는 20년 베테랑 Python 개발자이자 FastAPI 전문가야. AMORE RAG-KG Hybrid Agent 리팩토링의 마지막 세션이야: API 레이어 정리와 최종 검증.

## 이번 세션 목표
1. `dashboard_api.py` (5626줄) God Object 분할
2. `src/api/routes/` 미마운트 라우트 활성화/정리
3. infrastructure/ DI 컨테이너 정리
4. **전체 리팩토링 최종 검증**

## 컨텍스트
- 프로젝트: `/Users/leedongwon/Desktop/AMORE-RAG-ONTOLOGY-HYBRID AGENT/`
- 전체 마스터 플랜: `docs/refactoring/00_MASTER_PLAN.md` 참조
- Session 0~8이 모두 완료된 상태
- Python 3.13.7 (`python3` 사용)

## Part A: dashboard_api.py 분할

### 현재 문제
`dashboard_api.py`가 5626줄로 너무 큼. 내부에:
- FastAPI app 초기화
- 모든 라우트 핸들러
- WebSocket 핸들러
- 스케줄러 시작/종료
- 미들웨어 설정
- CORS 설정
- 온갖 import (29개 모듈)

### 분할 계획
```
dashboard_api.py (300줄 이하로)
├── App 초기화, CORS, 미들웨어
├── Router include (src/api/routes/)
├── Startup/Shutdown 이벤트
└── DI 설정 (infrastructure/bootstrap.py 호출)

src/api/
├── routes/
│   ├── health.py       # /api/health
│   ├── chat.py         # /api/v3/chat (이미 있을 수 있음)
│   ├── crawl.py        # /api/crawl/* (이미 있음)
│   ├── data.py         # /api/data (이미 있음)
│   ├── brain.py        # /api/v4/brain/* (이미 있음)
│   ├── alerts.py       # /api/alerts/*
│   ├── deals.py        # /api/deals/*
│   ├── signals.py      # /api/signals/*
│   ├── export.py       # /api/export/*
│   └── websocket.py    # WebSocket 핸들러 (NEW)
├── middleware.py        # 미들웨어 (NEW)
└── dependencies.py     # FastAPI Dependencies
```

### 작업 순서
1. `dashboard_api.py`를 읽고 라우트별 코드 블록 식별
2. `src/api/routes/`에 이미 있는 라우트 파일 확인 — 실제 마운트되어 있는지
3. 마운트 안 된 라우트 → dashboard_api.py의 해당 핸들러와 비교 → 통합
4. dashboard_api.py에서 라우트 코드를 routes/로 이동
5. dashboard_api.py에는 app 초기화 + router include만 남기기

### dashboard_api.py 최종 형태 예시
```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import health, chat, crawl, data, brain, alerts
from src.infrastructure.bootstrap import create_container

app = FastAPI(title="AMORE Dashboard API")

# CORS
app.add_middleware(CORSMiddleware, ...)

# DI Container
container = create_container()

# Routes
app.include_router(health.router, prefix="/api")
app.include_router(chat.router, prefix="/api/v3")
app.include_router(crawl.router, prefix="/api/crawl")
app.include_router(data.router, prefix="/api")
app.include_router(brain.router, prefix="/api/v4/brain")
app.include_router(alerts.router, prefix="/api")

@app.on_event("startup")
async def startup():
    ...

@app.on_event("shutdown")
async def shutdown():
    ...
```

## Part B: Infrastructure 정리

### bootstrap.py
- Session 2~8에서 만든 Protocol과 구현체를 연결하는 DI 설정
- 모든 의존성 주입이 여기서 이루어지도록:
```python
def create_container():
    kg = KnowledgeGraph(...)
    retriever = HybridRetriever(kg=kg, ...)
    chatbot = HybridChatbotAgent(retriever=retriever, ...)
    brain = UnifiedBrain(chatbot=chatbot, ...)
    return {"brain": brain, "chatbot": chatbot, ...}
```

## Part C: 최종 검증 (가장 중요!)

### C-1: 전체 테스트
```bash
python3 -m pytest tests/ -v --tb=short
```

### C-2: 커버리지 확인
```bash
python3 -m pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
```
목표: 40%+ (Session 0~9 전체 리팩토링 후)

### C-3: Import 검증
```bash
python3 -c "from dashboard_api import app; print('dashboard_api OK')"
python3 -c "from src.core.brain import UnifiedBrain; print('brain OK')"
python3 -c "from src.rag.hybrid_retriever import HybridRetriever; print('retriever OK')"
python3 -c "from src.ontology.knowledge_graph import KnowledgeGraph; print('kg OK')"
```

### C-4: 순환 의존성 최종 검증
```python
python3 -c "
import sys
import importlib

modules = [
    'src.domain', 'src.application', 'src.ontology', 'src.rag',
    'src.memory', 'src.monitoring', 'src.shared',
    'src.core', 'src.agents', 'src.tools', 'src.api', 'src.infrastructure'
]
for m in modules:
    try:
        importlib.import_module(m)
        print(f'  ✓ {m}')
    except ImportError as e:
        print(f'  ✗ {m}: {e}')

print('Done!')
"
```

### C-5: 서버 기동 테스트
```bash
timeout 10 python3 -m uvicorn dashboard_api:app --host 0.0.0.0 --port 8001 || true
# 10초 내에 정상 시작되는지 확인
```

### C-6: Clean Architecture 준수 검증
```bash
# domain이 다른 src/ 모듈을 import하지 않는지
grep -r "from src\." src/domain/ --include="*.py" | grep -v "from src.domain"
# 결과가 없어야 함
```

### C-7: 리팩토링 결과 보고서 생성
최종 결과를 `docs/refactoring/REFACTORING_RESULT.md`에 기록:
- 삭제된 파일 목록
- 이동된 파일 목록
- 새로 생성된 파일 목록
- 테스트 커버리지 변화 (10% → ?%)
- 순환 의존성 변화 (23개 → ?개)
- 코드 줄 수 변화
- 남은 이슈/TODO

## 주의사항
- dashboard_api.py 분할 시 WebSocket 핸들러 주의 (상태 관리 있을 수 있음)
- Startup/Shutdown 이벤트의 순서 중요
- 기존 API 엔드포인트 URL이 바뀌면 안 됨
- Railway 배포 호환성: PORT 환경변수, /api/health 헬스체크
- Context7 MCP로 FastAPI Router 관련 최신 문서 참조 가능
```

---

## 체크리스트

- [ ] dashboard_api.py 라우트 분석
- [ ] src/api/routes/ 미마운트 라우트 확인
- [ ] 라우트 코드 이동
- [ ] dashboard_api.py 300줄 이하로 축소
- [ ] bootstrap.py DI 설정
- [ ] 전체 테스트 통과
- [ ] 커버리지 40%+ 달성
- [ ] Import 검증 통과
- [ ] 순환 의존성 최종 검증 통과
- [ ] 서버 기동 테스트 통과
- [ ] Clean Architecture 준수 검증 통과
- [ ] 리팩토링 결과 보고서 생성

## 이 세션 완료 후
모든 리팩토링이 끝남. 로컬에서 다음을 수동 확인:
1. `uvicorn dashboard_api:app --port 8001` → 서버 정상 기동
2. 브라우저에서 대시보드 접속 → 정상 렌더링
3. 챗봇 API 테스트 → 응답 정상
4. 만족하면 `git push` 결정
