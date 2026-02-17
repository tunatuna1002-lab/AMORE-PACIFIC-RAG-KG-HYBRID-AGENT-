# 🧹 AMORE 프로젝트 루트 정리 계획

## 🧭 한 줄 요약
루트에 흩어진 27개 파일을 정석 Python 프로젝트 구조로 재배치 — 루트에는 13개 표준 파일만 남기고, 나머지 14개를 `docs/`, `scripts/`, `src/` 로 이동

---

## 📌 현재 루트 파일 분류

### ✅ 루트에 있어야 하는 파일 (13개) — 건드리지 않음

| 파일 | 이유 |
|------|------|
| `README.md` | GitHub 표준 |
| `LICENSE` | GitHub 표준 |
| `CLAUDE.md` | Claude Code 관례 (루트 필수) |
| `AGENTS.md` | Claude Code 에이전트 설정 (루트 관례) |
| `.gitignore` | Git 표준 |
| `.dockerignore` | Docker 표준 |
| `.pre-commit-config.yaml` | pre-commit 표준 |
| `.secrets.baseline` | detect-secrets 표준 |
| `.env.example` | 환경변수 템플릿 |
| `Dockerfile` | Docker 표준 |
| `pyproject.toml` | Python 패키징 표준 |
| `requirements.txt` | pip 표준 |
| `railway.toml` | Railway 배포 설정 |

### 🚚 이동해야 하는 파일 (14개)

#### → `docs/` (문서류 8개)

| 현재 위치 (루트) | 이동 위치 | 이유 |
|------------------|-----------|------|
| `ARCHITECTURE.md` | `docs/ARCHITECTURE.md` | 아키텍처 문서 |
| `PROJECT_PLAN.md` | `docs/PROJECT_PLAN.md` | 프로젝트 계획 |
| `SECURITY_AUDIT_REPORT.md` | `docs/SECURITY_AUDIT_REPORT.md` | 보안 감사 |
| `EMBEDDING_CACHE_SUMMARY.md` | `docs/EMBEDDING_CACHE_SUMMARY.md` | 기능 문서 |
| `IMPLEMENTATION_SUMMARY_H.1.3.md` | `docs/IMPLEMENTATION_SUMMARY_H.1.3.md` | 구현 요약 |
| `LANEIGE_influencer_map.md` | `docs/research/LANEIGE_influencer_map.md` | 리서치 자료 |
| `AMORE_Analyst_Report_2026-01-14_2026-01-25 (3).docx.md` | `docs/reports/AMORE_Analyst_Report.md` | 분석 리포트 (파일명도 정리) |
| `THIRD_PARTY_LICENSES.md` | `docs/THIRD_PARTY_LICENSES.md` | 서드파티 라이선스 |

#### → `docs/analysis/` (의존성 분석 4개)

| 현재 위치 (루트) | 이동 위치 |
|------------------|-----------|
| `DEPENDENCY_ANALYSIS.txt` | `docs/analysis/DEPENDENCY_ANALYSIS.txt` |
| `DEPENDENCY_GRAPH.txt` | `docs/analysis/DEPENDENCY_GRAPH.txt` |
| `DEPENDENCY_INDEX.md` | `docs/analysis/DEPENDENCY_INDEX.md` |
| `DEPENDENCY_SUMMARY.txt` | `docs/analysis/DEPENDENCY_SUMMARY.txt` |
| `FILE_IMPORT_MAP.txt` | `docs/analysis/FILE_IMPORT_MAP.txt` |

#### → 코드 파일 재배치 (4개) ⚠️ import 수정 필요

| 현재 위치 | 이동 위치 | 이유 | 영향도 |
|-----------|-----------|------|--------|
| `main.py` | 루트 유지 | 엔트리포인트 | - |
| `start.py` | `scripts/start.py` | 시작 스크립트 | LOW |
| `start_dashboard.command` | `scripts/start_dashboard.command` | macOS 시작 | LOW |
| `orchestrator.py` | `src/core/orchestrator.py` | 핵심 모듈 | **HIGH** |
| `dashboard_api.py` | `src/api/dashboard_api.py` | API 서버 | **HIGH** |

#### → 삭제 후보 (재확인 필요)

| 파일 | 이유 |
|------|------|
| `.env.test` | `.gitignore`에 추가하고 로컬만 유지 |
| `requirements-railway.txt` | `pyproject.toml`로 통합 가능 |
| `pytest.ini` | `pyproject.toml`에 이미 설정 있으면 중복 |

---

## 🎯 정리 후 루트 구조 (Before → After)

### Before (27개 파일 + 15개 폴더)
```
루트/
├── 15개 폴더 (OK)
├── 13개 표준 파일 (OK)
└── 14개 잡다한 파일 ← 문제
```

### After (13개 파일 + 15개 폴더)
```
AMORE-PACIFIC-RAG-KG-HYBRID-AGENT/
│
├── .claude/                    # Claude Code 설정
├── .github/workflows/          # CI/CD
├── .omc/                       # OMC 설정
├── config/                     # 설정 파일
├── dashboard/                  # 대시보드 UI (HTML/CSS/JS)
├── docs/                       # 📁 문서 통합
│   ├── analysis/               #   의존성 분석 5개
│   ├── guides/                 #   가이드 (기존)
│   ├── reports/                #   분석 리포트
│   ├── research/               #   리서치 자료
│   ├── ARCHITECTURE.md
│   ├── PROJECT_PLAN.md
│   ├── SECURITY_AUDIT_REPORT.md
│   ├── EMBEDDING_CACHE_SUMMARY.md
│   ├── IMPLEMENTATION_SUMMARY_H.1.3.md
│   └── THIRD_PARTY_LICENSES.md
├── eval/                       # 평가
├── examples/                   # 예제
├── prompts/                    # 프롬프트 템플릿
├── scripts/                    # 스크립트
│   ├── start.py
│   └── start_dashboard.command
├── src/                        # 소스코드
│   ├── api/
│   │   └── dashboard_api.py    # 🚚 루트에서 이동
│   ├── core/
│   │   └── orchestrator.py     # 🚚 루트에서 이동
│   └── ...
├── static/fonts/               # 폰트
├── tests/                      # 테스트
│
├── .dockerignore               # ── 표준 설정 ──
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── .secrets.baseline
├── AGENTS.md                   # Claude Code
├── CLAUDE.md                   # Claude Code
├── Dockerfile
├── LICENSE
├── README.md
├── main.py                     # 엔트리포인트
├── pyproject.toml
├── railway.toml
└── requirements.txt
```

---

## ⚠️ import 수정이 필요한 파일 (핵심)

### 1. `dashboard_api.py` → `src/api/dashboard_api.py`

영향받는 곳:
- `Dockerfile` — `CMD` 또는 `ENTRYPOINT`에서 참조
- `railway.toml` — 시작 명령어
- `main.py` — import 또는 subprocess 호출
- `start.py` — uvicorn 시작 경로
- `README.md` — Quick Start 가이드
- `start_dashboard.command` — 시작 스크립트

수정 예시:
```python
# Before
uvicorn dashboard_api:app --host 0.0.0.0 --port 8001

# After
uvicorn src.api.dashboard_api:app --host 0.0.0.0 --port 8001
```

### 2. `orchestrator.py` → `src/core/orchestrator.py`

영향받는 곳:
- `main.py` — import
- `dashboard_api.py` — import
- `src/` 내 다른 모듈 — import
- `tests/` — import

수정 예시:
```python
# Before
from orchestrator import Orchestrator

# After
from src.core.orchestrator import Orchestrator
```

---

## 🔧 실행 순서 (Claude Code 프롬프트)

### Step 1: 문서 이동 (안전, import 무관)
```
다음 파일들을 이동해줘. git mv 사용:

1. git mv ARCHITECTURE.md docs/ARCHITECTURE.md
2. git mv PROJECT_PLAN.md docs/PROJECT_PLAN.md
3. git mv SECURITY_AUDIT_REPORT.md docs/SECURITY_AUDIT_REPORT.md
4. git mv EMBEDDING_CACHE_SUMMARY.md docs/EMBEDDING_CACHE_SUMMARY.md
5. git mv IMPLEMENTATION_SUMMARY_H.1.3.md docs/IMPLEMENTATION_SUMMARY_H.1.3.md
6. git mv THIRD_PARTY_LICENSES.md docs/THIRD_PARTY_LICENSES.md
7. mkdir -p docs/research && git mv LANEIGE_influencer_map.md docs/research/
8. mkdir -p docs/reports && git mv "AMORE_Analyst_Report_2026-01-14_2026-01-25 (3).docx.md" docs/reports/AMORE_Analyst_Report.md
9. mkdir -p docs/analysis
10. git mv DEPENDENCY_ANALYSIS.txt docs/analysis/
11. git mv DEPENDENCY_GRAPH.txt docs/analysis/
12. git mv DEPENDENCY_INDEX.md docs/analysis/
13. git mv DEPENDENCY_SUMMARY.txt docs/analysis/
14. git mv FILE_IMPORT_MAP.txt docs/analysis/

커밋: "refactor: move documentation files to docs/ directory"
```

### Step 2: 스크립트 이동 (안전)
```
다음 파일들을 scripts/로 이동해줘:

1. git mv start.py scripts/start.py
2. git mv start_dashboard.command scripts/start_dashboard.command

scripts/start_dashboard.command 안의 경로도 수정해줘.
커밋: "refactor: move startup scripts to scripts/ directory"
```

### Step 3: dashboard_api.py 이동 (⚠️ 신중하게)
```
dashboard_api.py를 src/api/dashboard_api.py로 이동해줘.

이동 전에 먼저:
1. grep -rn "dashboard_api" . --include="*.py" --include="*.toml" --include="*.yaml" --include="Dockerfile" --include="*.command" --include="*.md"
로 모든 참조를 찾아줘.

그 다음:
1. git mv dashboard_api.py src/api/dashboard_api.py
2. 찾은 모든 참조를 src.api.dashboard_api로 수정
3. Dockerfile CMD 수정
4. railway.toml 시작 명령어 수정
5. README.md Quick Start 수정

전체 테스트 실행해서 확인:
python -m pytest tests/ -x --tb=short

커밋: "refactor: move dashboard_api.py to src/api/"
```

### Step 4: orchestrator.py 이동 (⚠️ 가장 신중하게)
```
orchestrator.py를 src/core/orchestrator.py로 이동해줘.

이동 전에 먼저:
1. grep -rn "from orchestrator\|import orchestrator" . --include="*.py"
2. grep -rn "orchestrator.py" . --include="*.py" --include="*.toml" --include="*.yaml" --include="*.md"
로 모든 참조를 찾아줘.

주의: src/core/에 이미 orchestrator 관련 파일이 있을 수 있으니 충돌 확인 필요.

그 다음:
1. git mv orchestrator.py src/core/orchestrator.py (충돌 시 이름 변경)
2. 모든 import 수정
3. 전체 테스트: python -m pytest tests/ -x --tb=short

커밋: "refactor: move orchestrator.py to src/core/"
```

### Step 5: 최종 확인 & 정리
```
1. 전체 테스트: python -m pytest tests/ -v --cov=src --cov-report=term
2. CI 확인: git push
3. 불필요 파일 정리:
   - pytest.ini 내용을 pyproject.toml에 통합 (중복이면 삭제)
   - requirements-railway.txt 필요 여부 확인

커밋: "refactor: complete root directory cleanup"
```

---

## ✅ 체크리스트

- [x] Step 1: 문서 14개 이동 (THIRD_PARTY_LICENSES.md — 나머지 13개는 이미 이동됨)
- [x] Step 2: 스크립트 2개 이동 (start.py, start_dashboard.command → scripts/)
- [x] Step 3: dashboard_api.py 이동 + 참조 수정 (→ src/api/dashboard_api.py, 10개 파일 수정)
- [x] Step 4: orchestrator.py 이동 + import 수정 (→ src/core/orchestrator.py, 5개 파일 수정)
- [x] Step 5: 전체 테스트 통과 확인 (4226 passed, 6 skipped, coverage 70.80%)
- [ ] Step 6: CI/CD 그린 확인 (push 후 확인 필요)
- [x] Step 7: README Quick Start 경로 업데이트 (Step 3에서 완료)
- [x] Step 8: Dockerfile CMD 경로 확인 (Step 3에서 scripts/start.py로 수정)

---

## 🧪 검증 포인트

1. `python -m pytest tests/ -x` — 테스트 전체 통과
2. `docker build -t test .` — Docker 빌드 성공
3. `uvicorn src.api.dashboard_api:app` — 서버 정상 시작
4. GitHub Actions CI — 그린
5. Railway 배포 — 정상 동작
