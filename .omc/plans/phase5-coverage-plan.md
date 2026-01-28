# Phase 5: 커버리지 개선 계획서 (수정본)

> 생성일: 2026-01-28
> 상태: 전문가 검토 완료 (Architect, TDD Guide, Critic)
> 판정: **조건부 승인** - 아래 절차 준수 시 실행 가능

---

## 현재 상황 분석

| 항목 | 값 | 비고 |
|------|-----|------|
| 전체 .py 파일 | 933개 | src/ 폴더 |
| 중복 파일 | 798개 (85.6%) | `* 2.py`, `* 3.py` 패턴 |
| 실제 소스 파일 | 135개 | |
| 실제 소스 LOC | 57,634줄 | |
| 현재 커버리지 | 0.56% | 중복으로 왜곡됨 |
| 예상 실제 커버리지 | ~4-5% | 중복 제거 후 |

---

## 목표 수정

| 원래 목표 | 수정 목표 | 근거 |
|-----------|-----------|------|
| 70% 전체 | **40% 전체** (1차) | Architect: 500+ 테스트 필요, 비현실적 |
| - | **70% 전체** (3개월) | 장기 목표로 전환 |
| 80% agents | **60% agents** | 복잡한 async + 외부 의존성 |
| 100% domain | **90% domain** | Pydantic 모델은 자체 문서화 |
| 90% validators | **95% validators** | 보안 임계 영역 |

---

## Phase 5.0: 베이스라인 측정

### 목표
중복 파일 영향 없이 현재 실제 커버리지 파악

### 실행 명령
```bash
# 핵심 테스트만 실행 (중복 제외)
pytest tests/unit/domain/test_exceptions.py \
       tests/unit/domain/test_entities.py \
       tests/unit/domain/test_relations.py \
       tests/unit/infrastructure/test_container.py \
       tests/unit/api/test_input_validation.py \
       -v --cov=src/domain --cov=src/infrastructure --cov=src/api/validators \
       --cov-report=term-missing
```

### 수용 기준
- [ ] 베이스라인 커버리지 수치 기록: ____%
- [ ] 핵심 모듈별 커버리지 기록

---

## Phase 5.1: 중복 파일 안전 삭제

### 삭제 대상 패턴
```
* [0-9].py      # 예: brain 2.py
* [0-9].json    # 예: thresholds 2.json
* [0-9].md      # 예: AP_1Q25_EN 2.md
* [0-9].toml    # 예: pyproject 2.toml
* [0-9].html    # 예: dashboard 2.html
```

### 안전 삭제 절차

#### Step 1: 삭제 목록 생성 및 검증
```bash
# 삭제 대상 목록 생성
find . -type f \( -name "* [0-9].py" -o -name "* [0-9].json" -o -name "* [0-9].md" -o -name "* [0-9].toml" -o -name "* [0-9].html" \) > /tmp/duplicates_to_delete.txt

# 개수 확인
wc -l /tmp/duplicates_to_delete.txt

# 샘플 확인 (원본 파일이 포함되지 않았는지)
head -20 /tmp/duplicates_to_delete.txt
tail -20 /tmp/duplicates_to_delete.txt
```

#### Step 2: 원본 파일 무결성 확인
```bash
# 원본 파일 존재 확인
ls src/core/brain.py           # 원본 있어야 함
ls "src/core/brain 2.py"       # 중복 있어야 함

# 원본과 중복 차이 확인 (동일해야 함)
diff src/core/brain.py "src/core/brain 2.py" | head -10
```

#### Step 3: 백업 브랜치 생성
```bash
git checkout -b backup-before-cleanup-$(date +%Y%m%d)
git add -A
git commit -m "backup: before duplicate file cleanup"
git checkout main
```

#### Step 4: 삭제 실행
```bash
# DRY RUN (실제 삭제 안 함)
cat /tmp/duplicates_to_delete.txt | xargs -I {} echo "Would delete: {}"

# 실제 삭제
cat /tmp/duplicates_to_delete.txt | xargs rm -f

# 또는 find로 직접 삭제
find . -type f \( -name "* [0-9].py" -o -name "* [0-9].json" -o -name "* [0-9].md" \) -delete
```

#### Step 5: 삭제 후 검증
```bash
# 중복 파일 0개 확인
find . -name "* [0-9].*" -type f | wc -l  # 0이어야 함

# 원본 import 테스트
python -c "from src.core.brain import UnifiedBrain; print('OK')"
python -c "from src.ontology.knowledge_graph import KnowledgeGraph; print('OK')"
python -c "from src.agents.crawler_agent import CrawlerAgent; print('OK')"

# 테스트 수집 확인
pytest tests/unit/ --collect-only 2>&1 | grep "error" | wc -l  # 0이어야 함
```

### 롤백 절차 (문제 발생 시)
```bash
git checkout backup-before-cleanup-YYYYMMDD
git checkout -B main
```

### 수용 기준
- [ ] `find . -name "* [0-9].*" -type f | wc -l` = 0
- [ ] Python import 오류 없음
- [ ] pytest 수집 오류 없음

---

## Phase 5.2: 실제 커버리지 측정

### 실행 명령
```bash
pytest tests/unit/ -v --cov=src --cov-report=html --cov-report=term-missing
```

### 예상 결과
- 중복 제거 후 커버리지: **15-25%** (추정)

### 수용 기준
- [ ] 전체 테스트 통과
- [ ] 커버리지 리포트 생성
- [ ] 모듈별 커버리지 기록

---

## Phase 5.3: 커버리지 목표 조정 및 추가 테스트 계획

### 모듈별 우선순위 (가치/노력 비율)

| 우선순위 | 모듈 | 현재 예상 | 목표 | 필요 테스트 |
|----------|------|-----------|------|-------------|
| P0 | `src/domain/` | 90% | 95% | 5개 |
| P0 | `src/api/validators/` | 80% | 95% | 10개 |
| P1 | `src/infrastructure/` | 70% | 85% | 10개 |
| P1 | `src/core/models.py` | 50% | 80% | 15개 |
| P2 | `src/rag/entity_linker.py` | 60% | 75% | 15개 |
| P2 | `src/ontology/reasoner.py` | 40% | 60% | 20개 |
| P3 | `src/agents/` | 30% | 50% | 40개 |
| **EXCLUDE** | `src/tools/` | 20% | 제외 | - |

### 제외 대상 (pyproject.toml 업데이트)
```toml
[tool.coverage.run]
omit = [
    "*/tests/*",
    "*/__pycache__/*",
    "*/venv/*",
    "*/_deprecated/*",      # 추가
    "src/tools/*",          # 추가 - I/O 무거움
]
```

### 추가 테스트 파일 목록

#### P0 - 즉시 필요
- `tests/unit/core/test_models.py` (NEW)
- `tests/unit/rag/test_reranker.py` (NEW)

#### P1 - 1주 내
- `tests/unit/core/test_brain.py` (NEW) - **최우선**
- `tests/unit/ontology/test_knowledge_graph.py` (NEW)
- `tests/unit/rag/test_hybrid_retriever.py` (NEW)

#### P2 - 2주 내
- `tests/unit/core/test_scheduler.py` (NEW)
- `tests/unit/core/test_rules_engine.py` (NEW)

### 수용 기준
- [ ] 40% 커버리지 달성 (1차 목표)
- [ ] P0 테스트 전체 통과
- [ ] CI/CD에서 `--cov-fail-under=40` 통과

---

## 실행 체크리스트

### 즉시 실행 (오늘)
- [ ] Phase 5.0: 베이스라인 측정
- [ ] Phase 5.1: 중복 파일 삭제
- [ ] Phase 5.2: 실제 커버리지 측정

### 1주 내
- [ ] Phase 5.3: 추가 테스트 계획 확정
- [ ] P0 테스트 작성 (15개)

### 4주 내 (40% 목표)
- [ ] P1 테스트 작성 (50개)
- [ ] P2 테스트 작성 (35개)
- [ ] 40% 커버리지 달성

### 3개월 내 (70% 목표)
- [ ] P3 테스트 작성 (100개)
- [ ] 통합 테스트 추가
- [ ] 70% 커버리지 달성

---

## 위험 완화

| 위험 | 심각도 | 완화 전략 |
|------|--------|-----------|
| 원본 파일 실수 삭제 | HIGH | 백업 브랜치 + import 테스트 |
| 70% 미달 | MEDIUM | 40%로 1차 목표 하향 |
| 테스트 유지보수 부담 | LOW | Mock 공유 fixture 사용 |
| CI 시간 증가 | LOW | `-x` 플래그 + 병렬 실행 |

---

## 전문가 검토 결과

| 전문가 | 판정 | 조건 |
|--------|------|------|
| Architect | **승인** | 40% 1차 목표 수용 |
| TDD Guide | **승인** | 중복 삭제 우선 실행 |
| Critic | **조건부 승인** | 위 절차 준수 시 |

---

## 참고 명령어

```bash
# 커버리지 측정
pytest tests/unit/ -v --cov=src --cov-report=html

# 커버리지 리포트 열기
open coverage_html/index.html

# 특정 모듈만 측정
pytest tests/unit/domain/ -v --cov=src/domain --cov-report=term-missing

# 커버리지 실패 임계값
pytest tests/ --cov=src --cov-fail-under=40
```
