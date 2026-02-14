# Ruff Lint Cleanup Plan

> 작성일: 2026-02-15
> 대상: 전체 코드베이스 (`src/`, `scripts/`, `tests/`, `examples/`)

---

## 1. 현황 요약

| 항목 | 수치 |
|------|------|
| 초기 에러 | 256개 |
| `ruff check --fix` 자동 수정 | 231개 |
| `ruff format` 포매팅 | 33개 파일 |
| 수동 수정 완료 | 21개 |
| **잔여 에러** | **4개** |

---

## 2. 잔여 4개 에러 — 근본 원인 분석

### 원인 1: ruff `--fix`의 safe/unsafe fix 분류 체계

ruff는 fix를 **safe fix**와 **unsafe fix**로 나눕니다:

- **safe fix** (`--fix`): 동작을 절대 변경하지 않는 수정 → 자동 적용
- **unsafe fix** (`--unsafe-fixes`): 동작이 미세하게 변경될 수 있는 수정 → 수동 승인 필요

잔여 4개 에러는 모두 **unsafe fix**로 분류되어 `--fix`에서 제외되었습니다.

#### C401 (2개) — `scripts/evaluate_golden.py:155-156`

```python
# Before (unsafe fix 대상)
found_set = set(item.upper() for item in found)
expected_set = set(item.upper() for item in expected)

# After
found_set = {item.upper() for item in found}
expected_set = {item.upper() for item in expected}
```

**왜 unsafe인가?**
- `set(generator)` → `{comprehension}` 변환 시, generator의 lazy evaluation이 eager evaluation으로 바뀜
- 이론적으로 사이드이펙트가 있는 generator에서 동작 차이 발생 가능
- 실제 이 코드에서는 `item.upper()`가 순수 함수이므로 안전함

#### F401 (2개) — `src/infrastructure/bootstrap.py:103, 112`

```python
# Line 103: UnifiedBrain 미사용
from src.core.brain import UnifiedBrain, get_brain
#                          ^^^^^^^^^^^^  ← 제거 대상

# Line 112: CrawlManager 미사용
from src.core.crawl_manager import CrawlManager, get_crawl_manager
#                                  ^^^^^^^^^^^^  ← 제거 대상
```

**왜 unsafe인가?**
- 같은 `from ... import A, B` 문에서 하나만 제거하면 import 구조가 변경됨
- ruff는 multi-import에서 부분 제거를 unsafe로 분류
- 제거된 이름이 다른 곳에서 `TYPE_CHECKING` 등으로 참조될 가능성을 배제 못함

### 원인 2: 수동 수정 시 라인 넘버 불일치

1. `ruff check --fix` 실행 (231개 수정) → 파일 내용 변경
2. `ruff format .` 실행 (33개 파일 재포매팅) → **라인 넘버 shift**
3. 최초 에러 리포트의 라인 넘버로 파일을 읽음 → 잘못된 위치 참조
4. `evaluate_golden.py`의 C401이 139→155로 이동하여 수동 수정에서 누락

### 원인 3: 터미널 출력 truncation

- 초기 `ruff check --fix` 출력이 30,000자 제한으로 잘림 (1,061자 truncated)
- `bootstrap.py`의 F401 에러가 잘린 영역에 포함되어 1차 수동 수정 시 누락

---

## 3. 수정 계획

### Step 1: 잔여 4개 에러 수동 수정

| # | 파일 | 규칙 | 수정 내용 |
|---|------|------|-----------|
| 1 | `scripts/evaluate_golden.py:155` | C401 | `set(...)` → `{...}` |
| 2 | `scripts/evaluate_golden.py:156` | C401 | `set(...)` → `{...}` |
| 3 | `src/infrastructure/bootstrap.py:103` | F401 | `UnifiedBrain` import 제거 |
| 4 | `src/infrastructure/bootstrap.py:112` | F401 | `CrawlManager` import 제거 |

### Step 2: 검증

```bash
ruff check .          # 에러 0개 확인
ruff format --check . # 포맷 일관성 확인
```

### Step 3: 커밋

```bash
git add -A
git commit -m "style: ruff format and lint cleanup"
```

---

## 4. 재발 방지 권장사항

### 옵션 A: `--unsafe-fixes` 활용 (권장)

```bash
# 앞으로 린트 정리 시:
ruff check --fix --unsafe-fixes .
ruff format .
ruff check .  # 최종 검증
```

`--unsafe-fixes`를 쓰면 C401, F401 multi-import 등도 자동 수정됩니다.

### 옵션 B: pre-commit hook 강화

`.pre-commit-config.yaml`에 ruff를 이미 사용 중이라면 `--unsafe-fixes` 추가:

```yaml
- repo: https://github.com/astral-sh/ruff-pre-commit
  hooks:
    - id: ruff
      args: [--fix, --unsafe-fixes]
    - id: ruff-format
```

### 옵션 C: 순서 최적화

`ruff format` → `ruff check --fix` 순서로 실행하면 라인 넘버 불일치를 줄일 수 있음.
단, `--fix`가 포매팅을 깨뜨릴 수 있으므로 마지막에 `ruff format`을 한 번 더 실행하는 것이 안전.

```bash
# 최적 순서
ruff check --fix --unsafe-fixes .
ruff format .
ruff check .  # 0 errors 확인
```
