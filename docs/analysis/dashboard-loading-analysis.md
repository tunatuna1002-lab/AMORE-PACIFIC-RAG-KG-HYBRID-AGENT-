# 대시보드 로딩 실패 심층 분석

> 작성일: 2026-02-19
> 대상: Railway 배포 대시보드 (`/dashboard`)

---

## 1. 증상

Railway에 배포된 대시보드가 데이터를 로드하지 못하고 무한 로딩 상태에 빠짐.

| 증상 | 위치 | 상태 |
|------|------|------|
| "데이터 로딩중..." | 우측 상단 | 무한 대기 (10초 후 강제 해제) |
| "현재 --:--:--" | 우측 상단 시간 | 업데이트 안됨 |
| 액션 보드 스피너 | Home 페이지 중앙 | 무한 로딩 |
| 차트 2개 빈 상태 | Home 페이지 하단 | 데이터 없음 |
| 상단 요약 카드 | Home 페이지 상단 | 일부 표시 (STABLE UP, TOP 3, 2개) |

---

## 2. 근본 원인

### Issue #1: `dependencies.py`의 DATA_PATH 하드코딩 (CRITICAL)

**파일**: `src/api/dependencies.py:178`

```python
DATA_PATH = "./data/dashboard_data.json"  # 상대 경로 하드코딩
```

**Railway 환경**:
- Docker WORKDIR: `/app/`
- Volume 마운트: `/data/`
- `./data/dashboard_data.json` → `/app/data/dashboard_data.json` (존재하지 않음)
- 실제 파일: `/data/dashboard_data.json` (Railway Volume)

**비교**: `brain.py:57-58`은 이미 올바르게 처리:
```python
_DATA_DIR = "/data" if os.path.isdir("/data") else "./data"
```

하지만 `dependencies.py`는 이 패턴을 따르지 않음.

**결과**: `load_dashboard_data()` → `FileNotFoundError` → `{}` 반환 → `/api/data`가 404 반환

---

### Issue #2: 프론트엔드 폴백 경로 잘못됨

**파일**: `dashboard/amore_unified_dashboard_v4.html:5681`

```javascript
response = await fetch('../data/dashboard_data.json');
```

대시보드 HTML은 `/dashboard` 경로로 StaticFiles 마운트됨 (`app_factory.py:140-144`).
`../data/dashboard_data.json`은 상위 디렉토리 접근이지만 StaticFiles 마운트는 이를 허용하지 않음 → 404.

---

### Issue #3: 액션 보드 무한 스피너

**파일**: `dashboard/amore_unified_dashboard_v4.html:4303, 5750-5751`

1. 초기 HTML에 스피너 포함 (line 4303)
2. `updateDashboardFromData()` 내에서 `updateActionTable()` 호출 (line 5750-5751)
3. `dashboardData`가 null이면 `updateDashboardFromData()`가 즉시 return (line 5718)
4. `updateActionTable()`이 호출되지 않아 스피너가 영원히 유지됨

---

### Issue #4: 차트 초기화 실패

**파일**: `dashboard/amore_unified_dashboard_v4.html:10710-10760`

1. `initChartsForPage('home')` → 차트 데이터 확인
2. `dashboardData.charts.sos_trend[periodKey]` (line 10759) 접근
3. `dashboardData`가 null → 차트 빈 상태로 렌더링
4. `/api/historical` 호출도 SQLite 경로 문제 가능

---

### Issue #5: 10초 안전장치가 문제를 숨김

**파일**: `dashboard/amore_unified_dashboard_v4.html:12992-12998`

```javascript
const loadingSafety = setTimeout(() => {
    hideLoading(); // 10초 후 강제 로딩 오버레이 해제
}, 10000);
```

로딩 오버레이만 숨기고, 데이터는 여전히 로드되지 않은 상태. 사용자는 빈 대시보드를 보게 됨.

---

### Issue #6: `data.py`의 경로도 하드코딩

**파일**: `src/api/routes/data.py:458, 515`

```python
latest_crawl_path = Path("./data/latest_crawl_result.json")
raw_data_dir = Path("./data/raw_products")
```

Railway에서 이 경로들도 `/data/`를 가리켜야 함.

---

## 3. 데이터 흐름도

```
[Browser] /dashboard
    │
    ├─→ fetch(`${API_BASE}/api/data`)
    │       │
    │       └─→ GET /api/data (routes/data.py:23-30)
    │               │
    │               └─→ load_dashboard_data() (dependencies.py:182-191)
    │                       │
    │                       └─→ open("./data/dashboard_data.json")
    │                               │
    │                               ├─ Local: ./data/dashboard_data.json ✅
    │                               └─ Railway: /app/data/dashboard_data.json ❌
    │                                       │
    │                                       └─→ FileNotFoundError → {} → 404
    │
    ├─→ Fallback: fetch('../data/dashboard_data.json')
    │       └─→ StaticFiles 마운트 밖 → 404 ❌
    │
    └─→ dashboardData = null
            ├─→ updateActionTable() 미호출 → 스피너 무한
            ├─→ charts 데이터 없음 → 빈 차트
            └─→ 10초 후 오버레이만 숨겨짐 → 빈 화면
```

---

## 4. 해결 방법

### Fix 1: `src/api/dependencies.py` — Railway 경로 감지 (P0)

```python
# Before
DATA_PATH = "./data/dashboard_data.json"

# After
_DATA_DIR = "/data" if Path("/data").exists() else "./data"
DATA_PATH = f"{_DATA_DIR}/dashboard_data.json"
```

### Fix 2: `src/api/routes/data.py` — 하드코딩 경로 수정 (P0)

`_DATA_DIR`를 `dependencies.py`에서 import하여 모든 `./data/` 경로 교체.

### Fix 3: `dashboard/amore_unified_dashboard_v4.html` — 에러 UI (P1)

무효한 폴백 경로 제거, 데이터 로드 실패 시 사용자에게 명확한 에러 메시지 표시.

---

## 5. 영향받는 파일

| 파일 | 라인 | 문제 |
|------|------|------|
| `src/api/dependencies.py` | 178-179 | `DATA_PATH` 하드코딩 |
| `src/api/routes/data.py` | 458, 515 | 로컬 폴백 경로 하드코딩 |
| `dashboard/amore_unified_dashboard_v4.html` | 5681 | 프론트엔드 폴백 경로 잘못됨 |
| `dashboard/amore_unified_dashboard_v4.html` | 4303 | 액션 보드 무한 스피너 |
| `dashboard/amore_unified_dashboard_v4.html` | 12992-12998 | 10초 타임아웃이 문제를 숨김 |
