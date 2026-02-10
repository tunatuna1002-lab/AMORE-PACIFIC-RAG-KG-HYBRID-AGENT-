# scripts/ - 운영 스크립트

## 개요

프로젝트 운영에 필요한 유틸리티 스크립트 모음입니다.
평가, 동기화, 백업, 마이그레이션, 데이터 검증 등을 자동화합니다.

## 파일 목록

| 파일 | LOC | 설명 |
|------|-----|------|
| `evaluate_golden.py` | ~300 | 골든셋 평가 실행 |
| `sync_from_railway.py` | ~150 | Railway → 로컬 동기화 |
| `sync_sheets_to_sqlite.py` | ~200 | Sheets → SQLite 동기화 |
| `sync_to_railway.py` | ~150 | 로컬 → Railway 동기화 |
| `backup_sqlite.py` | ~100 | SQLite 백업 |
| `check_duplicates.py` | ~80 | 중복 데이터 검사 |
| `export_dashboard.py` | ~120 | 대시보드 데이터 내보내기 |
| `fix_summer_brand.py` | ~60 | Summer Fridays 브랜드명 정규화 |
| `generate_insight_sample.py` | ~100 | 인사이트 샘플 생성 |
| `migrate_excel_to_sheets.py` | ~150 | Excel → Sheets 마이그레이션 |
| `migrate_kg_to_ontology.py` | ~200 | KG → Ontology 마이그레이션 |
| `run_evaluation.py` | ~250 | 전체 평가 파이프라인 실행 |
| `test_report_generator.py` | ~100 | 테스트 리포트 생성 |
| `AGENTS.md` | - | 현재 문서 |

## 평가 스크립트

### evaluate_golden.py

골든셋 30개 쿼리를 평가하고 L1-L5 메트릭을 측정합니다.

**사용법**:

```bash
# 기본 실행
python3 scripts/evaluate_golden.py

# Verbose 모드 (상세 로그)
python3 scripts/evaluate_golden.py --verbose

# HTML 리포트 생성
python3 scripts/evaluate_golden.py --format html --output results/report.html

# 특정 카테고리만 평가
python3 scripts/evaluate_golden.py --category factual

# 회귀 테스트 (베이스라인 비교)
python3 scripts/evaluate_golden.py --baseline results/baseline.json
```

**주요 기능**:
- L1-L5 5단계 평가
- 골든셋 로딩 및 검증
- 메트릭 집계 및 리포트
- 회귀 감지

**출력 예시**:

```
=== Golden Set Evaluation Results ===
Total Queries: 30
Overall Score: 0.84

Level Scores:
  L1 (Query Quality): 0.90
  L2 (Retrieval): 0.80
  L3 (KG): 0.85
  L4 (Ontology): 0.78
  L5 (Answer): 0.87

Category Breakdown:
  Factual: 0.92 (10 queries)
  Analytical: 0.81 (10 queries)
  Comparative: 0.75 (5 queries)
  Temporal: 0.88 (5 queries)

Report saved to: results/evaluation_2026-02-10.json
```

### run_evaluation.py

전체 평가 파이프라인을 실행합니다 (골든셋 + 회귀 + 비용 추적).

**사용법**:

```bash
# 전체 평가 실행
python3 scripts/run_evaluation.py

# 비용 추적 활성화
TRACK_COST=true python3 scripts/run_evaluation.py

# 베이스라인 저장
python3 scripts/run_evaluation.py --save-baseline results/baseline.json
```

### test_report_generator.py

테스트 커버리지 및 결과를 HTML 리포트로 생성합니다.

**사용법**:

```bash
# 리포트 생성
python3 scripts/test_report_generator.py

# 출력 경로 지정
python3 scripts/test_report_generator.py --output reports/test_report.html
```

## 동기화 스크립트

### sync_from_railway.py

Railway Volume (`/data/`) → 로컬 (`./data/`)로 데이터를 동기화합니다.

**사용법**:

```bash
# 전체 동기화
python3 scripts/sync_from_railway.py

# SQLite만 동기화
python3 scripts/sync_from_railway.py --db-only

# KG만 동기화
python3 scripts/sync_from_railway.py --kg-only

# Dry-run (실제 복사 없음)
python3 scripts/sync_from_railway.py --dry-run
```

**동기화 대상**:
- `amore_data.db` (SQLite 데이터베이스)
- `knowledge_graph.json` (Knowledge Graph)
- `backups/kg/` (KG 백업)

**주의사항**:
- Railway CLI 설치 필요: `npm install -g @railway/cli`
- 로그인 필요: `railway login`
- 프로젝트 링크: `railway link`

### sync_to_railway.py

로컬 (`./data/`) → Railway Volume (`/data/`)로 데이터를 업로드합니다.

**사용법**:

```bash
# 전체 업로드
python3 scripts/sync_to_railway.py

# 확인 없이 강제 업로드
python3 scripts/sync_to_railway.py --force
```

**주의**: Railway DB 덮어쓰기 위험 (백업 필수!)

### sync_sheets_to_sqlite.py

Google Sheets → SQLite로 데이터를 동기화합니다.

**사용법**:

```bash
# 전체 시트 동기화
python3 scripts/sync_sheets_to_sqlite.py

# 특정 시트만 동기화
python3 scripts/sync_sheets_to_sqlite.py --sheet products
```

**환경 변수**:

```bash
GOOGLE_SPREADSHEET_ID=...
GOOGLE_SHEETS_CREDENTIALS_JSON=...
```

**동기화 순서**:
1. Sheets API로 데이터 읽기
2. SQLite 트랜잭션 시작
3. 기존 데이터 삭제 (TRUNCATE)
4. 새 데이터 삽입
5. 커밋

## 백업 스크립트

### backup_sqlite.py

SQLite 데이터베이스를 백업합니다.

**사용법**:

```bash
# 기본 백업 (data/backups/sqlite/)
python3 scripts/backup_sqlite.py

# 백업 경로 지정
python3 scripts/backup_sqlite.py --output /path/to/backup.db

# 압축 백업
python3 scripts/backup_sqlite.py --compress
```

**백업 파일명**: `amore_data_2026-02-10_12-00-00.db`

**보관 정책**: 7일 롤링 (자동 삭제)

## 데이터 검증 스크립트

### check_duplicates.py

데이터베이스 중복 레코드를 검사합니다.

**사용법**:

```bash
# 전체 테이블 검사
python3 scripts/check_duplicates.py

# 특정 테이블만 검사
python3 scripts/check_duplicates.py --table products

# 중복 자동 삭제
python3 scripts/check_duplicates.py --fix
```

**검사 대상**:
- `products` (ASIN + Category 중복)
- `metrics` (Date + Category 중복)
- `insights` (Generated_at 중복)

**출력 예시**:

```
=== Duplicate Check Results ===
Table: products
  Duplicates found: 3
  Affected ASINs: B001, B002, B003

Table: metrics
  Duplicates found: 0

Total duplicates: 3
```

### fix_summer_brand.py

"Summer Friday" → "Summer Fridays" 브랜드명을 정규화합니다.

**사용법**:

```bash
# Dry-run (확인만)
python3 scripts/fix_summer_brand.py --dry-run

# 실제 수정
python3 scripts/fix_summer_brand.py
```

**수정 대상**:
- `products` 테이블
- `knowledge_graph.json`
- `config/asin_brand_mapping.json`

## 마이그레이션 스크립트

### migrate_excel_to_sheets.py

Excel 파일을 Google Sheets로 마이그레이션합니다.

**사용법**:

```bash
# Excel → Sheets
python3 scripts/migrate_excel_to_sheets.py --excel data/amore_data.xlsx
```

**지원 형식**: `.xlsx`, `.xls`

### migrate_kg_to_ontology.py

구버전 KG를 Ontology 형식으로 마이그레이션합니다.

**사용법**:

```bash
# 마이그레이션 실행
python3 scripts/migrate_kg_to_ontology.py

# 검증 모드 (마이그레이션 후 무결성 검사)
python3 scripts/migrate_kg_to_ontology.py --validate
```

**변환 규칙**:
- `hasCategory` → `belongsTo`
- `competesIn` → `competesInCategory`
- 타입 추가: `Product`, `Brand`, `Category`

## 내보내기 스크립트

### export_dashboard.py

대시보드 데이터를 JSON/CSV로 내보냅니다.

**사용법**:

```bash
# JSON 내보내기
python3 scripts/export_dashboard.py --format json --output data/export.json

# CSV 내보내기
python3 scripts/export_dashboard.py --format csv --output data/export.csv

# 특정 카테고리만
python3 scripts/export_dashboard.py --category lip_care
```

**출력 구조**:

```json
{
  "categories": {
    "beauty": {
      "products": [...],
      "metrics": {...}
    }
  },
  "latest_insight": {...},
  "exported_at": "2026-02-10T12:00:00"
}
```

### generate_insight_sample.py

샘플 인사이트를 생성합니다 (개발/테스트용).

**사용법**:

```bash
# 샘플 생성
python3 scripts/generate_insight_sample.py

# 카테고리 지정
python3 scripts/generate_insight_sample.py --category lip_care

# LLM 사용 (실제 생성)
python3 scripts/generate_insight_sample.py --use-llm
```

## 일괄 실행 스크립트

### 일일 백업

```bash
#!/bin/bash
# scripts/daily_backup.sh

# SQLite 백업
python3 scripts/backup_sqlite.py

# KG 백업 (자동 실행됨)
# python3 -m src.tools.kg_backup backup

# Railway 동기화
python3 scripts/sync_to_railway.py --force

echo "Backup completed at $(date)"
```

### 데이터 정합성 검사

```bash
#!/bin/bash
# scripts/data_integrity_check.sh

# 중복 검사
python3 scripts/check_duplicates.py

# 브랜드명 정규화
python3 scripts/fix_summer_brand.py --dry-run

# Ontology 검증
python3 -m eval.validators.ontology_validator

echo "Integrity check completed"
```

## 스케줄링

### Cron 설정 (로컬)

```bash
# crontab -e

# 매일 오전 2시 백업
0 2 * * * cd /path/to/project && python3 scripts/backup_sqlite.py

# 매일 오전 3시 중복 검사
0 3 * * * cd /path/to/project && python3 scripts/check_duplicates.py --fix

# 매주 월요일 평가 실행
0 9 * * 1 cd /path/to/project && python3 scripts/evaluate_golden.py --verbose
```

### Railway Cron (클라우드)

Railway는 내장 Cron을 지원하지 않으므로 GitHub Actions 사용 권장.

## 주의사항

1. **백업 필수**: 운영 DB 수정 전 반드시 백업
2. **Dry-run 우선**: `--dry-run`으로 먼저 확인
3. **Railway CLI**: 동기화 스크립트는 Railway CLI 필요
4. **환경 변수**: Sheets 동기화는 Google API 키 필요
5. **권한**: 파일 쓰기 권한 확인 (`data/`, `results/`)

## 에러 핸들링

```python
# scripts/sync_from_railway.py 예시
try:
    subprocess.run(["railway", "run", "cp", ...], check=True)
except subprocess.CalledProcessError as e:
    logger.error(f"Railway 명령 실패: {e}")
    sys.exit(1)
except FileNotFoundError:
    logger.error("Railway CLI가 설치되지 않았습니다")
    sys.exit(1)
```

## 로깅

모든 스크립트는 표준 로깅을 사용합니다:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)
```

## 참고

- `data/` - 데이터 저장 경로
- `results/` - 평가 결과 저장
- `eval/` - 평가 프레임워크
- `CLAUDE.md` - 운영 명령어
