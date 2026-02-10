# Session 7: Tools 분할 + Core 순환 의존성 해소

> ⏱ 예상 시간: 60~90분 | 위험도: 🔴 높음 | 선행 조건: Session 2, 3, 4, 5 모두 완료

---

## 프롬프트 (아래를 복사해서 새 Claude Code 세션에 붙여넣기)

```
너는 20년 베테랑 Python 개발자이자 소프트웨어 아키텍트야. AMORE RAG-KG Hybrid Agent에서 가장 복잡한 리팩토링 세션이야: tools 분할과 순환 의존성 해소.

## 이번 세션 목표
1. `src/tools/` (38개 파일, 19K줄)를 하위 패키지로 분할
2. `src/core/ ↔ src/tools/` 순환 의존성 해소
3. `src/tools/ ↔ src/api/` 순환 의존성 해소

## 컨텍스트
- 프로젝트: `/Users/leedongwon/Desktop/AMORE-RAG-ONTOLOGY-HYBRID AGENT/`
- 전체 마스터 플랜: `docs/refactoring/00_MASTER_PLAN.md` 참조
- 의존성 그래프: `DEPENDENCY_GRAPH.txt` 참조
- Python 3.13.7 (`python3` 사용)
- Session 2에서 Protocol 인터페이스 추가됨
- Session 3에서 Application Use Case 추가됨

## 핵심 순환 의존성 (해소 대상)
```
src.core → src.tools → src.api → src.core  (3-way 순환)
src.tools → src.agents → src.tools          (2-way 순환)
src.tools → src.api → src.tools             (2-way 순환)
```

## 수행할 작업

### Part A: tools/ 하위 패키지 분할

현재 38개 파일이 평면적으로 나열됨. 역할별로 분류:

```
src/tools/
├── scrapers/              # 웹 크롤링
│   ├── __init__.py
│   ├── amazon_product_scraper.py   (← amazon_scraper.py 리네임)
│   ├── amazon_scraper.py           (← 둘 다 있으면 통합)
│   ├── deals_scraper.py
│   ├── tiktok_collector.py
│   ├── instagram_collector.py
│   ├── youtube_collector.py
│   └── reddit_collector.py
├── collectors/            # 데이터 수집 (비-크롤링)
│   ├── __init__.py
│   ├── external_signal_collector.py
│   ├── google_trends_collector.py
│   └── public_data_collector.py
├── calculators/           # 순수 계산 로직
│   ├── __init__.py
│   ├── metric_calculator.py
│   └── period_analyzer.py
├── storage/               # 저장소
│   ├── __init__.py
│   └── sqlite_storage.py
├── exporters/             # 내보내기
│   ├── __init__.py
│   ├── dashboard_exporter.py
│   ├── report_generator.py
│   └── insight_formatter.py
├── notifications/         # 알림
│   ├── __init__.py
│   ├── email_sender.py
│   ├── telegram_bot.py
│   └── alert_service.py
├── utilities/             # 범용 유틸리티
│   ├── __init__.py
│   ├── brand_resolver.py
│   ├── kg_backup.py
│   ├── data_integrity_checker.py
│   ├── reference_tracker.py
│   └── insight_verifier.py
├── __init__.py            # re-export (호환성)
└── job_queue.py           # 작업 큐 (어디로?)
```

**작업 순서:**
1. 먼저 각 파일을 읽고 실제 역할 확인 (위 분류가 맞는지)
2. 하위 디렉토리 생성
3. 파일 이동
4. `__init__.py` re-export로 기존 import 호환

### Part B: 순환 의존성 해소

#### B-1: tools → agents 의존 제거
- `src/tools/` 안에서 `from src.agents import ...` 하는 곳을 찾아줘
- 대부분 콜백이나 참조일 것. Protocol로 대체:
  ```python
  # Before (순환!)
  from src.agents.alert_agent import AlertAgent

  # After (Protocol 사용)
  from src.domain.interfaces.agent import AlertAgentProtocol
  ```

#### B-2: tools → api 의존 제거
- `src/tools/` 안에서 `from src.api import ...` 하는 곳을 찾아줘
- api를 직접 참조하면 안 됨. 이벤트/콜백 패턴으로 대체

#### B-3: core → tools 의존 정리
- `src/core/brain.py`에서 tools를 직접 import하는 곳을 Protocol로 대체:
  ```python
  # Before
  from src.tools.metric_calculator import MetricCalculator

  # After
  from src.domain.interfaces.metric import MetricCalculatorProtocol

  class UnifiedBrain:
      def __init__(self, metric_calc: MetricCalculatorProtocol, ...):
          self.metric_calc = metric_calc
  ```

### Part C: core/ 정리

`src/core/` (24개 파일, 8075줄)도 검토:
- `brain.py` (1787줄) — God Object. 스케줄링/라우팅/오케스트레이션 분리 검토
- `batch_workflow.py` — Session 3의 Application 워크플로우와 중복?
- 나머지 파일들 역할 확인 및 정리

### Part D: 테스트
- 이동된 모든 파일에 대해 import 테스트
- 기존 테스트 전부 통과 확인
- 순환 의존성이 실제로 해소되었는지 검증:
  ```python
  # 순환 의존성 검증 스크립트
  python3 -c "
  import importlib
  modules = ['src.core', 'src.tools', 'src.agents', 'src.api']
  for m in modules:
      importlib.import_module(m)
  print('No circular import errors!')
  "
  ```

### Part E: 검증
- `python3 -m pytest tests/ -v --tb=short` — 전체 테스트 통과
- `python3 -c "from dashboard_api import app; print('OK')"` — 서버 import 확인
- 순환 의존성 검증 스크립트 실행

## 주의사항
- 이 세션이 가장 위험함. 한 번에 다 바꾸지 말고 단계별로:
  1. tools/ 분할 → 테스트 → 커밋
  2. 순환 의존성 해소 → 테스트 → 커밋
  3. core/ 정리 → 테스트 → 커밋
- `__init__.py` re-export는 필수 (기존 import 경로 깨지면 안 됨)
- brain.py의 대규모 분할은 이번 세션에서 "시작"만 하고, 완전한 분할은 선택사항
- 변경이 너무 크면 Part A만 하고 Part B-C는 다음 세션으로 미뤄도 됨
```

---

## 체크리스트

- [ ] tools/ 파일 역할 분류 완료
- [ ] 하위 패키지 생성 및 파일 이동
- [ ] `__init__.py` re-export 설정
- [ ] tools → agents 순환 의존성 해소
- [ ] tools → api 순환 의존성 해소
- [ ] core → tools Protocol 적용
- [ ] core/ 파일 정리
- [ ] 순환 의존성 검증 통과
- [ ] 전체 테스트 통과

## 주의: 세션 분할 가능
이 세션이 너무 크면 Part A(tools 분할)와 Part B-C(순환 해소)를 별도 세션으로 나눠도 됨.
