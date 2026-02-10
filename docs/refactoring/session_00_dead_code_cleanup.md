# Session 0: Dead Code 삭제 + 폴더 정리

> ⏱ 예상 시간: 20~30분 | 위험도: 🟢 낮음 | 선행 조건: 없음

---

## 프롬프트 (아래를 복사해서 새 Claude Code 세션에 붙여넣기)

```
너는 20년 베테랑 Python 개발자이고, AMORE RAG-KG Hybrid Agent 코드베이스의 대대적인 리팩토링을 진행 중이야.

## 이번 세션 목표
Dead Code 삭제와 폴더 정리. 기능 변경 없이 코드베이스를 깔끔하게 만드는 것이 목표야.

## 컨텍스트
- 프로젝트: `/Users/leedongwon/Desktop/AMORE-RAG-ONTOLOGY-HYBRID AGENT/`
- 전체 마스터 플랜: `docs/refactoring/00_MASTER_PLAN.md` 참조
- 의존성 분석: `DEPENDENCY_INDEX.md`, `DEPENDENCY_GRAPH.txt`, `FILE_IMPORT_MAP.txt` 참조
- Python 3.13.7 (`python3` 사용)

## 수행할 작업 (순서대로)

### 1. Dead Code 삭제
다음을 삭제해줘:
- `_deprecated/` 폴더 전체
- `_backup_unused_modules/` 폴더 전체
- 프로젝트 루트의 테스트 스크립트들 (tests/ 안에 없는 것들):
  - `test_brain_import.py`
  - `test_cache_integration.py`
  - `test_embedding_cache.py`
  - `test_failed_signals.py`
  - `run_type_flow_tests.py`
- 비어있는 `__init__.py` 스텁 파일들 중 실제 사용되지 않는 것들 확인 후 삭제

### 2. 중복 파일 확인
다음 파일 쌍을 비교해서 어떤 것이 실제 사용 중인지 확인해줘:
- `src/agents/hybrid_insight_agent.py` vs `src/agents/true_hybrid_insight_agent.py`
- `src/ontology/reasoner.py` vs `src/ontology/owl_reasoner.py`
- `src/rag/retriever.py` vs `src/rag/hybrid_retriever.py`
- `src/core/batch_workflow.py` vs `src/application/workflows/batch_workflow.py`

각 쌍에 대해:
- 어디서 import 되는지 추적
- 실제 사용되는 파일과 미사용 파일 식별
- 미사용 파일은 삭제, 단 "이 파일은 Session 4/5에서 통합 예정"이라는 주석을 남겨도 됨

### 3. 불필요한 루트 파일 정리
- `generate_insight_sample.py` → 필요한지 확인
- `export_dashboard.py` → 필요한지 확인
- `migrate_excel_to_sheets.py` → 일회성 스크립트면 `scripts/`로 이동

### 4. 검증
- `python3 -m pytest tests/ -v --tb=short` 실행해서 기존 테스트 깨지지 않는지 확인
- import 에러 없는지 확인: `python3 -c "from dashboard_api import app; print('OK')"`

## 주의사항
- 기능을 변경하지 마. 삭제만 해.
- 삭제 전에 반드시 grep으로 import 여부 확인
- git에 이력이 있으니 삭제해도 복구 가능
- 삭제한 파일 목록을 마지막에 정리해서 알려줘
```

---

## 이 세션의 체크리스트

- [ ] `_deprecated/` 삭제됨
- [ ] `_backup_unused_modules/` 삭제됨
- [ ] 루트 테스트 스크립트 삭제됨
- [ ] 중복 파일 분석 완료
- [ ] 기존 테스트 통과 확인
- [ ] 삭제된 파일 목록 기록됨

## 예상 결과
- 약 2000줄+ 코드 제거
- 폴더 구조 간소화
- 인지 부하 감소
