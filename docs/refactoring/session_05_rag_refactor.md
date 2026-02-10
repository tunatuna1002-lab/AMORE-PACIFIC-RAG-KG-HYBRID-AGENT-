# Session 5: RAG 모듈 리팩토링 (중복 통합)

> ⏱ 예상 시간: 40~50분 | 위험도: 🟡 중간 | 선행 조건: Session 4 완료

---

## 프롬프트 (아래를 복사해서 새 Claude Code 세션에 붙여넣기)

```
너는 20년 베테랑 Python 개발자이자 RAG (Retrieval-Augmented Generation) 전문가야. AMORE RAG-KG Hybrid Agent의 RAG 모듈을 리팩토링하는 세션이야.

## 이번 세션 목표
`src/rag/`의 중복을 제거하고, 검색 파이프라인을 깔끔하게 정리해.

## 컨텍스트
- 프로젝트: `/Users/leedongwon/Desktop/AMORE-RAG-ONTOLOGY-HYBRID AGENT/`
- 전체 마스터 플랜: `docs/refactoring/00_MASTER_PLAN.md` 참조
- Python 3.13.7 (`python3` 사용)
- 이 모듈은 `src/domain` + `src/ontology`만 의존 (Clean Architecture 준수)
- Session 4에서 ontology 모듈이 정리되었음

## 현재 구조 & 문제점
```
src/rag/
├── hybrid_retriever.py   # 1,184줄 — KG + RAG 통합 검색
├── retriever.py          # 1,173줄 — 기본 문서 검색 (중복?)
├── reranker.py           # Reranking 로직
├── chunker.py            # 텍스트 분할
├── confidence_fusion.py  # 신뢰도 융합
├── entity_linker.py      # 엔티티 링킹
├── query_rewriter.py     # 쿼리 재작성
├── router.py             # 라우팅
├── templates.py          # 프롬프트 템플릿
└── __init__.py
```

총 6,207줄. 문제:
1. `retriever.py`(1173줄) vs `hybrid_retriever.py`(1184줄) — 기능 중복 가능
2. 파이프라인 흐름이 명확하지 않음

## 수행할 작업 (TDD 방식)

### 1. 중복 분석: retriever.py vs hybrid_retriever.py
- 두 파일을 비교. `retriever.py`가 순수 RAG만, `hybrid_retriever.py`가 RAG+KG인지 확인
- 어디서 import되는지 추적
- 통합 가능하면 `hybrid_retriever.py`를 메인으로, `retriever.py`의 고유 기능은 흡수
- 또는 `retriever.py`를 base class로 두고 `hybrid_retriever.py`가 상속하는 구조도 가능

### 2. 파이프라인 명확화
검색 파이프라인의 단계를 명확하게:
```
Query → [query_rewriter] → [router] → [retriever/hybrid_retriever] → [reranker] → [confidence_fusion] → Result
                                              ↑
                                        [entity_linker]
```
- 이 흐름이 코드에서 명확하게 보이도록 정리
- 각 단계가 독립적으로 테스트 가능하도록

### 3. templates.py 검토
- 프롬프트 템플릿이 여기에 있는 게 맞는지 확인
- 프롬프트는 `prompts/` 폴더와 중복될 수 있음

### 4. 테스트 보강
- `tests/unit/rag/` 테스트 확인 및 보강
- 각 파이프라인 단계별 단위 테스트
- ChromaDB는 반드시 mock

### 5. Import 경로 호환성
- `__init__.py`에서 re-export:
  ```python
  from .hybrid_retriever import HybridRetriever
  from .reranker import Reranker
  # ...
  ```

### 6. 검증
- `python3 -m pytest tests/unit/rag/ -v` — RAG 테스트 통과
- `python3 -m pytest tests/ -v --tb=short` — 전체 테스트 통과
- import 경로 확인

## 주의사항
- ChromaDB 관련 코드는 이후 Session 7에서 infrastructure/adapters로 이동 가능
- 이번 세션에서는 rag/ 내부 정리에 집중
- Context7 MCP로 ChromaDB 최신 문서 참조 가능
```

---

## 체크리스트

- [ ] retriever.py vs hybrid_retriever.py 분석 및 통합/정리
- [ ] 파이프라인 흐름 명확화
- [ ] templates.py 위치 검토
- [ ] 테스트 보강
- [ ] `__init__.py` re-export 설정
- [ ] 전체 테스트 통과
