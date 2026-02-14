# 학술 기준 온톨로지 + RAG 업그레이드 플랜

> 작성일: 2026-02-14 | 상태: In Progress

## 목표

AMORE RAG-KG Hybrid Agent의 온톨로지/RAG 구현을 학술 기준에 맞게 업그레이드한다.

## 감사 결과 요약

| 영역 | 현재 점수 | 목표 |
|------|:--------:|:----:|
| OWL 2 온톨로지 | Partial (4 클래스, axiom 없음) | Full (restriction, disjointness, cardinality) |
| RDF/IRI 체계 | None (bare string ID) | Basic (namespace prefix + IRI) |
| SPARQL | Stub (빈 리스트 반환) | Basic (triple pattern + filter + join) |
| T-Box/A-Box 분리 | 용어만 (soft validation) | 강제 (hard validation) |
| Dense Retrieval | Implemented | Maintained |
| Sparse Retrieval (BM25) | None | Implemented + RRF |
| Self-RAG | None | Basic (검색 필요성 판단) |
| Multi-hop Retrieval | None | IRCoT 패턴 |
| 인라인 인용 (AIS) | None | 문장별 [출처N] |
| 하이브리드 통합 | 6.2/10 | 8.0/10 목표 |

## 병렬 실행 전략

```
Phase 0: 준비 (plan 파일 생성, 의존성 설치)
    │
    ├── Track A: OWL 온톨로지 심화 ──┐
    ├── Track B: RAG BM25+RRF      ├── 3트랙 병렬 실행
    ├── Track C: Multi-hop+SPARQL  ──┘
    │
    └── Track D: 통합 + IRI + 검증 (A,B,C 완료 후)
```

### Track A: OWL 온톨로지 심화
- A-1: OWL Class Restriction 정의 (DominantBrand ≡ Brand ⊓ ∃hasShareOfShelf[≥0.30])
- A-2: inverseOf 선언 (hasBrand ↔ hasProduct)
- A-3: Disjointness 공리 (AllDisjoint)
- A-4: Cardinality 제약 (Product → exactly 1 belongsToCategory)
- A-5: Hard Validation (T-Box 위반 시 triple 추가 거부)
- A-6: T-Box를 OWL에서 직접 읽기

### Track B: RAG 학술 기준 달성
- B-1: BM25 Sparse Retrieval 추가 (rank_bm25)
- B-2: RRF 병합 구현 (k=60)
- B-3: RRF 융합 전략 추가 (ConfidenceFusion)
- B-4: Self-RAG 검색 필요성 판단

### Track C: 에이전트 + Multi-hop 추론
- C-1: Multi-hop refine_search 액션
- C-2: IRCoT 패턴
- C-3: SPARQL 기본 지원
- C-4: 문장별 인라인 인용

### Track D: 통합 + 검증
- D-1: IRI 체계 도입
- D-2: KG IRI 마이그레이션
- D-3: HybridRetriever Self-RAG 통합
- D-4: 전체 통합 테스트
- D-5: OWL Consistency Check

## 파일 소유권 (충돌 방지)

| Track | 독점 파일 |
|-------|----------|
| A | `src/ontology/owl_reasoner.py`, `src/ontology/ontology_knowledge_graph.py` |
| B | `src/rag/retriever.py`, `src/rag/confidence_fusion.py` |
| C | `src/core/react_agent.py`, `src/ontology/kg_query.py`, `src/rag/context_builder.py` |
| D | `src/ontology/knowledge_graph.py`, `src/domain/entities/relations.py`, `src/rag/hybrid_retriever.py` |

## 검증 체크리스트

- [ ] OWL Consistency Check (Pellet/HermiT)
- [ ] BM25+RRF recall@10 비교
- [ ] Multi-hop 2-hop 질문 5개 테스트
- [ ] AIS 문장별 출처 매핑률 80%+
- [ ] `python3 -m pytest tests/ -v` 전체 통과
