# True RAG-Ontology Hybrid Integration 계획서

> **목표**: 현재 v3 챗봇의 가짜 RAG/Ontology를 진짜 구현으로 교체
> **작성일**: 2026-01-24

---

## 1. 현재 상태 분석

### 1.1 문제점

| 구성 요소 | 현재 (가짜) | 목표 (진짜) |
|----------|------------|------------|
| **RAG** | 키워드 매칭 (정규식) | Vector Embedding + Semantic Search |
| **Ontology** | IF-THEN 하드코딩 규칙 | OWL 2 + Pellet/HermiT 추론 |
| **KG** | Python Dict + List | Triple Store + 관계 추론 |

### 1.2 이미 구현되어 있지만 연결 안 된 것

```
src/rag/true_hybrid_retriever.py    ← 진짜 구현 (미사용)
src/ontology/owl_reasoner.py        ← OWL 2 추론기 (미사용)
src/rag/entity_linker.py            ← 엔티티 링킹 (미사용)
src/rag/confidence_fusion.py        ← 신뢰도 융합 (미사용)
src/rag/reranker.py                 ← Cross-Encoder (미사용)
```

---

## 2. 목표 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        사용자 질문                               │
│                 "LANEIGE vs COSRX 경쟁력 비교"                   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                    1. Entity Linking                              │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ "LANEIGE" → Brand:LANEIGE (confidence: 1.0)                 │ │
│  │ "COSRX"   → Brand:COSRX (confidence: 1.0)                   │ │
│  │ "경쟁력"  → Metric:competition_analysis (confidence: 0.9)    │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│  2. RAG       │   │ 3. Knowledge  │   │ 4. OWL        │
│  (Vector)     │   │    Graph      │   │    Ontology   │
│               │   │               │   │               │
│ ChromaDB +    │   │ 브랜드-제품   │   │ owlready2 +   │
│ OpenAI        │   │ 경쟁 관계     │   │ Pellet 추론   │
│ Embeddings    │   │ 카테고리      │   │               │
│               │   │               │   │ Brand         │
│ docs/guides/  │   │ LANEIGE       │   │ ├─Dominant    │
│ 문서 검색     │   │ ──has──→      │   │ ├─Strong      │
│               │   │ Lip Mask      │   │ └─Niche       │
│ "SoS란..."    │   │ ──competes──→ │   │               │
│ "시장지배..."  │   │ COSRX         │   │ SoS → 분류   │
└───────┬───────┘   └───────┬───────┘   └───────┬───────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                    5. Confidence Fusion                           │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ RAG Score (0.85) × 0.4 +                                    │ │
│  │ KG Coverage (0.90) × 0.3 +                                  │ │
│  │ Ontology Inference (0.95) × 0.3                             │ │
│  │ = Final Confidence: 0.89                                    │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                    6. Cross-Encoder Reranking                     │
│  최종 문서 순위 재조정 (query-document relevance)                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                    7. Combined Context                            │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ ## 온톨로지 추론 결과                                        │ │
│  │ - LANEIGE: NicheBrand (SoS 5.2%)                            │ │
│  │ - COSRX: StrongBrand (SoS 15.3%)                            │ │
│  │ - 경쟁관계: LANEIGE ↔ COSRX (대칭)                           │ │
│  │                                                              │ │
│  │ ## 관련 지식 (RAG)                                           │ │
│  │ - "SoS가 15% 미만이면 틈새 브랜드로 분류..."                   │ │
│  │ - "경쟁사 대비 SoS가 낮으면 마케팅 강화 필요..."               │ │
│  │                                                              │ │
│  │ ## 실시간 데이터 (크롤링)                                     │ │
│  │ - LANEIGE 평균순위: 42위, Top10 제품: 2개                    │ │
│  │ - COSRX 평균순위: 28위, Top10 제품: 5개                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────────────┐
│                         8. LLM (GPT-4)                            │
│                                                                   │
│  "LANEIGE는 현재 SoS 5.2%로 틈새(Niche) 포지션입니다.             │
│   경쟁사 COSRX는 SoS 15.3%로 강자(Strong) 포지션을 차지하며,      │
│   직접적인 위협이 됩니다.                                         │
│                                                                   │
│   전략적 권고:                                                    │
│   1. 핵심 제품(Lip Sleeping Mask)에 마케팅 집중                   │
│   2. COSRX의 Snail Mucin 대항 신제품 검토                         │
│   3. Top 10 진입 제품 확대 (현재 2개 → 5개 목표)"                  │
└───────────────────────────────────────────────────────────────────┘
```

---

## 3. 구현 단계

### Phase 1: 인프라 준비 (필수 의존성 설치)

```bash
# 필수 패키지
pip install chromadb           # Vector Store
pip install owlready2          # OWL 2 추론
pip install sentence-transformers  # Cross-Encoder Reranking

# 선택 패키지 (성능 향상)
pip install spacy              # NER 기반 Entity Linking
python -m spacy download en_core_web_sm
```

**확인 사항:**
- [ ] ChromaDB 초기화 테스트
- [ ] OpenAI Embeddings API 연결 테스트
- [ ] owlready2 + Pellet 추론 테스트

---

### Phase 2: RAG 벡터 검색 활성화

**현재 문제:**
```python
# src/rag/retriever.py:46
VECTOR_SEARCH_AVAILABLE = None  # 비활성화 상태
```

**수정 작업:**

1. **DocumentRetriever 초기화 수정**
   - `use_vector_search=True` 기본값 변경
   - ChromaDB PersistentClient 설정
   - OpenAI Embeddings 연결

2. **문서 인덱싱**
   ```python
   # docs/guides/ 문서들을 벡터화하여 ChromaDB에 저장
   await doc_retriever.initialize()  # 벡터 인덱스 생성
   ```

3. **시맨틱 검색 구현**
   ```python
   # 키워드 매칭 → 벡터 유사도 검색
   results = await doc_retriever.search(query, top_k=5)
   ```

**테스트:**
```python
retriever = DocumentRetriever(use_vector_search=True)
await retriever.initialize()
results = await retriever.search("SoS 지표 해석 방법")
assert len(results) > 0
assert results[0]["score"] > 0.7
```

---

### Phase 3: OWL Ontology 연결

**현재 문제:**
- `OWLReasoner` 구현되어 있지만 v3에서 사용 안 함
- `cosmetics_ontology.owl` 파일 존재하지만 로드 안 함

**수정 작업:**

1. **OWL 파일 초기화**
   ```python
   owl_reasoner = OWLReasoner(
       owl_file="src/ontology/cosmetics_ontology.owl",
       reasoner_type="pellet"
   )
   await owl_reasoner.initialize()
   ```

2. **크롤링 데이터 → OWL 마이그레이션**
   ```python
   # 크롤링 후 KG 데이터를 OWL로 변환
   owl_reasoner.import_from_knowledge_graph(kg)
   owl_reasoner.import_from_metrics(metrics_data)

   # 추론 실행
   owl_reasoner.run_reasoner()
   owl_reasoner.infer_market_positions()
   ```

3. **추론 결과 활용**
   ```python
   # 브랜드 시장 포지션 자동 분류
   position = owl_reasoner.get_brand_market_position("LANEIGE")
   # → "NicheBrand" (SoS 5.2% < 15%)

   # 경쟁사 조회
   competitors = owl_reasoner.get_competitors("LANEIGE")
   # → ["COSRX", "TIRTIR", "SKIN1004"]
   ```

**테스트:**
```python
owl_reasoner = OWLReasoner()
await owl_reasoner.initialize()
owl_reasoner.add_brand("LANEIGE", sos=0.052)
owl_reasoner.run_reasoner()
position = owl_reasoner.get_brand_market_position("LANEIGE")
assert position == "NicheBrand"
```

---

### Phase 4: TrueHybridRetriever를 v3에 연결

**현재 코드 (src/core/simple_chat.py):**
```python
from src.rag.hybrid_retriever import HybridRetriever  # 가짜
```

**변경 후:**
```python
from src.rag.true_hybrid_retriever import TrueHybridRetriever  # 진짜

class SimpleChatService:
    async def _get_hybrid_retriever(self):
        if self._hybrid_retriever is None:
            self._hybrid_retriever = TrueHybridRetriever(
                owl_reasoner=OWLReasoner(),
                knowledge_graph=KnowledgeGraph(),
                doc_retriever=DocumentRetriever(use_vector_search=True)
            )

        if not self._retriever_initialized:
            await self._hybrid_retriever.initialize()
            self._retriever_initialized = True

        return self._hybrid_retriever
```

---

### Phase 5: Entity Linking 통합

**현재 문제:**
- `EntityLinker` 구현되어 있지만 사용 안 함

**수정 작업:**

1. **쿼리 전처리에 Entity Linking 추가**
   ```python
   entity_linker = EntityLinker(knowledge_graph=kg)
   entities = entity_linker.link("LANEIGE Lip Care 경쟁력 분석")

   # entities = [
   #   LinkedEntity(text="LANEIGE", type="brand", confidence=1.0),
   #   LinkedEntity(text="Lip Care", type="category", confidence=1.0),
   #   LinkedEntity(text="경쟁력", type="metric", confidence=0.9)
   # ]
   ```

2. **Ontology-Guided 필터 생성**
   ```python
   filters = entity_linker.get_ontology_filters(entities)
   # → {"brand": {"$in": ["LANEIGE"]}, "category": {"$in": ["lip_care"]}}
   ```

3. **벡터 검색에 필터 적용**
   ```python
   results = await doc_retriever.search(
       query=query,
       where=filters  # ChromaDB 필터
   )
   ```

---

### Phase 6: Confidence Fusion 적용

**현재 문제:**
- `ConfidenceFusion` 구현되어 있지만 사용 안 함

**수정 작업:**

```python
from src.rag.confidence_fusion import ConfidenceFusion

fusion = ConfidenceFusion()
fused_results = fusion.fuse(
    vector_results=rag_results,
    ontology_results=owl_inferences,
    reranker_results=reranked_docs,
    entity_confidence=entity_linker.avg_confidence,
    weights={
        "vector": 0.4,
        "ontology": 0.3,
        "reranker": 0.3
    }
)
```

---

### Phase 7: 크롤링 데이터 → KG/Ontology 자동 반영

**현재 문제:**
- 크롤링 데이터가 KG에 자동 반영되지 않음

**수정 작업:**

1. **크롤링 후 훅 추가 (orchestrator.py)**
   ```python
   async def on_crawl_complete(crawl_result):
       # 1. KnowledgeGraph 업데이트
       kg.load_from_crawl_data(crawl_result)

       # 2. OWL Ontology 동기화
       owl_reasoner.import_from_knowledge_graph(kg)
       owl_reasoner.run_reasoner()
       owl_reasoner.infer_market_positions()

       # 3. 영속화
       owl_reasoner.save("data/ontology/amore_brand.owl")
       kg.save("data/kg/knowledge_graph.json")
   ```

2. **브랜드 관계 자동 생성**
   ```python
   # 같은 카테고리 Top 10 브랜드 → 경쟁 관계
   for category in categories:
       top_brands = kg.get_top_brands(category, limit=10)
       for i, brand1 in enumerate(top_brands):
           for brand2 in top_brands[i+1:]:
               owl_reasoner.add_competitor_relation(brand1, brand2)
   ```

---

## 4. 파일 변경 목록

### 수정할 파일

| 파일 | 변경 내용 |
|------|----------|
| `src/core/simple_chat.py` | HybridRetriever → TrueHybridRetriever |
| `src/rag/retriever.py` | 벡터 검색 기본 활성화 |
| `orchestrator.py` | 크롤링 후 KG/OWL 동기화 훅 |
| `src/infrastructure/container.py` | OWLReasoner, TrueHybridRetriever DI 추가 |

### 새로 생성할 파일

| 파일 | 용도 |
|------|------|
| `src/ontology/cosmetics_ontology.owl` | 화장품 도메인 OWL 스키마 (이미 존재하면 검증) |
| `data/ontology/amore_brand.owl` | 런타임 OWL 데이터 |
| `data/vectordb/` | ChromaDB 영속 저장소 |

---

## 5. 테스트 계획

### 단위 테스트

```python
# tests/unit/rag/test_true_hybrid_retriever.py
async def test_retrieve_returns_hybrid_result():
    retriever = TrueHybridRetriever()
    await retriever.initialize()
    result = await retriever.retrieve("LANEIGE 경쟁력 분석")

    assert result.confidence > 0.5
    assert len(result.documents) > 0
    assert len(result.entity_links) > 0
    assert "LANEIGE" in result.combined_context

# tests/unit/ontology/test_owl_reasoner.py
def test_infer_market_position():
    reasoner = OWLReasoner()
    reasoner.add_brand("TEST_BRAND", sos=0.05)
    reasoner.run_reasoner()
    position = reasoner.get_brand_market_position("TEST_BRAND")
    assert position == "NicheBrand"
```

### 통합 테스트

```python
# tests/integration/test_chat_with_true_hybrid.py
async def test_chat_uses_ontology_inference():
    response = await chat_service.chat("LANEIGE 시장 포지션은?")

    assert "NicheBrand" in response["text"] or "틈새" in response["text"]
    assert response["confidence"] > 0.7
```

---

## 6. 롤백 계획

문제 발생 시 환경 변수로 폴백:

```python
USE_TRUE_HYBRID = os.getenv("USE_TRUE_HYBRID", "true").lower() == "true"

if USE_TRUE_HYBRID:
    from src.rag.true_hybrid_retriever import TrueHybridRetriever as Retriever
else:
    from src.rag.hybrid_retriever import HybridRetriever as Retriever
```

---

## 7. 예상 결과

### Before (현재)

```
Q: LANEIGE vs COSRX 경쟁력 비교

A: LANEIGE의 SoS는 5.2%이고, COSRX의 SoS는 8.1%입니다.
   (단순 수치 나열, 추론 없음)
```

### After (목표)

```
Q: LANEIGE vs COSRX 경쟁력 비교

A: 온톨로지 분석 결과:

   ■ 시장 포지션
   - LANEIGE: NicheBrand (SoS 5.2% < 15%)
   - COSRX: StrongBrand (SoS 15.3% ≥ 15%)

   ■ 경쟁 관계
   LANEIGE ↔ COSRX는 Lip Care 카테고리에서 직접 경쟁 중
   COSRX가 순위(28위)와 점유율(15.3%)에서 우위

   ■ 전략 가이드 (RAG 문서)
   "SoS 격차가 10%p 이상이면 적극적 마케팅 필요"
   "경쟁사 주력 제품 대항 신제품 검토 권고"

   ■ 권고사항
   1. Lip Sleeping Mask 마케팅 강화
   2. COSRX Snail Mucin 대항 제품 개발 검토
   3. 신제품으로 Top 10 진입 확대

   (신뢰도: 0.89)
```

---

## 8. 일정

| Phase | 작업 | 예상 시간 |
|-------|------|----------|
| 1 | 인프라 준비 | 30분 |
| 2 | RAG 벡터 검색 활성화 | 2시간 |
| 3 | OWL Ontology 연결 | 2시간 |
| 4 | TrueHybridRetriever 연결 | 1시간 |
| 5 | Entity Linking 통합 | 1시간 |
| 6 | Confidence Fusion 적용 | 30분 |
| 7 | 크롤링 → KG/OWL 동기화 | 1시간 |
| - | 테스트 및 검증 | 2시간 |
| **총합** | | **약 10시간** |

---

## 9. 핵심 정리

이 에이전트의 **핵심 가치**:

1. **RAG**: 화장품 시장 지식 문서(docs/guides/)를 벡터 검색하여 정의/해석 기준 제공
2. **Knowledge Graph**: 브랜드-제품-카테고리-경쟁관계를 구조화하여 관계 질의 지원
3. **OWL Ontology**: 도메인 규칙(SoS 기반 포지션 분류)을 형식적으로 정의하고 자동 추론
4. **크롤링 데이터**: 실시간 Amazon 베스트셀러 데이터로 KG/Ontology 업데이트

**결과**: 단순 수치 나열이 아닌, **추론 기반 전략적 인사이트** 제공
