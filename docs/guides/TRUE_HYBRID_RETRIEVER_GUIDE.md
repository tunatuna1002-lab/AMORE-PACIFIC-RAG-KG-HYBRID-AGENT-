# True Hybrid Retriever 구현 가이드

## 개요

`TrueHybridRetriever`는 RAG와 Ontology를 진정으로 통합한 하이브리드 검색 시스템입니다.

기존 `HybridRetriever`와의 주요 차이점:
- ✅ 필수 벡터 검색 (키워드 폴백 제거)
- ✅ OWL 추론 (owlready2 기반 진정한 추론)
- ✅ Entity Linking (쿼리 → 온톨로지 개념 매핑)
- ✅ Confidence Fusion (다중 소스 신뢰도 통합)
- ✅ Cross-Encoder Reranking (재순위화)

---

## 아키텍처

```
Query: "LANEIGE Lip Care 경쟁력 분석"
    │
    ├─→ EntityLinker ─→ Linked Entities
    │   - Brand: LANEIGE (confidence: 1.0)
    │   - Category: Lip Care (confidence: 1.0)
    │   - Indicator: (none)
    │
    ├─→ OWLReasoner ─→ Ontology-Guided Filters
    │   - target_brand: {"$in": ["laneige"]}
    │   - category: {"$in": ["lip_care"]}
    │
    ├─→ Query Expansion (LLM)
    │   - "LANEIGE Lip Care 경쟁력 분석"
    │   - "LANEIGE 립케어 시장 포지션"
    │   - "LANEIGE Lip Care competition analysis"
    │
    ├─→ Vector Search (filtered) ─→ Candidates (top 15)
    │   - Filter: target_brand=laneige, category=lip_care
    │   - 의미적 유사도 기반 검색
    │
    ├─→ Cross-Encoder Reranking ─→ Top 10
    │   - 쿼리-문서 쌍 직접 점수화
    │
    └─→ Confidence Fusion ─→ Final Results (top 5)
        - Vector: 0.4 weight
        - Ontology: 0.3 weight
        - Reranker: 0.3 weight
```

---

## 핵심 컴포넌트

### 1. EntityLinker (`src/rag/entity_linker.py`)

**기능:**
- 쿼리에서 엔티티 추출 (브랜드, 카테고리, 제품, 지표)
- 온톨로지 개념에 매핑
- 신뢰도 점수 계산

**사용 예:**
```python
from src.rag.entity_linker import EntityLinker, LinkedEntity

linker = EntityLinker(knowledge_graph=kg)
entities = linker.link("LANEIGE Lip Care 경쟁력 분석해줘")

# 결과:
# [
#   LinkedEntity(
#     text="LANEIGE",
#     entity_type="brand",
#     concept_uri="http://amorepacific.com/ontology/amore_brand.owl#Brand/LANEIGE",
#     concept_label="LANEIGE",
#     confidence=1.0
#   ),
#   LinkedEntity(
#     text="Lip Care",
#     entity_type="category",
#     concept_uri="http://amorepacific.com/ontology/amore_brand.owl#Category/lip_care",
#     concept_label="Lip Care",
#     confidence=1.0
#   )
# ]
```

**NER 방식:**
- spaCy 기반 (우선)
- 규칙 기반 (폴백)

**지원 엔티티:**
- Brand: LANEIGE, COSRX, TIRTIR 등
- Category: Lip Care, Skin Care 등
- Indicator: SoS, HHI, CPI 등
- Ingredient: Peptide, Ceramide, PDRN 등
- Trend: Morning Shade, Glass Skin, Viral 등
- Product: ASIN (B0XXXXXXXX)

---

### 2. ConfidenceFusion (`src/rag/confidence_fusion.py`)

**기능:**
- 다중 소스 점수 통합 (Vector + Ontology + Reranker)
- 가중치 기반 융합
- 상충 정보 감지

**융합 전략:**
- `WEIGHTED_SUM` (기본): 가중합
- `HARMONIC_MEAN`: 조화평균 (보수적)
- `GEOMETRIC_MEAN`: 기하평균
- `MAX_SCORE`: 최대값 (낙관적)

**사용 예:**
```python
from src.rag.confidence_fusion import ConfidenceFusion, ScoredDocument

fusion = ConfidenceFusion(
    weights={
        "vector": 0.4,
        "ontology": 0.3,
        "reranker": 0.3
    },
    method="weighted"
)

fused_docs = fusion.fuse(
    vector_results=vector_results,
    ontology_results=ontology_results,
    reranker_results=reranked_results,
    entity_confidence=0.9
)

# 결과:
# [
#   ScoredDocument(
#     id="doc_001",
#     content="...",
#     scores={"vector": 0.85, "ontology": 0.78, "reranker": 0.92},
#     combined_score=0.85,
#     rank=1
#   ),
#   ...
# ]
```

---

### 3. TrueHybridRetriever (`src/rag/true_hybrid_retriever.py`)

**메인 API:**

```python
from src.rag.true_hybrid_retriever import TrueHybridRetriever

retriever = TrueHybridRetriever(
    knowledge_graph=kg,
    owl_reasoner=owl_reasoner,
    use_semantic_chunking=True,
    use_reranking=True,
    use_query_expansion=True,
    fusion_method="weighted"
)

await retriever.initialize()

# 하이브리드 검색 실행
result = await retriever.retrieve(
    query="LANEIGE Lip Care 경쟁력 분석",
    current_metrics=dashboard_data,
    top_k=5
)

# 결과 구조
print(result.confidence)          # 0.87
print(result.entity_links)        # [LinkedEntity(...), ...]
print(result.ontology_context)    # {"inferences": [...], "facts": [...]}
print(result.documents)            # [{"id": "...", "content": "...", "score": 0.85}, ...]
print(result.combined_context)    # LLM 프롬프트용 통합 컨텍스트
```

---

## 검색 파이프라인 상세

### Step 1: Entity Linking

```python
entities = self._link_entities(query)
# → [LinkedEntity(text="LANEIGE", type="brand", confidence=1.0), ...]
```

### Step 2: Ontology-Guided Filters

```python
ontology_filters = self._build_ontology_filters(entities)
# → {"target_brand": {"$in": ["laneige"]}, "category": {"$in": ["lip_care"]}}
```

### Step 3: Query Expansion (LLM)

```python
expanded_queries = await self._expand_query(query)
# → ["LANEIGE Lip Care 경쟁력 분석", "LANEIGE 립케어 시장 포지션", ...]
```

### Step 4: Ontology-Guided Vector Search

```python
vector_results = await self._ontology_guided_search(
    expanded_queries,
    ontology_filters,
    top_k=15  # 재순위화를 위해 많이 검색
)
```

**필터 적용:**
- ChromaDB `where` 조건으로 벡터 검색 범위 제한
- 메타데이터 매칭 (`target_brand`, `category`, `doc_type`)

### Step 5: OWL Ontology Reasoning

```python
ontology_context = await self._infer_with_ontology(entities, current_metrics)
# → {
#   "inferences": [
#     {"type": "market_position", "brand": "laneige", "position": "StrongBrand"},
#     {"type": "competition", "brand": "laneige", "competitors": ["cosrx", "tirtir"]}
#   ],
#   "facts": [...]
# }
```

### Step 6: Cross-Encoder Reranking

```python
reranked_results = await self._rerank(query, vector_results, top_k=10)
# → OpenAI 또는 CrossEncoder 모델로 재순위화
```

### Step 7: Confidence Fusion

```python
fused_docs = self._fuse_results(
    vector_results=vector_results[:10],
    ontology_results=ontology_context.get("related_docs", []),
    reranked_results=reranked_results,
    entity_confidence=0.9
)
# → ScoredDocument 리스트 (combined_score 기준 정렬)
```

### Step 8: Combined Context 생성

```python
result.combined_context = self._build_combined_context(result)
# → LLM 프롬프트로 전달할 통합 컨텍스트 문자열
```

---

## HybridResult 데이터 클래스

```python
@dataclass
class HybridResult:
    query: str
    documents: List[Dict[str, Any]]        # 최종 검색 결과
    ontology_context: Dict[str, Any]       # 온톨로지 추론 결과
    entity_links: List[LinkedEntity]       # 연결된 엔티티
    confidence: float                      # 전체 신뢰도 (0.0 ~ 1.0)
    combined_context: str                  # LLM 프롬프트용
    metadata: Dict[str, Any]               # 검색 메타데이터

    def to_dict(self) -> Dict[str, Any]:
        """기존 HybridContext 호환 형식으로 변환"""
        return {
            "query": self.query,
            "entities": {...},
            "ontology_facts": self.ontology_context.get("facts", []),
            "inferences": self.ontology_context.get("inferences", []),
            "rag_chunks": self.documents,
            "combined_context": self.combined_context,
            "metadata": self.metadata
        }
```

---

## 기존 HybridRetriever와의 호환성

`TrueHybridRetriever`는 기존 `HybridRetriever`와 API 호환성을 유지합니다.

**마이그레이션 방법:**

```python
# Before (기존)
from src.rag.hybrid_retriever import HybridRetriever

retriever = HybridRetriever(kg, reasoner, doc_retriever)
await retriever.initialize()
context = await retriever.retrieve(query, current_metrics)

# After (새로운)
from src.rag.true_hybrid_retriever import TrueHybridRetriever

retriever = TrueHybridRetriever(knowledge_graph=kg)
await retriever.initialize()
result = await retriever.retrieve(query, current_metrics)

# 호환성 변환
context = result.to_dict()  # 기존 HybridContext 형식
```

---

## 통합 예제

```python
from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.owl_reasoner import OWLReasoner
from src.rag.true_hybrid_retriever import TrueHybridRetriever

# 1. 초기화
kg = KnowledgeGraph()
owl_reasoner = OWLReasoner()
retriever = TrueHybridRetriever(
    knowledge_graph=kg,
    owl_reasoner=owl_reasoner,
    use_semantic_chunking=True,
    use_reranking=True,
    use_query_expansion=True
)

await retriever.initialize()

# 2. 검색 실행
result = await retriever.retrieve(
    query="LANEIGE Lip Care SoS가 떨어진 이유를 분석해줘",
    current_metrics=dashboard_data,
    top_k=5
)

# 3. 결과 활용
print(f"신뢰도: {result.confidence:.2f}")
print(f"추출된 엔티티: {len(result.entity_links)}개")
print(f"검색된 문서: {len(result.documents)}개")
print(f"온톨로지 추론: {len(result.ontology_context.get('inferences', []))}개")

# 4. LLM 프롬프트 생성
llm_prompt = f"""
사용자 질문: {result.query}

관련 정보:
{result.combined_context}

위 정보를 바탕으로 답변해주세요.
"""
```

---

## Graceful Degradation

각 컴포넌트가 실패해도 시스템은 계속 동작합니다.

| 컴포넌트 실패 | 대응 방식 |
|------------|---------|
| EntityLinker | 규칙 기반 폴백 (spaCy 없을 시) |
| OWLReasoner | 기존 OntologyReasoner로 폴백 |
| Reranker | 원래 순서 유지 (벡터 검색 점수 사용) |
| Query Expansion | 원본 쿼리만 사용 |
| Vector Search | 오류 발생 (필수) |

---

## 성능 최적화

### 1. 벡터 검색 캐싱 (TTL 5분)

```python
# DocumentRetriever에 내장
cache_key = f"{query}:{top_k}:{doc_filter}"
if cache_key in self._search_cache and self._is_cache_valid(cache_key):
    return self._search_cache[cache_key]
```

### 2. 배치 임베딩

```python
# 쿼리 리스트 한 번에 임베딩
embeddings = self._embed_texts(expanded_queries)
```

### 3. 비동기 처리

```python
# 여러 검색을 동시에 수행
results = await asyncio.gather(
    self._ontology_guided_search(...),
    self._infer_with_ontology(...),
    self._expand_query(...)
)
```

---

## 확장 가능성

### Custom EntityLinker

```python
from src.rag.entity_linker import EntityLinker

class MyEntityLinker(EntityLinker):
    def _extract_with_rules(self, text: str):
        # 커스텀 추출 로직
        entities = super()._extract_with_rules(text)
        # ... 추가 로직
        return entities

retriever = TrueHybridRetriever(
    entity_linker=MyEntityLinker()
)
```

### Custom Fusion Strategy

```python
from src.rag.confidence_fusion import ConfidenceFusion, FusionStrategy

fusion = ConfidenceFusion(
    weights={"vector": 0.5, "ontology": 0.3, "reranker": 0.2},
    method="weighted",  # or "rrf", "hybrid"
    normalization="min_max"  # or "softmax", "z_score"
)

retriever = TrueHybridRetriever(
    confidence_fusion=fusion
)
```

---

## 주의사항

1. **벡터 검색 필수**: ChromaDB와 OpenAI Embeddings 필요
2. **OWL 추론 선택**: owlready2 미설치 시 기존 OntologyReasoner로 폴백
3. **메모리 사용량**: 대용량 온톨로지 로드 시 주의
4. **API 비용**: Query Expansion과 Reranking은 OpenAI API 사용

---

## 테스트

```python
import pytest
from src.rag.true_hybrid_retriever import TrueHybridRetriever

@pytest.mark.asyncio
async def test_true_hybrid_retriever():
    retriever = TrueHybridRetriever()
    await retriever.initialize()

    result = await retriever.retrieve(
        query="LANEIGE Lip Care 분석",
        top_k=3
    )

    assert result.confidence > 0.0
    assert len(result.documents) > 0
    assert len(result.entity_links) > 0
    assert result.combined_context != ""
```

---

## 참고 자료

- `src/rag/hybrid_retriever.py`: 기존 하이브리드 검색기
- `src/rag/entity_linker.py`: 엔티티 연결 모듈
- `src/rag/confidence_fusion.py`: 신뢰도 융합 모듈
- `src/ontology/owl_reasoner.py`: OWL 추론 엔진
- `src/rag/chunker.py`: 시맨틱 청킹
- `src/rag/reranker.py`: Cross-Encoder 재순위화
