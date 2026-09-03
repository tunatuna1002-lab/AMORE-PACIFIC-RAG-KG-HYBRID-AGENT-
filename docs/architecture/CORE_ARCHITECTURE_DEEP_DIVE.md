# AMORE RAG-KG-Ontology Hybrid Agent: 핵심 아키텍처 심층 분석

> 이 문서는 시스템의 3대 핵심 축인 **RAG**, **Knowledge Graph**, **Ontology Reasoning**의 내부 구조와 통합 방식을 상세하게 설명합니다.

---

## 목차

1. [시스템 전체 아키텍처](#1-시스템-전체-아키텍처)
2. [RAG (Retrieval-Augmented Generation)](#2-rag-retrieval-augmented-generation)
3. [Knowledge Graph (Triple Store)](#3-knowledge-graph-triple-store)
4. [Ontology Reasoning (추론 엔진)](#4-ontology-reasoning-추론-엔진)
5. [Hybrid Retrieval (3자 통합)](#5-hybrid-retrieval-3자-통합)
6. [에이전트 통합 레이어](#6-에이전트-통합-레이어)
7. [End-to-End 쿼리 처리 파이프라인](#7-end-to-end-쿼리-처리-파이프라인)
8. [성능 최적화](#8-성능-최적화)

---

## 1. 시스템 전체 아키텍처

```
┌──────────────────────────────────────────────────────────────────────┐
│                        User Query                                   │
│                  "LANEIGE Lip Care 경쟁력 분석해줘"                    │
└──────────────────────────┬───────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    UnifiedBrain (Orchestrator)                       │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────────┐   │
│  │ Cache Check  │→│ Complexity   │→│ Mode Selection            │   │
│  │             │  │ Detection    │  │ Simple → Direct           │   │
│  │             │  │             │  │ Complex → ReAct            │   │
│  └─────────────┘  └──────────────┘  └──────────────────────────┘   │
└──────────────────────────┬───────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Hybrid Retrieval (병렬 실행)                       │
│                                                                      │
│  ┌─────────────────┐  ┌──────────────────┐  ┌────────────────────┐  │
│  │  RAG Documents  │  │  Knowledge Graph  │  │ Ontology Reasoning │  │
│  │  (ChromaDB)     │  │  (Triple Store)   │  │ (Business Rules)   │  │
│  │                 │  │                   │  │                    │  │
│  │ • 14개 MD 문서   │  │ • 50,000 트리플    │  │ • 40+ 추론 규칙     │  │
│  │ • 벡터 검색      │  │ • 38개 관계 유형    │  │ • OWL 2 클래스     │  │
│  │ • Cross-Encoder │  │ • BFS 탐색        │  │ • Forward Chaining │  │
│  └─────────────────┘  └──────────────────┘  └────────────────────┘  │
└──────────────────────────┬───────────────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    Context Assembly & LLM Generation                 │
│  온톨로지 추론 > KG 팩트 > RAG 청크 > 외부 신호 (우선순위 순)           │
└──────────────────────────────────────────────────────────────────────┘
```

### 핵심 설계 원칙

| 원칙 | 설명 |
|------|------|
| **Hybrid-First** | 단일 소스가 아닌 RAG + KG + Ontology 3자 결합으로 응답 품질 극대화 |
| **Intent-Aware** | 쿼리 의도(분석/트렌드/위기/지표)에 따라 검색 전략을 동적으로 변경 |
| **Explainable** | 모든 추론에 근거(evidence)와 신뢰도(confidence)를 명시 |
| **Cost-Efficient** | 임베딩 캐시, 검색 캐시, TTL 기반 만료로 API 비용 33%+ 절감 |

---

## 2. RAG (Retrieval-Augmented Generation)

### 2.1 문서 소스 (14개 MD 파일)

시스템은 5가지 유형의 참조 문서를 보유합니다:

| 유형 | 코드 | 설명 | 예시 |
|------|------|------|------|
| **Metric Guide** | `D` | KPI 지표 해석 가이드 | Strategic Indicators, Metric Interpretation |
| **Playbook** | `A` | 분석 플레이북 | 아마존 랭킹 급등 원인 역추적 보고서 |
| **Intelligence** | `B` | 시장 인텔리전스 | K-beauty 트렌드, US 뷰티 트렌드 |
| **Response Guide** | `C` | 위기 대응 가이드 | 부정적 이슈 대응, 인플루언서 매핑 |
| **IR Report** | `E` | 분기 실적 보고서 | AP_1Q25_EN, AP_2Q25_EN, AP_3Q25_EN |

### 2.2 임베딩 & 벡터 저장소

```
문서 → Chunking → Embedding → ChromaDB 저장 → 코사인 유사도 검색
```

**임베딩 모델**: `text-embedding-3-small` (OpenAI)

**문서 유형별 청크 크기**:

| 문서 유형 | 청크 크기 (tokens) | 이유 |
|-----------|-------------------|------|
| Playbook | 800 | 분석 맥락 보존을 위해 큰 청크 |
| Intelligence | 600 | 시장 동향 단위 보존 |
| Response Guide | 500 | 대응 절차 단위 |
| Metric Guide | 500 | 지표 정의 단위 |
| IR Report | 700 | 테이블 포함 보고서 |

**테이블 특별 처리**: 마크다운 테이블은 분리하지 않고 통째로 하나의 청크로 유지합니다.

```python
# 테이블 패턴 감지 → 청크 분리 방지
table_pattern = r"(\|[^\n]+\|\n(?:\|[-:| ]+\|\n)?(?:\|[^\n]+\|\n)+)"
```

### 2.3 시맨틱 청킹 (Semantic Chunking)

단순한 고정 길이 분할이 아닌, **문장 간 의미 유사도**를 기반으로 청크 경계를 결정합니다.

```
1. 문서를 문장 단위로 분리
2. 각 문장을 OpenAI 임베딩으로 변환
3. 인접 문장 간 코사인 유사도 계산
4. 유사도가 급격히 떨어지는 지점(하위 25%)을 경계로 설정
5. 경계 기준으로 청크 생성 (min/max 크기 제약 적용)
```

이 방식의 장점:
- 의미적 일관성(coherence) 보존
- 문장 중간 절단 방지
- 주제 전환 지점에서 자연스럽게 분리

### 2.4 검색 파이프라인 (2-Stage Retrieval)

```
Query → Query Expansion → Vector Search → Cross-Encoder Reranking → Top-K 결과
```

**Stage 1: Bi-Encoder (빠른 후보 선별)**

```python
# ChromaDB 코사인 유사도 검색 (상위 100개)
results = chromadb_collection.query(
    query_embeddings=[query_embedding],
    n_results=100
)
```

**Stage 2: Cross-Encoder (정밀 재순위)**

```python
# sentence-transformers/ms-marco-MiniLM-L-6-v2
pairs = [(query, doc.text) for doc in candidates]
scores = cross_encoder.predict(pairs, batch_size=32)
# 상위 5개만 최종 선택
```

Cross-Encoder가 없을 경우 OpenAI LLM 기반 스코어링으로 폴백합니다.

### 2.5 쿼리 확장 (Query Expansion)

원본 쿼리를 LLM으로 확장하여 검색 재현율을 높입니다:

```
원본: "LANEIGE Lip Care 경쟁력"
확장: "LANEIGE Lip Care 경쟁력 분석 시장 포지션 SoS 점유율 해석"
```

### 2.6 임베딩 캐시 (API 비용 절감)

```python
# MD5 해시 기반 캐시 키 생성
text_hash = hashlib.md5(text.encode()).hexdigest()

# 캐시 히트 → OpenAI API 호출 생략
if text_hash in self._embedding_cache:
    return self._embedding_cache[text_hash]  # 즉시 반환

# 캐시 미스 → API 호출 후 저장
embedding = openai.embeddings.create(model="text-embedding-3-small", input=text)
self._embedding_cache[text_hash] = embedding

# FIFO 방식 자동 만료 (최대 1000개)
if len(self._embedding_cache) >= 1000:
    oldest_key = next(iter(self._embedding_cache))
    del self._embedding_cache[oldest_key]
```

| 메트릭 | 값 |
|--------|-----|
| 최대 캐시 크기 | 1,000 entries |
| 엔트리당 메모리 | ~6 KB |
| 전체 메모리 사용량 | ~6 MB |
| API 비용 절감 | 33%+ (반복 쿼리 기준) |

---

## 3. Knowledge Graph (Triple Store)

### 3.1 핵심 데이터 구조

Knowledge Graph는 **RDF 스타일 트리플(Subject-Predicate-Object)** 을 인메모리에 저장합니다.

```python
# 트리플 표현
Relation(
    subject="LANEIGE",              # 주어 (브랜드)
    predicate=RelationType.HAS_PRODUCT,  # 술어 (관계)
    object="B08XYZ123",             # 목적어 (ASIN)
    properties={                    # 속성
        "rank": 8,
        "price": 29.99,
        "rating": 4.7
    },
    confidence=0.95,                # 신뢰도
    source="crawler"                # 출처
)
```

### 3.2 관계 유형 (38가지)

| 관계 | 예시 | 설명 |
|------|------|------|
| `HAS_PRODUCT` | LANEIGE → B08XYZ | 브랜드가 제품을 보유 |
| `BELONGS_TO_CATEGORY` | B08XYZ → lip_care | 제품이 카테고리에 속함 |
| `COMPETES_WITH` | LANEIGE ↔ COSRX | 경쟁 관계 (양방향) |
| `PARENT_CATEGORY` | lip_care → skin_care | 카테고리 계층 구조 |
| `HAS_SENTIMENT` | B08XYZ → "Moisturizing" | 제품 감성 태그 |
| `OWNED_BY_GROUP` | COSRX → AMOREPACIFIC | 기업 소유 관계 |
| `SIBLING_BRAND` | LANEIGE ↔ Sulwhasoo | 같은 그룹 브랜드 |
| `HAS_TREND` | LANEIGE → "viral_tiktok" | 트렌드 연결 |

### 3.3 인덱스 구조 (O(1) 조회)

```python
# 3중 인덱스로 어떤 방향에서든 빠른 조회 가능
subject_index:   Dict[str, List[Relation]]       # "LANEIGE" → 모든 나가는 관계
object_index:    Dict[str, List[Relation]]        # "B08XYZ" → 모든 들어오는 관계
predicate_index: Dict[RelationType, List[Relation]] # HAS_PRODUCT → 모든 제품 관계
```

**조회 예시**:

```python
# LANEIGE의 모든 제품 조회
products = kg.query(subject="LANEIGE", predicate=RelationType.HAS_PRODUCT)

# lip_care 카테고리의 모든 브랜드 조회
brands = kg.query(predicate=RelationType.BELONGS_TO_CATEGORY, object_="lip_care")

# 특정 ASIN의 모든 관계 조회
all_relations = kg.get_neighbors("B08XYZ", direction="both")
```

### 3.4 그래프 탐색 (BFS)

```python
# 너비 우선 탐색으로 관련 엔티티 발견
result = kg.bfs_traverse(
    start_entity="LANEIGE",
    max_depth=3,
    predicate_filter=[RelationType.COMPETES_WITH, RelationType.HAS_PRODUCT]
)

# 결과:
# depth 0: ["LANEIGE"]
# depth 1: ["COSRX", "TIRTIR", "B08XYZ", "B09ABC"]  (경쟁사 + 제품)
# depth 2: ["B11DEF", "B12GHI"]                       (경쟁사 제품)
```

### 3.5 스마트 Eviction (메모리 관리)

트리플 수가 50,000개를 초과하면 **중요도 점수** 기반으로 하위 10%를 자동 삭제합니다.

```python
def _calculate_importance(self, relation):
    score = 0.5  # 기본 점수

    # 보호 관계 (절대 삭제 안 됨)
    if relation.predicate in [OWNED_BY_GROUP, PARENT_CATEGORY]:
        score += 10.0

    # 높은 신뢰도
    score += relation.confidence * 2.0

    # 브랜드 관련 관계 (중요)
    if relation.predicate in [HAS_PRODUCT, COMPETES_WITH]:
        score += 1.0

    # 최근 데이터 (7일 이내)
    if age < timedelta(days=7):
        score += 0.5

    return score
```

| 보호 등급 | 관계 유형 | 삭제 가능 여부 |
|-----------|-----------|---------------|
| **절대 보호** | OWNED_BY_GROUP, PARENT_CATEGORY | 불가 |
| **높은 보호** | HAS_PRODUCT, COMPETES_WITH | 거의 불가 |
| **일반** | HAS_SENTIMENT, HAS_TREND | 7일 후 가능 |
| **낮은 보호** | 오래된 저신뢰도 트리플 | 우선 삭제 대상 |

### 3.6 영속성 (Railway Volume)

```python
# Railway 환경 자동 감지
is_railway = bool(os.environ.get("RAILWAY_ENVIRONMENT"))

# 저장 경로
persist_path = "/data/knowledge_graph.json" if is_railway else "data/knowledge_graph.json"

# 자동 저장 (100개 변경 누적 시)
kg.add_relation(...)  # dirty flag 설정
# 100개 추가 후 → 자동으로 persist_path에 JSON 저장
```

### 3.7 도메인 특화 API

```python
# 브랜드 분석
products = kg.get_brand_products("LANEIGE", category="lip_care")
competitors = kg.get_competitors("LANEIGE", competition_type="direct")
metadata = kg.get_entity_metadata("LANEIGE")  # SoS, 평균 순위 등

# 카테고리 분석
brands = kg.get_category_brands("lip_care", min_products=3)
hierarchy = kg.get_category_hierarchy("lip_care")
# → {name: "Lip Care", level: 2, ancestors: ["Skin Care", "Beauty"], descendants: [...]}

# 감성 분석
sentiments = kg.get_product_sentiments("B08XYZ")
# → {ai_summary: "...", tags: ["Moisturizing", "Value"], clusters: {"Hydration": [...]}}
profile = kg.get_brand_sentiment_profile("LANEIGE")  # 전 제품 감성 집계
```

---

## 4. Ontology Reasoning (추론 엔진)

### 4.1 아키텍처 개요

시스템에는 두 가지 추론 엔진이 있습니다:

| 엔진 | 파일 | 방식 | 규칙 수 |
|------|------|------|---------|
| **OntologyReasoner** | `reasoner.py` | Rule-based Forward Chaining | 40+ |
| **OWLReasoner** | `owl_reasoner.py` | OWL 2 + Pellet/HermiT | 클래스 분류 |

### 4.2 추론 규칙 구조

```python
InferenceRule(
    name="market_dominance_fragmented",
    conditions=[
        StandardConditions.sos_above(0.15),  # SoS >= 15%
        StandardConditions.hhi_below(0.15),  # 분산 시장 (HHI < 0.15)
    ],
    conclusion=lambda ctx: {
        "insight": f"{ctx['brand']}은 분산 시장에서 {ctx['sos']*100}% SoS로 지배적 위치",
        "recommendation": "시장 통합 전 점유율 확대 필요",
        "related_entities": [ctx["brand"]],
    },
    insight_type=InsightType.MARKET_DOMINANCE,
    priority=10,
    confidence=0.9
)
```

### 4.3 추론 규칙 카테고리 (40+)

#### 시장 포지션 규칙 (5개)

| 규칙 | 조건 | 결론 |
|------|------|------|
| `MARKET_DOMINANCE` | SoS >= 30% | "시장 지배적 브랜드" |
| `STRONG_POSITION` | 15% <= SoS < 30% | "강한 시장 포지션" |
| `CHALLENGER_POSITION` | 5% <= SoS < 15%, 순위 상승 중 | "도전자 포지션, 성장 잠재력" |
| `NICHE_PLAYER` | SoS < 5%, 특정 카테고리 집중 | "니치 플레이어" |
| `FRAGMENTED_DOMINANCE` | SoS >= 15%, HHI < 0.15 | "분산 시장 내 지배" |

#### 리스크/경고 규칙 (4개)

| 규칙 | 조건 | 결론 |
|------|------|------|
| `PRICE_QUALITY_MISMATCH` | 가격 상위 20% but 평점 하위 30% | "가격 대비 품질 괴리" |
| `RANK_DECLINE` | 순위 10위 이상 하락 | "순위 급락 경고" |
| `SOS_EROSION` | SoS 2%p 이상 하락 | "점유율 잠식 경고" |
| `COMPETITIVE_THREAT` | 경쟁사 SoS 급증 + 자사 하락 | "경쟁 위협 감지" |

#### 할인-순위 인과관계 규칙 (5개)

| 규칙 | 조건 | 결론 |
|------|------|------|
| `DISCOUNT_DEPENDENT` | 할인 기간과 순위 상승 80%+ 겹침 | "할인 의존형 (빨간 태그)" |
| `BRAND_DRIVEN` | 할인 없이도 순위 유지 | "브랜드 파워형 (녹색 태그)" |
| `MIXED_PATTERN` | 부분적 겹침 (40-80%) | "혼합형" |
| `VIRAL_EFFECT` | 소셜 언급 급증 후 순위 상승 | "바이럴 효과" |
| `PRIME_DAY_SPIKE` | Prime Day 전후 순위 급등 | "이벤트 의존형" |

```python
# 할인 의존도 점수 계산
def calculate_discount_dependency_score(product_history):
    correlation = 0
    for i in range(1, len(history)):
        if discount_increased(history[i]) and rank_improved(history[i]):
            correlation += 1
    score = (correlation / total_periods) * 100

    # 0-30: 낮음 (브랜드 주도)
    # 31-60: 중간
    # 61-100: 높음 (할인 의존)
    return score
```

#### 감성 분석 규칙 (8개)

| 규칙 | 조건 | 결론 |
|------|------|------|
| `SENTIMENT_HYDRATION` | 보습 클러스터 태그 2개 이상 | "보습력을 핵심 마케팅 메시지로" |
| `SENTIMENT_VALUE` | 가성비 태그 존재 | "가격 경쟁력 활용 전략" |
| `SENTIMENT_PACKAGING` | 패키징 관련 부정 태그 | "패키징 개선 필요" |
| `SENTIMENT_COMPARISON` | 경쟁 제품 대비 감성 차이 | "차별화 포인트 식별" |

#### IR(실적 보고서) 교차 분석 규칙 (6개)

| 규칙 | 조건 | 결론 |
|------|------|------|
| `IR_PRIME_DAY_IMPACT` | IR에서 Prime Day 언급 + 순위 10위 이상 상승 | "Prime Day 재고 확보 권고" |
| `IR_REVENUE_RANK_CORRELATION` | 매출 증가 + 순위 상승 동시 | "실적-순위 연동 확인" |
| `BRAND_OWNERSHIP_VERIFICATION` | OWNED_BY_GROUP 관계 존재 | "그룹 내 브랜드 시너지 분석" |

### 4.4 Forward Chaining 실행

```
1. 컨텍스트 수집 (브랜드, SoS, HHI, 순위, 가격 등)
2. Knowledge Graph에서 추가 컨텍스트 보강 (경쟁사, 제품 목록)
3. 우선순위 순으로 규칙 평가
4. 모든 조건 충족 시 → InferenceResult 생성
5. 결과에 근거(evidence)와 신뢰도(confidence) 첨부
```

```python
# 컨텍스트 보강 (자동)
enriched_context = {
    "brand": "LANEIGE",
    "sos": 0.052,
    "hhi": 0.08,
    "avg_rank": 15.2,
    "competitors": ["COSRX", "TIRTIR"],      # KG에서 자동 조회
    "products": ["B08XYZ", "B09ABC"],          # KG에서 자동 조회
    "brand_metadata": {"group": "AMOREPACIFIC"} # KG에서 자동 조회
}

# 규칙 평가 결과
InferenceResult(
    rule_name="challenger_position",
    insight="LANEIGE는 5.2% SoS로 도전자 포지션, 성장 잠재력 확인",
    confidence=0.85,
    evidence={
        "satisfied_conditions": ["sos_between_5_15", "rank_improving"],
        "data": {"sos": 0.052, "rank_trend": "improving"}
    }
)
```

### 4.5 쿼리 의도 기반 동적 규칙 필터링

모든 40+개 규칙을 매번 실행하지 않고, **쿼리 의도에 맞는 규칙만 선택적으로 실행**합니다.

```python
INTENT_KEYWORDS = {
    "competition": {
        "keywords": ["경쟁", "competitor", "vs", "비교"],
        "insight_types": [COMPETITIVE_THREAT, MARKET_POSITION]
    },
    "pricing": {
        "keywords": ["가격", "price", "cpi", "프리미엄"],
        "insight_types": [PRICE_POSITION, PRICE_QUALITY_GAP]
    },
    "sentiment": {
        "keywords": ["리뷰", "평판", "감성", "sentiment"],
        "insight_types": [SENTIMENT_STRENGTH, SENTIMENT_COMPARISON]
    },
    "trend": {
        "keywords": ["트렌드", "요즘", "최근", "변화"],
        "insight_types": [GROWTH_OPPORTUNITY, VIRAL_EFFECT]
    }
}

# "LANEIGE vs 경쟁사 가격 비교" 쿼리 시:
# → 의도 감지: ["competition", "pricing"]
# → COMPETITIVE_THREAT, MARKET_POSITION, PRICE_POSITION, PRICE_QUALITY_GAP 규칙만 실행
```

### 4.6 추론 결과 설명 (Explainability)

```python
explanation = reasoner.explain_inference(result)

# 출력 예시:
# ## 추론 규칙: challenger_position
# **설명**: SoS 5-15% 구간에서 순위 상승 추세는 도전자 포지션을 나타냄
#
# ### 적용된 조건
# - [O] 5% <= SoS < 15%
# - [O] 순위 상승 추세 확인
#
# ### 근거 데이터
# - sos: 0.052 (5.2%)
# - rank_trend: improving
#
# ### 결론
# **인사이트**: LANEIGE는 도전자 포지션에서 성장 잠재력 확인
# **권고**: 점유율 10% 돌파를 위한 공격적 마케팅 필요
# **신뢰도**: 0.85
```

### 4.7 OWL 2 추론 (선택적)

`owlready2` 라이브러리가 설치된 경우 형식 온톨로지 추론이 추가로 가동됩니다.

**OWL 클래스 계층**:

```
owl:Thing
├── Brand
│   ├── DominantBrand    (SoS >= 30%)
│   ├── StrongBrand      (15% <= SoS < 30%)
│   └── NicheBrand       (SoS < 15%)
├── Product
├── Category
└── Trend
```

**OWL 프로퍼티**:

| 유형 | 프로퍼티 | 특성 |
|------|---------|------|
| Object | `hasBrand` (Product → Brand) | inverse: hasProduct |
| Object | `competsWith` (Brand → Brand) | symmetric (자동 양방향) |
| Object | `belongsToCategory` (Product → Category) | |
| Data | `shareOfShelf` (Brand → float) | functional (단일값) |
| Data | `rank` (Product → int) | functional |
| Data | `price`, `rating` (Product → float) | functional |

```python
owl = OWLReasoner()
owl.add_brand("LANEIGE", sos=0.25)
owl.run_reasoner()  # Pellet/HermiT 분류기 실행

positions = owl.infer_market_positions()
# {"LANEIGE": "StrongBrand"}  → 15% <= 25% < 30%

# symmetric 관계 자동 추론
owl.add_competitor_relation("LANEIGE", "COSRX")
owl.get_competitors("COSRX")  # → ["LANEIGE"] (자동 추론됨)
```

`owlready2`가 없으면 기본 Rule-based Reasoner로 폴백합니다.

---

## 5. Hybrid Retrieval (3자 통합)

### 5.1 HybridRetriever 파이프라인

```python
async def retrieve(self, query, current_metrics, include_explanations=True):
    # 0. 쿼리 의도 분류
    query_intent = classify_intent(query)  # DIAGNOSIS / TREND / CRISIS / METRIC / GENERAL
    doc_type_filter = get_doc_type_filter(query_intent)

    # 1. 엔티티 추출
    entities = self.entity_extractor.extract(query, self.kg)
    # → brands: ["laneige"], categories: ["lip_care"], indicators: ["sos"]

    # 2. Knowledge Graph 조회
    kg_facts = self._query_knowledge_graph(entities)
    # → brand_info, competitors, category_hierarchy, sentiment_data

    # 3. Ontology 추론
    inference_context = self._build_inference_context(entities, current_metrics)
    inferences = self.reasoner.infer(inference_context)
    # → market_position, risk_alerts, recommendations

    # 4. RAG 문서 검색 (의도 기반 필터링)
    expanded_query = self._expand_query(query, inferences, entities)
    rag_results = await self.doc_retriever.search(
        expanded_query,
        top_k=5,
        doc_type_filter=doc_type_filter  # 의도에 맞는 문서 유형만
    )

    # 4-1. 폴백: 결과 < 3개이면 전체 문서에서 추가 검색
    if len(rag_results) < 3 and doc_type_filter:
        additional = await self.doc_retriever.search(
            expanded_query, top_k=5 - len(rag_results),
            doc_type_filter=None  # 필터 해제
        )

    # 5. 컨텍스트 결합
    return HybridContext(
        kg_triples=kg_facts,
        rag_docs=rag_results,
        inferences=inferences,
        entities=entities
    )
```

### 5.2 의도별 문서 유형 우선순위

| 쿼리 의도 | 1순위 | 2순위 | 3순위 |
|-----------|-------|-------|-------|
| **DIAGNOSIS** (분석) | Playbook | Metric Guide | Intelligence |
| **TREND** (트렌드) | Intelligence | Knowledge Base | Response Guide |
| **CRISIS** (위기) | Response Guide | Intelligence | Playbook |
| **METRIC** (지표) | Metric Guide | Playbook | Intelligence |
| **GENERAL** (일반) | 전체 (필터 없음) | - | - |

### 5.3 엔티티 추출 (EntityExtractor)

```python
# config/entities.json에서 동적 로드 (TTL: 300초)
entities = extractor.extract("LANEIGE Lip Care SoS 분석해줘", kg)

# 결과:
{
    "brands": ["laneige"],            # 정규화된 브랜드명
    "categories": ["lip_care"],        # 카테고리 ID
    "indicators": ["sos"],             # KPI 지표
    "products": [],                    # ASIN 패턴 (B0XXXXXXXX)
    "sentiments": [],                  # 감성 키워드
    "sentiment_clusters": []           # 감성 클러스터
}
```

### 5.4 TrueHybridRetriever (고급 버전)

엔티티 링킹과 신뢰도 융합을 추가한 고급 버전입니다.

**엔티티 링킹 (NER + Ontology)**:

```python
# spaCy NER 또는 규칙 기반 추출
entities = linker.link("LANEIGE Lip Care 경쟁력")

# 퍼지 매칭으로 브랜드 정규화
best_match, score = fuzzy_match("라네즈", ["laneige", "라네즈"])

# 온톨로지 개념 매핑
LinkedEntity(
    text="LANEIGE",
    entity_type="brand",
    concept_uri="http://amorepacific.com/ontology/amore_brand.owl#Brand/LANEIGE",
    confidence=1.0
)
```

**신뢰도 융합 (Multi-Source)**:

```python
# 3개 소스의 가중 결합
DEFAULT_WEIGHTS = {
    'vector': 0.40,     # 벡터 유사도 (의미적)
    'ontology': 0.35,   # 온톨로지 추론 (논리적)
    'entity': 0.25      # 엔티티 관계 (구조적)
}

# 점수 불일치 감지 (gap > 0.3)
if max_score - min_score > 0.3:
    warnings.append("점수 불일치 감지 - 소스 간 신뢰도 차이 큼")
```

---

## 6. 에이전트 통합 레이어

### 6.1 UnifiedBrain (오케스트레이터)

모든 에이전트를 통합 조율하는 최상위 컨트롤러입니다.

```python
async def process_query(self, query, session_id, current_metrics):
    # 1. 캐시 확인
    if cached := self.cache.get(query, "query"):
        return cached

    # 2. 컨텍스트 수집 (Hybrid Retrieval)
    context = await self._context_gatherer.gather(query, current_metrics)

    # 3. 복잡도 판단 → ReAct 모드 결정
    if self._react_agent and self._is_complex_query(query, context):
        response = await self._process_with_react(query, context)
    else:
        # 4. LLM 도구 선택 (DecisionMaker)
        decision = await self.decision_maker.decide(query, context, system_state)

        # 5. 도구 실행 (ToolCoordinator)
        tool_result = await self.tool_coordinator.execute(
            tool_name=decision["tool"],
            params=decision["tool_params"]
        )

        # 6. 응답 생성
        response = await self._generate_response(query, context, decision, tool_result)

    return response
```

### 6.2 모드 선택 매트릭스

| 조건 | 모드 | 설명 |
|------|------|------|
| 캐시 히트 | **Cached** | 즉시 반환 (LLM 호출 없음) |
| 단순 쿼리 | **Direct** | DecisionMaker → 도구 실행 → 응답 |
| 복잡 쿼리 | **ReAct** | 다단계 추론 루프 (최대 3회) |

### 6.3 복잡도 판단 휴리스틱

```python
def _is_complex_query(self, query, context):
    # 신호 1: 분석 키워드 포함
    complex_keywords = ["왜", "어떻게", "비교", "분석", "추천", "전략", "예측", "원인"]
    has_complex_keyword = any(kw in query for kw in complex_keywords)

    # 신호 2: 컨텍스트 부족 (RAG 결과 < 2개 또는 KG 트리플 없음)
    low_context = not context.rag_docs or len(context.rag_docs) < 2 or not context.kg_triples

    # 신호 3: 다단계 질문 ("?" 2개 이상 또는 접속사)
    multi_step = query.count("?") > 1 or "그리고" in query or "하지만" in query

    return has_complex_keyword or (low_context and multi_step)
```

**판단 예시**:

| 쿼리 | 결과 | 이유 |
|------|------|------|
| "LANEIGE 순위 알려줘" | Direct | 단순 조회 |
| "LANEIGE 경쟁력 **분석**해줘" | ReAct | "분석" 키워드 |
| "순위 하락 **원인**과 개선 **전략**은?" | ReAct | "원인" + "전략" + 다단계 |
| "SoS가 뭐야?" | Direct | 정의 질문 |

### 6.4 ReAct Self-Reflection Loop

복잡한 쿼리에 대해 **Thought → Action → Observation → Reflection** 루프를 실행합니다.

```
┌────────────────────────────────────────────────┐
│                 ReAct Loop                      │
│                                                 │
│  Iteration 1:                                   │
│  ┌─────────┐                                    │
│  │ THOUGHT │ "LANEIGE 경쟁력을 분석하려면..."     │
│  └────┬────┘                                    │
│       ▼                                         │
│  ┌─────────┐                                    │
│  │ ACTION  │ query_knowledge_graph               │
│  │         │ entity="LANEIGE", relation="all"    │
│  └────┬────┘                                    │
│       ▼                                         │
│  ┌──────────────┐                               │
│  │ OBSERVATION  │ SoS=5.2%, competitors=[...]    │
│  └────┬─────────┘                               │
│       ▼                                         │
│  Iteration 2:                                   │
│  ┌─────────┐                                    │
│  │ THOUGHT │ "경쟁사 대비 포지션을 확인..."       │
│  └────┬────┘                                    │
│       ▼                                         │
│  ┌─────────┐                                    │
│  │ ACTION  │ calculate_metrics                   │
│  │         │ brand="LANEIGE", category="lip_care"│
│  └────┬────┘                                    │
│       ▼                                         │
│  ┌──────────────┐                               │
│  │ OBSERVATION  │ rank_trend=improving, ...       │
│  └────┬─────────┘                               │
│       ▼                                         │
│  Iteration 3:                                   │
│  ┌─────────┐                                    │
│  │ ACTION  │ final_answer                        │
│  │         │ "LANEIGE는 도전자 포지션으로..."     │
│  └────┬────┘                                    │
│       ▼                                         │
│  ┌──────────────┐                               │
│  │ REFLECTION   │ quality_score=0.85             │
│  │              │ needs_improvement=false         │
│  └──────────────┘                               │
└────────────────────────────────────────────────┘
```

**허용된 도구 (보안)**:

```python
ALLOWED_ACTIONS = frozenset({
    "query_data",              # SQLite 데이터 조회
    "query_knowledge_graph",   # KG 트리플 조회
    "calculate_metrics",       # KPI 계산
    "final_answer",            # 최종 답변
})
```

### 6.5 HybridChatbotAgent (챗봇)

```python
async def chat(self, user_message):
    # 1. 의도 라우팅
    route_result = self.router.route(user_message)

    # 2. 쿼리 리라이팅 (대화 맥락 반영)
    # "그 제품의 가격은?" → "LANEIGE Lip Sleeping Mask의 가격은?"
    rewrite_result = await self._maybe_rewrite_query(user_message)

    # 3. 하이브리드 검색
    hybrid_context = await self.hybrid_retriever.retrieve(
        query=rewrite_result.rewritten_query,
        current_metrics=self._current_data
    )

    # 4. 외부 신호 수집 (Tavily 뉴스, RSS, Reddit)
    external_signals = await self._collect_external_signals(query, entities)

    # 5. 컨텍스트 빌드
    context = self.context_builder.build(
        hybrid_context=hybrid_context,
        current_metrics=self._current_data,
        query=user_message,
        knowledge_graph=self.kg
    )

    # 6. LLM 응답 생성 (gpt-4.1-mini, temp=0.4)
    response = await self._generate_response(user_message, query_type, context)

    # 7. 응답 검증 (선택적)
    if self._enable_verification:
        verified = await self.verification_pipeline.verify(response)
```

### 6.6 HybridInsightAgent (인사이트 생성)

매일 크롤링 후 자동으로 실행되어 전략적 인사이트를 생성합니다.

```python
async def execute(self, metrics_data, crawl_data, crawl_summary):
    # 1. Knowledge Graph 업데이트 (크롤링 데이터 반영)
    kg_stats = self._update_knowledge_graph(crawl_data, metrics_data)

    # 2. Hybrid Retrieval + 추론
    hybrid_context = await self._run_hybrid_retrieval(metrics_data)
    inferences = hybrid_context.inferences

    # 3. RAG → KG 지식 추출 (문서에서 새로운 관계 발견)
    rag_kg_stats = self._ingest_rag_knowledge(hybrid_context.rag_chunks)

    # 4. 외부 신호 수집 (Tavily + RSS + Reddit)
    external_signals = await self._collect_external_signals()

    # 5. 4-Layer 시장 인텔리전스 구성
    market_intelligence = await self._collect_market_intelligence()

    # 6. LLM 인사이트 생성 (temp=0.6, 창의적)
    daily_insight = await self._generate_daily_insight(...)

    # 7. 액션 아이템 & 하이라이트 추출
    action_items = self._extract_action_items(inferences, metrics_data)
```

**4-Layer 원인 분석 프레임워크**:

```
Layer 4: 거시경제/무역
  → 관세청 수출입 데이터: 화장품 수출 전월비 +12.3%

Layer 3: 산업/기업 동향
  → 아모레퍼시픽 3Q 영업이익 +41% YoY

Layer 2: 소비자 트렌드
  → Reddit r/AsianBeauty 라네즈 언급 +34%
  → TikTok #LipSleepingMask 조회수 2.4M

Layer 1: Amazon 성과
  → Lip Care 카테고리 SoS: 8.2% (전주 7.1%)
```

---

## 7. End-to-End 쿼리 처리 파이프라인

### 예시: "LANEIGE Lip Care 경쟁력 분석해줘"

```
Step 1: UnifiedBrain.process_query()
  ├─ 캐시 미스 확인
  └─ 복잡도 판단: "분석" 키워드 → ReAct 모드

Step 2: EntityExtractor.extract()
  ├─ brands: ["laneige"]
  ├─ categories: ["lip_care"]
  └─ intent: DIAGNOSIS

Step 3: Knowledge Graph 조회
  ├─ brand_info: {sos: 0.052, avg_rank: 15.2}
  ├─ competitors: ["COSRX", "TIRTIR", "Burt's Bees"]
  ├─ products: [{asin: "B08XYZ", rank: 8, price: 29.99}]
  └─ hierarchy: {name: "Lip Care", level: 2, parent: "Skin Care"}

Step 4: Ontology 추론 (규칙 필터링: DIAGNOSIS 의도)
  ├─ Rule: challenger_position (SoS 5.2%, 순위 상승 중)
  │   → "도전자 포지션, 성장 잠재력 확인" (confidence: 0.85)
  └─ Rule: brand_driven (할인 없이 순위 유지)
      → "브랜드 파워형" (confidence: 0.80)

Step 5: RAG 문서 검색 (Playbook + Metric Guide 우선)
  ├─ "Metric Interpretation Guide.md" (score: 0.89)
  ├─ "아마존 랭킹 급등 원인 역추적 보고서.md" (score: 0.82)
  └─ "K-Beauty 미국 트렌드.md" (score: 0.75)

Step 6: ReAct Loop (3회 반복)
  ├─ Thought 1: "경쟁사 대비 포지션 확인 필요"
  │   Action: query_knowledge_graph → 경쟁사 SoS 데이터
  ├─ Thought 2: "시계열 트렌드 확인"
  │   Action: calculate_metrics → 최근 4주 추이
  └─ Thought 3: "종합 분석 완료"
      Action: final_answer

Step 7: Self-Reflection
  ├─ quality_score: 0.85
  └─ needs_improvement: false

Step 8: Context Assembly (우선순위 순)
  ├─ 1순위: 온톨로지 추론 결과 (challenger_position, brand_driven)
  ├─ 2순위: KG 팩트 (SoS, 경쟁사, 제품)
  ├─ 3순위: RAG 청크 (가이드라인, 플레이북)
  └─ 4순위: 외부 신호 (뉴스, 소셜)

Step 9: LLM 응답 생성 (gpt-4.1-mini, temp=0.4)
  → "LANEIGE는 Lip Care 카테고리에서 5.2% SoS로 도전자 포지션을 차지하고 있으며..."
```

---

## 8. 성능 최적화

### 8.1 캐싱 전략

| 캐시 | 대상 | TTL | 크기 | 절감 효과 |
|------|------|-----|------|-----------|
| **임베딩 캐시** | OpenAI 임베딩 | 무제한 (FIFO) | 1,000개 | API 비용 33%+ |
| **검색 캐시** | 벡터 검색 결과 | 300초 | 무제한 | 반복 쿼리 즉시 응답 |
| **엔티티 설정 캐시** | entities.json | 300초 | 1개 | 파일 I/O 절감 |
| **쿼리 리라이팅 캐시** | 리라이팅 결과 | 무제한 (FIFO) | 100개 | LLM 호출 절감 |
| **응답 캐시** | 최종 응답 | TTL 기반 | 가변 | LLM 호출 절감 |

### 8.2 검색 최적화

| 기법 | 설명 | 효과 |
|------|------|------|
| **2-Stage Retrieval** | Bi-Encoder(빠른 100개) → Cross-Encoder(정확한 5개) | 정밀도 향상 |
| **쿼리 확장** | LLM 기반 동의어/관련어 추가 | 재현율 향상 |
| **의도 기반 필터링** | 쿼리 의도에 맞는 문서 유형만 검색 | 노이즈 감소 |
| **시맨틱 청킹** | 문장 유사도 기반 분할 | 컨텍스트 일관성 |
| **테이블 보존** | MD 테이블 분리 방지 | 데이터 무결성 |

### 8.3 추론 최적화

| 기법 | 설명 | 효과 |
|------|------|------|
| **동적 규칙 필터링** | 쿼리 의도에 맞는 규칙만 실행 | 불필요한 추론 방지 |
| **우선순위 기반 실행** | 높은 우선순위 규칙 먼저 | 중요 인사이트 우선 |
| **컨텍스트 자동 보강** | KG에서 경쟁사/제품 자동 조회 | 수동 입력 불필요 |
| **스마트 Eviction** | 중요도 점수 기반 메모리 관리 | 메모리 50K 트리플 제한 |

### 8.4 비용 최적화

| 항목 | 전략 | 예상 절감 |
|------|------|-----------|
| **임베딩** | MD5 해시 캐시 + FIFO | 33%+ |
| **LLM 호출** | 응답 캐시 + 쿼리 리라이팅 캐시 | 20%+ |
| **토큰** | 컨텍스트 빌더 토큰 제한 (4,000) | 과도한 토큰 방지 |
| **추론** | 의도 기반 규칙 필터링 | 불필요한 LLM 호출 방지 |

---

## 부록: 주요 파일 맵

| 디렉토리 | 파일 | 핵심 클래스/함수 |
|----------|------|-----------------|
| `src/rag/` | `retriever.py` | `DocumentRetriever`, `search()`, `_embed_texts()` |
| | `hybrid_retriever.py` | `HybridRetriever`, `EntityExtractor`, `classify_intent()` |
| | `true_hybrid_retriever.py` | `TrueHybridRetriever`, `ConfidenceFusion` |
| | `chunker.py` | `SemanticChunker` |
| | `reranker.py` | `CrossEncoderReranker` |
| | `query_rewriter.py` | `QueryRewriter`, `needs_rewrite()` |
| | `entity_linker.py` | `EntityLinker`, `link()` |
| | `confidence_fusion.py` | `ConfidenceFusion`, `fuse()` |
| | `context_builder.py` | `ContextBuilder`, `build()` |
| `src/ontology/` | `knowledge_graph.py` | `KnowledgeGraph`, `query()`, `bfs_traverse()` |
| | `reasoner.py` | `OntologyReasoner`, `infer()`, `infer_with_intent()` |
| | `owl_reasoner.py` | `OWLReasoner`, `run_reasoner()` |
| | `business_rules.py` | 40+ `InferenceRule` 정의 |
| | `category_service.py` | `CategoryService`, `get_hierarchy()` |
| | `sentiment_service.py` | `SentimentService`, `get_product_sentiments()` |
| `src/agents/` | `hybrid_chatbot_agent.py` | `HybridChatbotAgent`, `chat()` |
| | `hybrid_insight_agent.py` | `HybridInsightAgent`, `execute()` |
| `src/core/` | `brain.py` | `UnifiedBrain`, `process_query()` |
| | `react_agent.py` | `ReActAgent`, `run()`, `_reflect()` |
